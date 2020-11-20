import UdfUtility.fillWeatherDataUdf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SaveMode}

class DataLoader(config: Configuration) {

  import Utility.sparkSession.implicits._

  def loadStationsData(): DataFrame = {
    val schema: StructType = StructType(Array(
      StructField("AirportID", StringType, nullable = true),
      StructField("WBAN", StringType, nullable = true),
      StructField("TimeZone", LongType, nullable = true)))
    Utility.log(s"Loading weather stations ${config.wbanAirportsPath}")
    val data = Utility.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "false").schema(schema)
      .load(config.wbanAirportsPath)
      .withColumn("TimeZone", $"TimeZone" * 3600L)
      .withColumnRenamed("WBAN", "STATION_WBAN")
    Utility.show(data)
    data
  }

  def loadFlightData(): DataFrame = {
    val s = config.flightsPath.mkString(",")
    Utility.log(s"Loading flights records from $s")
    var data = Utility.sparkSession.read.parquet(config.flightsPath: _*)

    Utility.log("Removing non-weather related delayed records...")
    data = data.filter(s"ARR_DELAY_NEW <= ${config.flightsDelayThreshold} " +
      s"or  (WEATHER_DELAY + NAS_DELAY >= ${config.flightsDelayThreshold})")
    val x = data.select("FL_ID", "ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY")
    Utility.show(x.filter("WEATHER_DELAY > 0").limit(10).union(x.filter("NAS_DELAY > 0").limit(10)))
    Utility.log(s"number of flights: ${data.count()}")

    Utility.log("computing FL_ONTIME flag (1=on-time; 0=delayed)")
    data = data.withColumn("FL_ONTIME", ($"ARR_DELAY_NEW" <= config.flightsDelayThreshold).cast(DoubleType))
    val y = data.select("FL_ID", "FL_ONTIME", "ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY")
    Utility.show(y.filter("FL_ONTIME = 0").limit(10).union(y.filter("FL_ONTIME = 1").limit(10)))
    Utility.show(y.select("FL_ONTIME").groupBy("FL_ONTIME").count())
    data = data.drop("WEATHER_DELAY", "NAS_DELAY")

    //data = balanceDataset(data)

    data
  }

  def loadWeatherData(): DataFrame = {
    Utility.log(s"Loading weather records from ${config.weatherPath.mkString(",")}")
    var data = Utility.sparkSession.read.parquet(config.weatherPath: _*)
    Utility.log(s"weather initial number of records: ${Utility.count(data)};")
    Utility.show(data)

    data = data.select("AIRPORTID", "WEATHER_COND", "WEATHER_TIME")
    Utility.show(data)

    data
  }

  def combineData(flightData: DataFrame, weatherData: DataFrame): DataFrame = {
    val tf: Int = 3600 * config.weatherTimeFrame
    Utility.log("joining flights data with weather data")

    Utility.log("joining dep...")
    val depData = flightData.select("FL_ID", "FL_DEP_TIME", "ORIGIN_AIRPORT_ID", "FL_ONTIME")
      .join(weatherData.as("dep"), $"ORIGIN_AIRPORT_ID" === $"dep.AIRPORTID", "inner")
      .where(s"dep.WEATHER_TIME >= FL_DEP_TIME - $tf and dep.WEATHER_TIME <= FL_DEP_TIME ")
      .drop("dep.AIRPORTID", "ORIGIN_AIRPORT_ID")

    Utility.log("repartitioning arr...")
    val arrData = flightData.select("FL_ID", "FL_ARR_TIME", "DEST_AIRPORT_ID", "FL_ONTIME")
      .join(weatherData.as("arr"), $"DEST_AIRPORT_ID" === $"arr.AIRPORTID", "inner")
      .where(s"arr.WEATHER_TIME >= FL_ARR_TIME - $tf and arr.WEATHER_TIME <= FL_ARR_TIME ")
      .drop("arr.AIRPORTID", "ORIGIN_AIRPORT_ID", "FL_ONTIME")

    var data = depData.join(arrData, Array("FL_ID"), "inner")

    Utility.log("collecting conditions...")
    data = data.groupBy("FL_ID")
      .agg(first($"FL_DEP_TIME").as("FL_DEP_TIME"),
        first($"FL_ARR_TIME").as("FL_ARR_TIME"),
        first($"FL_ONTIME").as("FL_ONTIME"),
        collect_list($"dep.WEATHER_TIME").as("ORIGIN_WEATHER_TIME"),
        collect_list($"dep.WEATHER_COND").as("ORIGIN_WEATHER_COND"),
        collect_list($"arr.WEATHER_TIME").as("DEST_WEATHER_TIME"),
        collect_list($"arr.WEATHER_COND").as("DEST_WEATHER_COND")
      )

    Utility.log("filling missing weather records for each flight ...")
    data = data.withColumn("WEATHER_COND",
      fillWeatherDataUdf(
        $"FL_DEP_TIME", $"ORIGIN_WEATHER_TIME", $"ORIGIN_WEATHER_COND",
        $"FL_ARR_TIME", $"DEST_WEATHER_TIME", $"DEST_WEATHER_COND",
        lit(config.weatherTimeFrame), lit(config.weatherTimeStep)
      ))
      .drop("FL_DEP_TIME", "ORIGIN_WEATHER_TIME", "ORIGIN_WEATHER_COND",
        "FL_ARR_TIME", "DEST_WEATHER_TIME", "DEST_WEATHER_COND")
      .filter("WEATHER_COND is not null")
    //Utility.log(s"data after filling: ${Utility.count(data)}")
    Utility.show(data, truncate = true)

    val outputPath = config.persistPath + "data.train"
    Utility.log(s"saving train dataset into $outputPath")
    data.repartition(config.partitions, col("FL_ONTIME"))
      .write.mode(SaveMode.Overwrite)
      .partitionBy("FL_ONTIME")
      .parquet(outputPath)
    Utility.log(s"reading persisted train dataset from $outputPath")
    data = Utility.sparkSession.read.parquet(outputPath)

    Utility.log("balancing...")
    val ontimeCount = Utility.sparkSession.read.parquet(outputPath + "/FL_ONTIME=1.0").count().toDouble
    val delayedCount = Utility.sparkSession.read.parquet(outputPath + "/FL_ONTIME=0.0").count()
    Utility.log(s"ontime=$ontimeCount, delayed=$delayedCount")
    val fractions = if (ontimeCount >= delayedCount) Map(0.0 -> 1.0, 1.0 -> delayedCount / ontimeCount) else Map(0.0 -> ontimeCount / delayedCount, 1.0 -> 1.0)
    data = data.stat.sampleBy("FL_ONTIME", fractions, 42L)
    Utility.log("balanced")

    data
  }
}
