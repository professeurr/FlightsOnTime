import UdfUtility.{assembleVectors, fillWeatherDataUdf}
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
    val s = config.flightsTrainPath.mkString(",")
    Utility.log(s"Loading flights records from $s")
    var data = Utility.sparkSession.read.parquet(config.flightsTrainPath: _*)

    Utility.log("Removing non-weather related delayed records...")
    data = data.filter(s"ARR_DELAY_NEW <= ${config.flightsDelayThreshold} " +
      s"or  (WEATHER_DELAY + NAS_DELAY >= ${config.flightsDelayThreshold})")

    val x = data.select("FL_ID", "ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY")
    Utility.show(x.filter("WEATHER_DELAY > 0").limit(10).union(x.filter("NAS_DELAY > 0").limit(10)))

    Utility.log("computing FL_ONTIME flag (1=on-time; 0=delayed)")
    data = data.withColumn("FL_ONTIME", when($"ARR_DELAY_NEW" <= config.flightsDelayThreshold, 1.0).otherwise(0.0))
    val y = data.select("FL_ID", "FL_ONTIME", "ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY")
    Utility.show(y.filter("FL_ONTIME = 0").limit(10).union(y.filter("FL_ONTIME = 1").limit(10)))
    Utility.show(y.select("FL_ONTIME").groupBy("FL_ONTIME").count())
    data = data.drop("WEATHER_DELAY", "NAS_DELAY")

    data
  }

  def loadWeatherData(): DataFrame = {
    Utility.log(s"Loading weather records from ${config.weatherTrainPath.mkString(",")}")
    var data = Utility.sparkSession.read.parquet(config.weatherTrainPath: _*)
    Utility.log(s"weather initial number of records: ${Utility.count(data)};")
    Utility.show(data)

    data = data.select("AIRPORTID", "WEATHER_COND", "WEATHER_TIME")
    Utility.show(data)

    data
  }

  def combineData(flightData: DataFrame, weatherData: DataFrame): DataFrame = {
    val tf: Int = 3600 * config.weatherTimeFrame
    Utility.log("joining flights data with weather data")

    Utility.log("joining departure and arrival flights with eather conditions...")
    var depDta = flightData.join(weatherData, $"ORIGIN_AIRPORT_ID" === $"AIRPORTID", "inner")
      .where(s"WEATHER_TIME >= FL_DEP_TIME - $tf and WEATHER_TIME <= FL_DEP_TIME ")
      .drop("AIRPORTID", "ORIGIN_AIRPORT_ID", "FL_ARR_TIME", "FL_ONTIME")
    depDta = depDta.groupBy("FL_ID", "FL_DEP_TIME")
      .agg(fillWeatherDataUdf($"FL_DEP_TIME", collect_list("WEATHER_TIME"), collect_list("WEATHER_COND"),
        lit(config.weatherTimeFrame), lit(config.weatherTimeStep)).as("ORIGIN_WEATHER_COND"))
      .drop("WEATHER_COND", "WEATHER_TIME", "FL_DEP_TIME")
      .filter("ORIGIN_WEATHER_COND is not null")

    var arrData = flightData.join(weatherData, $"DEST_AIRPORT_ID" === $"AIRPORTID", "inner")
      .where(s"WEATHER_TIME >= FL_ARR_TIME - $tf and WEATHER_TIME <= FL_ARR_TIME ")
      .drop("AIRPORTID", "DEST_AIRPORT_ID", "FL_DEP_TIME")
    arrData = arrData.groupBy("FL_ID", "FL_ARR_TIME")
      .agg(first("FL_ONTIME").as("FL_ONTIME"),
        fillWeatherDataUdf($"FL_ARR_TIME", collect_list("WEATHER_TIME"), collect_list("WEATHER_COND"),
          lit(config.weatherTimeFrame), lit(config.weatherTimeStep)).as("DEST_WEATHER_COND"))
      .drop("WEATHER_COND", "WEATHER_TIME", "FL_ARR_TIME")
      .filter("DEST_WEATHER_COND is not null")

    Utility.log("assembling departure and destination flights...")
    var data = depDta.join(arrData, Array("FL_ID"), "inner")
      .withColumn("WEATHER_COND", assembleVectors($"ORIGIN_WEATHER_COND", $"DEST_WEATHER_COND"))
      .drop("ORIGIN_WEATHER_COND", "DEST_WEATHER_COND")

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
    data.repartition(config.partitions, col("FL_ONTIME"))
      .write.mode(SaveMode.Overwrite)
      .partitionBy("FL_ONTIME")
      .parquet(outputPath + "2")
    data = Utility.sparkSession.read.parquet(outputPath + "2")
    Utility.log("balanced")

    data
  }
}
