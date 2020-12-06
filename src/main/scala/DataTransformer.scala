import UdfUtility.{assembleVectors, fillWeatherDataUdf}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode}

class DataTransformer(config: Configuration) {

  import Utility.sparkSession.implicits._

  def transformFlightData(): DataFrame = {
    val s = config.flightsPath.mkString(",")
    Utility.log(s"Loading flights records from $s")
    var data = Utility.sparkSession.read.parquet(config.flightsPath: _*)

    Utility.log("Removing non-weather related delayed records...")
    data = data.filter(s"ARR_DELAY_NEW <= ${config.flightsDelayThreshold} " +
      s"or (DELAY >= ${config.flightsDelayThreshold})")
      .drop("ARR_DELAY_NEW")

    Utility.log("computing FL_ONTIME flag (1=on-time; 0=delayed)")
    data = data.withColumn("FL_ONTIME", when($"DELAY" <= config.flightsDelayThreshold, 1.0).otherwise(0.0))


    val outputPath = config.persistPath + config.mlMode + "/flights"
    Utility.log(s"saving departure flights dataset into $outputPath")
    data.repartition(config.partitions)
      .write.mode(SaveMode.Overwrite)
      .parquet(outputPath)

    Utility.sparkSession.read.parquet(outputPath)
  }

  def transformWeatherData(): DataFrame = {
    Utility.log(s"Loading weather records from ${config.weatherPath.mkString(",")}")
    var data = Utility.sparkSession.read.parquet(config.weatherPath: _*)
    Utility.log(s"weather initial number of records: ${Utility.count(data)};")
    Utility.show(data)

    data = data.select("AIRPORTID", "WEATHER_COND", "WEATHER_TIME")
    Utility.show(data)

    val outputPath = config.persistPath + config.mlMode + "/weather.bin"
    Utility.log(s"saving weather dataset into $outputPath")
    data.repartition(config.partitions, col("AIRPORTID"))
      .write.mode(SaveMode.Overwrite)
      .partitionBy("AIRPORTID")
      .parquet(outputPath)

    Utility.sparkSession.read.parquet(outputPath)
  }

  def joinData(flightData: DataFrame, weatherData: DataFrame): DataFrame = {
    val tf: Int = 3600 * config.weatherTimeFrame
    val step : Int = 3600 * config.weatherTimeStep
    Utility.log("joining flights data with weather data")

    Utility.log("joining departure and arrival flights with eather conditions...")
    val depDta = flightData.join(weatherData, $"ORIGIN_AIRPORT_ID" === $"AIRPORTID", "inner")
      .where(s"WEATHER_TIME >= FL_DEP_TIME - $tf and WEATHER_TIME <= FL_DEP_TIME ")
      .drop("AIRPORTID", "ORIGIN_AIRPORT_ID", "FL_ARR_TIME", "FL_ONTIME", "DELAY")
      .groupBy("FL_ID", "FL_DEP_TIME")
      .agg(fillWeatherDataUdf($"FL_DEP_TIME", collect_list("WEATHER_TIME"), collect_list("WEATHER_COND"),
        lit(config.weatherTimeFrame), lit(step)).as("ORIGIN_WEATHER_COND"))
      .drop("WEATHER_COND", "WEATHER_TIME", "FL_DEP_TIME")
      .filter("ORIGIN_WEATHER_COND is not null")

    val arrData = flightData.join(weatherData, $"DEST_AIRPORT_ID" === $"AIRPORTID", "inner")
      .where(s"WEATHER_TIME >= FL_ARR_TIME - $tf and WEATHER_TIME <= FL_ARR_TIME ")
      .drop("AIRPORTID", "DEST_AIRPORT_ID", "FL_DEP_TIME")
      .groupBy("FL_ID", "FL_ARR_TIME")
      .agg(first("FL_ONTIME").as("FL_ONTIME"), first("DELAY").as("DELAY"),
        fillWeatherDataUdf($"FL_ARR_TIME", collect_list("WEATHER_TIME"), collect_list("WEATHER_COND"),
          lit(config.weatherTimeFrame), lit(step)).as("DEST_WEATHER_COND"))
      .drop("WEATHER_COND", "WEATHER_TIME", "FL_ARR_TIME")
      .filter("DEST_WEATHER_COND is not null")

    Utility.log("assembling departure and destination flights...")
    var data = depDta.join(arrData, Array("FL_ID"), "inner")
      .withColumn("WEATHER_COND", assembleVectors($"ORIGIN_WEATHER_COND", $"DEST_WEATHER_COND"))
      .select("FL_ID", "WEATHER_COND", "FL_ONTIME", "DELAY")

    val outputPath = config.persistPath + config.mlMode + "/data"
    Utility.log(s"saving train dataset into $outputPath")
    data.repartition(config.partitions, col("FL_ONTIME"))
      .write.mode(SaveMode.Overwrite)
      .partitionBy("FL_ONTIME")
      .parquet(outputPath)
    Utility.log(s"reading persisted train dataset from $outputPath")
    data = Utility.sparkSession.read.parquet(outputPath)

    val ontimeCount = Utility.sparkSession.read.parquet(outputPath + "/FL_ONTIME=1.0").count().toDouble
    val delayedCount = Utility.sparkSession.read.parquet(outputPath + "/FL_ONTIME=0.0").count()
    Utility.log(s"ontime=$ontimeCount, delayed=$delayedCount")

    //if (config.mlMode.equalsIgnoreCase("transform")) {
    Utility.log("balancing...")
    val fractions = if (ontimeCount >= delayedCount) Map(0.0 -> 1.0, 1.0 -> delayedCount / ontimeCount) else Map(0.0 -> ontimeCount / delayedCount, 1.0 -> 1.0)
    data = data.stat.sampleBy("FL_ONTIME", fractions, 42L)
    data.repartition(config.partitions, col("FL_ONTIME"))
      .write.mode(SaveMode.Overwrite)
      .partitionBy("FL_ONTIME")
      .parquet(outputPath + ".balanced")

    Utility.log(s"saving balanced dataset into $outputPath")
    data = Utility.sparkSession.read.parquet(outputPath + ".balanced")

    Utility.log("balanced")
    //}

    data
  }
}
