import UdfUtility.{assembleVectors, fillWeatherDataUdf}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode}

class DataTransformer(config: Configuration) {

  import Utility.sparkSession.implicits._

  def transformFlightData(): (DataFrame, DataFrame) = {
    val s = config.flightsPath.mkString(",")
    Utility.log(s"Loading flights records from $s")
    var data = Utility.sparkSession.read.parquet(config.flightsPath: _*)

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

    val depData = data.select("FL_ID", "ORIGIN_AIRPORT_ID", "FL_DEP_TIME")
    val destData = data.select("FL_ID", "FL_ONTIME", "DEST_AIRPORT_ID", "FL_ARR_TIME")
    val depOutputPath = config.persistPath + config.mlMode + "/flights.dep"
    val destOutputPath = config.persistPath + config.mlMode + "/flights.dest"

    Utility.log(s"saving departure flights dataset into $depOutputPath")
    depData.repartition(config.partitions, col("ORIGIN_AIRPORT_ID"))
      .write.mode(SaveMode.Overwrite)
      .partitionBy("ORIGIN_AIRPORT_ID")
      .parquet(depOutputPath)

    Utility.log(s"saving arrival flights dataset into $destOutputPath")
    destData.repartition(config.partitions, col("DEST_AIRPORT_ID"))
      .write.mode(SaveMode.Overwrite)
      .partitionBy("DEST_AIRPORT_ID")
      .parquet(destOutputPath)

    (Utility.sparkSession.read.parquet(depOutputPath),
      Utility.sparkSession.read.parquet(destOutputPath))
  }

  def transformWeatherData(): DataFrame = {
    Utility.log(s"Loading weather records from ${config.weatherPath.mkString(",")}")
    var data = Utility.sparkSession.read.parquet(config.weatherPath: _*)
    Utility.log(s"weather initial number of records: ${Utility.count(data)};")
    Utility.show(data)

    data = data.select("W_ID", "AIRPORTID", "WEATHER_COND", "WEATHER_TIME")
    Utility.show(data)

    val outputPath = config.persistPath + config.mlMode + "/weather.bin"
    Utility.log(s"saving weather dataset into $outputPath")
    data.repartition(config.partitions, col("AIRPORTID"))
      .write.mode(SaveMode.Overwrite)
      .partitionBy("AIRPORTID")
      .parquet(outputPath)

    Utility.sparkSession.read.parquet(outputPath)
  }

  def joinData(flightDepData: DataFrame, flightArrData: DataFrame, weatherData: DataFrame): DataFrame = {
    val tf: Int = 3600 * config.weatherTimeFrame
    Utility.log("joining flights data with weather data")

    Utility.log("joining departure and arrival flights with eather conditions...")
    val depDta = flightDepData.join(weatherData, $"ORIGIN_AIRPORT_ID" === $"AIRPORTID", "inner")
      .where(s"WEATHER_TIME >= FL_DEP_TIME - $tf and WEATHER_TIME <= FL_DEP_TIME ")
      .drop("AIRPORTID", "ORIGIN_AIRPORT_ID")
      .groupBy("FL_ID", "FL_DEP_TIME")
      .agg(fillWeatherDataUdf($"FL_DEP_TIME", collect_list("WEATHER_TIME"), collect_list("W_ID"),
        lit(config.weatherTimeFrame), lit(config.weatherTimeStep)).as("W_ID"))
      .filter("W_ID is not null")
      .select($"FL_ID", explode($"W_ID").as("W_ID"))
      .join(weatherData, Array("W_ID"), "inner")
      .groupBy("FL_ID").agg(collect_list("WEATHER_COND").as("ORIGIN_WEATHER_COND")).drop("W_ID")

    //      .agg(fillWeatherDataUdf($"FL_DEP_TIME", collect_list("WEATHER_TIME"), collect_list("WEATHER_COND"),
    //        lit(config.weatherTimeFrame), lit(config.weatherTimeStep)).as("ORIGIN_WEATHER_COND"))
    //      .drop("WEATHER_COND", "WEATHER_TIME", "FL_DEP_TIME")
    //      .filter("ORIGIN_WEATHER_COND is not null")

    val arrData = flightArrData.join(weatherData, $"DEST_AIRPORT_ID" === $"AIRPORTID", "inner")
      .where(s"WEATHER_TIME >= FL_ARR_TIME - $tf and WEATHER_TIME <= FL_ARR_TIME ")
      .drop("AIRPORTID", "DEST_AIRPORT_ID")
      .groupBy("FL_ID", "FL_ARR_TIME")
      .agg(first("FL_ONTIME").as("FL_ONTIME"),
        fillWeatherDataUdf($"FL_ARR_TIME", collect_list("WEATHER_TIME"), collect_list("W_ID"),
          lit(config.weatherTimeFrame), lit(config.weatherTimeStep)).as("W_ID"))
      .filter("W_ID is not null")
      .select($"FL_ID", $"FL_ONTIME", explode($"W_ID").as("W_ID"))
      .join(weatherData, Array("W_ID"), "inner")
      .groupBy("FL_ID").agg(first("FL_ONTIME").as("FL_ONTIME"), collect_list("WEATHER_COND").as("DEST_WEATHER_COND"))
      .drop("W_ID")
    //      .drop("WEATHER_COND", "WEATHER_TIME", "FL_ARR_TIME")
    //      .filter("DEST_WEATHER_COND is not null")
    //arrData.show(truncate = false)

    //Utility.exit()
    Utility.log("assembling departure and destination flights...")
    var data = depDta.join(arrData, Array("FL_ID"), "inner")
      .withColumn("WEATHER_COND", assembleVectors($"ORIGIN_WEATHER_COND", $"DEST_WEATHER_COND"))
      .select("FL_ID", "WEATHER_COND", "FL_ONTIME")

    Utility.show(data, truncate = true)

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
