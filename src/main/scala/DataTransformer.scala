import UdfUtility.{assembleVectors, fillWeatherDataUdf}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode}

class DataTransformer(config: Configuration) {

  import Utility.sparkSession.implicits._

  def joinData(flightDataDep: DataFrame, flightDataArr: DataFrame, weatherData: DataFrame): DataFrame = {
    val tf: Int = 3600 * config.weatherTimeFrame
    val step: Int = 3600 * config.weatherTimeStep

    Utility.log("joining departure flights with eather conditions...")
    val depDta = flightDataDep.join(weatherData, Array("AIRPORTID"), "inner")
      .where(s"WEATHER_TIME >= FL_DEP_TIME - $tf and WEATHER_TIME <= FL_DEP_TIME ")
      .groupBy("FL_ID")
      .agg(first("delay").as("delay"),
        fillWeatherDataUdf(collect_list("WEATHER_COND"), lit(config.weatherTimeFrame), lit(step)).as("ORIGIN_WEATHER_COND"))
      .drop("WEATHER_COND", "WEATHER_TIME")
      .na.drop()

    Utility.log("joining arrival flights with eather conditions...")
    val arrData = flightDataArr.join(weatherData, Array("AIRPORTID"), "inner")
      .where(s"WEATHER_TIME >= FL_ARR_TIME - $tf and WEATHER_TIME <= FL_ARR_TIME ")
      .groupBy("FL_ID")
      .agg(fillWeatherDataUdf(collect_list("WEATHER_COND"), lit(config.weatherTimeFrame), lit(step)).as("DEST_WEATHER_COND"))
      .drop("WEATHER_COND", "WEATHER_TIME").na.drop()

    Utility.log("assembling departure and destination flights...")
    val data = depDta.join(arrData, Array("FL_ID"), "inner")
      .withColumn("features", assembleVectors($"ORIGIN_WEATHER_COND", $"DEST_WEATHER_COND"))
      .select("features", "delay")

    val outputPath = config.persistPath + config.mlMode + "/data"
    Utility.log(s"saving train dataset into $outputPath")
    data.repartition(config.partitions)
      .write.mode(SaveMode.Overwrite)
      .parquet(outputPath)
    Utility.log(s"reading persisted train dataset from $outputPath")

    Utility.readParquet(outputPath)
  }
}
