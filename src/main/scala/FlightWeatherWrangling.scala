import org.apache.log4j.Logger
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class FlightWeatherWrangling(flightData: DataFrame, weatherData: DataFrame, config: Configuration) {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  import Utils.sparkSession.implicits._

  var Data: DataFrame = _

  def loadData(): DataFrame = {
    val tf: Int = 3600 * config.weatherTimeFrame

    logger.info("joining flights data with weather data")
    Data = flightData.join(weatherData.as("dep"), $"ORIGIN_AIRPORT_ID" === $"dep.AIRPORTID", "inner")
      .join(weatherData.as("arr"), $"DEST_AIRPORT_ID" === $"arr.AIRPORTID", "inner")
      .where(s"dep.WEATHER_TIME >= FL_DEP_TIME - $tf and dep.WEATHER_TIME <= FL_DEP_TIME " +
        s"and arr.WEATHER_TIME >= FL_ARR_TIME - $tf and arr.WEATHER_TIME <= FL_ARR_TIME")
      .drop("dep.AIRPORTID", "arr.AIRPORTID", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID")
      .groupBy($"FL_ID", $"FL_DEP_TIME", $"FL_ARR_TIME", $"FL_ONTIME")
      .agg(collect_list($"dep.WEATHER_TIME").as("ORIGIN_WEATHER_TIME"),
        collect_list($"dep.WEATHER_COND").as("ORIGIN_WEATHER_COND"),
        collect_list($"arr.WEATHER_TIME").as("DEST_WEATHER_TIME"),
        collect_list($"arr.WEATHER_COND").as("DEST_WEATHER_COND"))
      .drop("FL_ID")
    logger.info(s"flights data with weather data: ${Data.count()}")
    //Data.show(true)

    if (config.mlBalanceDataset) {
      logger.info("balancing dataset...")
      var ontimeFlights = Data.filter("Fl_ONTIME = 1.0")
      val ontimeFlightsCount = ontimeFlights.count().toDouble
      var delayedFlights = Data.filter("FL_ONTIME = 0.0")
      val delayedFlightsCount = delayedFlights.count().toDouble
      logger.info(s"ontimeFlightsCount=$ontimeFlightsCount, delayedFlightsCount=$delayedFlightsCount")
      if (ontimeFlightsCount > delayedFlightsCount)
        ontimeFlights = ontimeFlights.sample(withReplacement = false, delayedFlightsCount / ontimeFlightsCount, 42L)
      else
        delayedFlights = delayedFlights.sample(withReplacement = false, ontimeFlightsCount / delayedFlightsCount, 42L)

      Data = ontimeFlights.union(delayedFlights).cache()
      logger.info(s"data after balancing: ${Data.count()}")
    }

    logger.info("filling weather data...")
    Data = Data.withColumn("WEATHER_COND",
      UtilUdfs.fillWeatherDataUdf(
        $"FL_DEP_TIME", $"ORIGIN_WEATHER_TIME", $"ORIGIN_WEATHER_COND",
        $"FL_ARR_TIME", $"DEST_WEATHER_TIME", $"DEST_WEATHER_COND",
        lit(config.weatherTimeFrame), lit(config.weatherTimeStep)
      ))
      .drop("FL_DEP_TIME", "ORIGIN_WEATHER_TIME", "ORIGIN_WEATHER_COND",
        "FL_ARR_TIME", "DEST_WEATHER_TIME", "DEST_WEATHER_COND")
      .filter("WEATHER_COND is not null")
    logger.info(s"data after filling: ${Data.count()}")
    //Data.show(truncate = false)

    return Data

  }

}
