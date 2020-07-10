import org.apache.log4j.Logger
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class FlightWeatherWrangling(flightWrangling: FlightWrangling, weatherWrangling: WeatherWrangling, weatherTimeFrame: Int, weatherTimeStep: Int) {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  import Utils.sparkSession.implicits._

  var Data: DataFrame = _

  def loadData(): DataFrame = {
    val tf: Int = 3600 * weatherTimeFrame

    logger.info("loading origin weather data")
    var OriginData = flightWrangling.Data.join(weatherWrangling.Data, $"ORIGIN_AIRPORT_ID" === $"AirportId", "inner")
      .drop("DEST_AIRPORT_ID", "ORIGIN_AIRPORT_ID", "AirportID", "FL_CRS_ARR_TIME")
      .filter(s"WEATHER_TIME >= FL_CRS_DEP_TIME - $tf and WEATHER_TIME <= FL_CRS_DEP_TIME")
    logger.info(OriginData)

    logger.info("building origin weather data")
    logger.info(s"OriginData.count() before filling weather ${OriginData.count()}")
    OriginData = OriginData.groupBy($"FL_ID", $"FL_CRS_DEP_TIME", $"FL_ONTIME")
      .agg(UtilUdfs.fillMissingDataUdf($"FL_CRS_DEP_TIME",
        collect_list($"WEATHER_TIME"), collect_list($"WEATHER_COND"), lit(weatherTimeFrame), lit(weatherTimeStep)).as("WEATHER_COND"))
      .filter("WEATHER_COND is not null")
      .drop("FL_CRS_DEP_TIME")
    logger.info(OriginData.schema.treeString)
    logger.info(s"OriginData.count() after filling weather ${OriginData.count()}")

    logger.info("loading destination weather data")
    var DestinationData = flightWrangling.Data.join(weatherWrangling.Data, $"DEST_AIRPORT_ID" === $"AirportId", "inner")
      .drop("DEST_AIRPORT_ID", "ORIGIN_AIRPORT_ID", "AirportID", "FL_CRS_DEP_TIME")
      .filter(s"WEATHER_TIME >= FL_CRS_ARR_TIME - $tf and WEATHER_TIME <= FL_CRS_ARR_TIME")
    logger.info(DestinationData.schema.treeString)

    logger.info("building destination weather data")
    logger.info(s"DestinationData.count() before filling weather ${DestinationData.count()}")
    DestinationData = DestinationData.groupBy($"FL_ID", $"FL_CRS_ARR_TIME", $"FL_ONTIME")
      .agg(UtilUdfs.fillMissingDataUdf($"FL_CRS_ARR_TIME",
        collect_list($"WEATHER_TIME"), collect_list($"WEATHER_COND"), lit(weatherTimeFrame), lit(weatherTimeStep)).as("WEATHER_COND"))
      .filter("WEATHER_COND is not null")
      .drop("FL_CRS_ARR_TIME")
    logger.info(DestinationData.schema.treeString)
    logger.info(s"DestinationData.count() after filling weather ${DestinationData.count()}")

    logger.info("Building final dataset with origin and destination weather conditions + on-time flag")
    Data = OriginData.as("origin").join(DestinationData.as("dest"), $"origin.FL_ID" === $"dest.FL_ID")
      .select($"origin.FL_ID".as("FL_ID"), $"origin.FL_ONTIME".as("FL_ONTIME"),
        UtilUdfs.toVectorUdf(concat($"origin.WEATHER_COND", $"dest.WEATHER_COND")).as("WEATHER_COND"))
    logger.info(Data.schema.treeString)

    Data = Data.cache() // cache the resulting data for being used during ML analysis

    // Utils.show(Data)
    Data
  }

}
