import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class FlightWeatherWrangling(flightWrangling: FlightWrangling, weatherWrangling: WeatherWrangling, weatherTimeFrame: Int) {

  import Utils.sparkSession.implicits._

  var Data: DataFrame = _

  def loadData(): DataFrame = {
    val tf: Int = 3600 * weatherTimeFrame

    Utils.log("loading origin weather data")
    var OriginData = flightWrangling.Data.join(weatherWrangling.Data, $"ORIGIN_AIRPORT_ID" === $"AirportId", "inner")
      .drop("DEST_AIRPORT_ID", "ORIGIN_AIRPORT_ID", "AirportID", "FL_CRS_ARR_TIME")
      .filter(s"WEATHER_TIME >= FL_CRS_DEP_TIME - $tf and WEATHER_TIME <= FL_CRS_DEP_TIME")
    Utils.log(OriginData)

    Utils.log("building origin weather data")
    OriginData = OriginData.groupBy($"FL_ID", $"FL_CRS_DEP_TIME", $"FL_ONTIME")
      .agg(Utils.fillMissingDataUdf($"FL_CRS_DEP_TIME",
        collect_list($"WEATHER_TIME"), collect_list($"WEATHER_COND"), lit(weatherTimeFrame)).as("WEATHER_COND"))
      .filter("WEATHER_COND is not null")
      .drop("FL_CRS_DEP_TIME")
    Utils.log(OriginData)

    Utils.log("loading destination weather data")
    var DestinationData = flightWrangling.Data.join(weatherWrangling.Data, $"DEST_AIRPORT_ID" === $"AirportId", "inner")
      .drop("DEST_AIRPORT_ID", "ORIGIN_AIRPORT_ID", "AirportID", "FL_CRS_DEP_TIME")
      .filter(s"WEATHER_TIME >= FL_CRS_ARR_TIME - $tf and WEATHER_TIME <= FL_CRS_ARR_TIME")
    Utils.log(DestinationData)

    Utils.log("building destination weather data")
    DestinationData = DestinationData.groupBy($"FL_ID", $"FL_CRS_ARR_TIME", $"FL_ONTIME")
      .agg(Utils.fillMissingDataUdf($"FL_CRS_ARR_TIME",
        collect_list($"WEATHER_TIME"), collect_list($"WEATHER_COND"), lit(weatherTimeFrame)).as("WEATHER_COND"))
      .filter("WEATHER_COND is not null")
      .drop("FL_CRS_ARR_TIME")
    Utils.log(DestinationData)

    Utils.log("Building final dataset with origin and destion weather conditions + on-time flag")
    Data = OriginData.as("origin").join(DestinationData.as("dest"), $"origin.FL_ID" === $"dest.FL_ID")
      .select($"origin.FL_ID".as("FL_ID"), $"origin.FL_ONTIME".as("FL_ONTIME"),
        Utils.toVectorUdf(concat($"origin.WEATHER_COND", $"dest.WEATHER_COND")).as("WEATHER_COND"))
    Utils.log(Data)

    Data = Data.cache() // cache the resulting data for being used during ML analysis

    // Utils.show(Data)
    Data
  }
}
