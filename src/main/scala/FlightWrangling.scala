import org.apache.log4j.Logger
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

class FlightWrangling(val path: String, val airportWbanWrangling: AirportWbanWrangling, val delayThreshold: Integer) {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  import Utils.sparkSession.implicits._

  var Data: DataFrame = _

  def loadData(): DataFrame = {
    logger.info("Loading flights data")
    Data = Utils.sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load(path)
    //.na.fill("0", Array("ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY", "CANCELLED", "DIVERTED"))
    logger.info(Data)

    // remove cancelled and diverted data
    logger.info("Removing cancelled, diverted and non-weather related delay records")
    Data = Data.filter("CANCELLED <> \"1\" or DIVERTED <> \"1\"")
      .filter(s"NOT(ARR_DELAY_NEW > $delayThreshold and  WEATHER_DELAY < $delayThreshold and NAS_DELAY < $delayThreshold)")
      .drop("CANCELLED", "DIVERTED")
    logger.info(Data)

    logger.info("computing FL_ID")
    Data = Data.withColumn("FL_DATE", unix_timestamp(concat_ws("", $"FL_DATE", $"CRS_DEP_TIME"), "yyyy-MM-ddHHmm"))
      .drop("_c12", "CRS_DEP_TIME")

    Data = Data.withColumn("FL_ID",
      concat_ws("_", $"OP_CARRIER_AIRLINE_ID", $"FL_DATE", $"ORIGIN_AIRPORT_ID", $"DEST_AIRPORT_ID", $"OP_CARRIER_FL_NUM"))
      .drop("OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM")
    logger.info(Data)

    logger.info("computing FL_ONTIME")
    Data = Data.withColumn("FL_ONTIME", (!($"ARR_DELAY_NEW" > delayThreshold &&
      ($"WEATHER_DELAY" > delayThreshold || $"NAS_DELAY" > delayThreshold))).cast(IntegerType))
      .drop("ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY")
    logger.info(Data)

    logger.info("getting timezones of each airport")
    Data = Data.join(airportWbanWrangling.Data, $"ORIGIN_AIRPORT_ID" === $"AirportId", "inner")
      .drop("AirportId", "JOIN_WBAN")
    logger.info(Data)

    logger.info("normalizing times to UTC: FL_CRS_DEP_TIME, FL_CRS_ARR_TIME")
    Data = Data.withColumn("FL_CRS_DEP_TIME", $"FL_DATE".minus($"TimeZone"))
      .withColumn("FL_CRS_ARR_TIME", $"FL_CRS_DEP_TIME" + $"CRS_ELAPSED_TIME")
    logger.info(Data)

    logger.info("selecting useful columns")
    Data = Data.select("FL_ID", "FL_ONTIME", "FL_CRS_DEP_TIME", "FL_CRS_ARR_TIME", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID")
    logger.info(Data)

    Data = Data.cache()

    Data
  }

}
