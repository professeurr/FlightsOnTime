import org.apache.log4j.Logger
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, LongType}

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
      .drop("_c12")
      .na.fill("0.0", Array("ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY", "CANCELLED", "DIVERTED"))

    logger.info(Data.schema.treeString)
    logger.info(s"full flights dataset: ${Data.count()}")
    Data.show(truncate = false)

    logger.info("computing FL_ID")
    Data = Data.withColumn("FL_ID",
      concat_ws("_", $"OP_CARRIER_AIRLINE_ID", $"FL_DATE", $"ORIGIN_AIRPORT_ID", $"DEST_AIRPORT_ID", $"OP_CARRIER_FL_NUM"))
      .drop("OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM")
    logger.info(Data.schema.treeString)

    // remove cancelled and diverted data
    logger.info("Removing cancelled, diverted")
    Data = Data.filter("CANCELLED = 0 and DIVERTED = 0").drop("CANCELLED", "DIVERTED")
    logger.info(s"flights dataset without cancelled and diverted flights: ${Data.count()}")

    logger.info("Removing non-weather related delay records")
    Data = Data.filter(s"NOT(ARR_DELAY_NEW > $delayThreshold and  WEATHER_DELAY < $delayThreshold and NAS_DELAY < $delayThreshold)")
    logger.info(s"flights dataset without non-weather related delay records: ${Data.count()}")

    logger.info("computing FL_ONTIME")
    Data = Data.withColumn("FL_ONTIME", ($"ARR_DELAY_NEW" <= delayThreshold ||
      ($"WEATHER_DELAY" <= delayThreshold && $"NAS_DELAY" <= delayThreshold)).cast(IntegerType))
    logger.info(Data.schema.treeString)
    Data.show(truncate = false)
    Data = Data.drop("ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY")

    logger.info("Mapping flights with the weather wban and timezone")
    Data = Data.join(airportWbanWrangling.Data, $"ORIGIN_AIRPORT_ID" === $"AirportId", "inner")
      .drop("AirportId")

    logger.info(Data.schema.treeString)
    logger.info(s"flights data after the mapping: ${Data.count()}")
    Data.show(truncate = false)

    logger.info("computing FL_DATE")
    Data = Data.withColumn("FL_DATE", unix_timestamp(concat_ws("", $"FL_DATE", $"CRS_DEP_TIME"), "yyyy-MM-ddHHmm"))
      .drop("CRS_DEP_TIME")
    Data.show(truncate = false)

    logger.info("normalizing times to UTC: FL_CRS_DEP_TIME, FL_CRS_ARR_TIME")
    Data = Data.withColumn("FL_CRS_DEP_TIME", $"FL_DATE".minus($"TimeZone"))
    Data = Data.withColumn("FL_CRS_ARR_TIME", ($"FL_CRS_DEP_TIME" + $"CRS_ELAPSED_TIME").cast(LongType))
    logger.info(Data.schema.treeString)

    logger.info("selecting useful columns")
    Data = Data.select("FL_ID", "FL_ONTIME", "FL_CRS_DEP_TIME", "FL_CRS_ARR_TIME", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID")
    logger.info(Data.schema.treeString)
    Data.show(truncate = false)

    logger.info(s"ontime flights: ${Data.where("Fl_ONTIME = 1").count()}")
    logger.info(s"delayed flights: ${Data.where("Fl_ONTIME = 0").count()}")

    Data = Data.cache()

    Data
  }

}
