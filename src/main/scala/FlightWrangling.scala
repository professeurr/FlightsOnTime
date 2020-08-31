import org.apache.log4j.Logger
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, LongType}

class FlightWrangling(val path: String, val mappingData: DataFrame, val delayThreshold: Integer) {

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

    //logger.info(Data.schema.treeString)
    logger.info(s"full flights dataset: ${Data.count()}")
    //Data.show(truncate = false)

    logger.info("computing FL_ID")
    Data = Data.withColumn("FL_ID",
      concat_ws("_", $"OP_CARRIER_AIRLINE_ID", $"FL_DATE", $"ORIGIN_AIRPORT_ID", $"DEST_AIRPORT_ID", $"OP_CARRIER_FL_NUM"))
      .drop("OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM")
    //logger.info(Data.schema.treeString)

    // remove cancelled and diverted data
    logger.info("Removing cancelled, diverted")
    Data = Data.filter("CANCELLED + DIVERTED = 0").drop("CANCELLED", "DIVERTED")
    logger.info(s"flights dataset without cancelled and diverted flights: ${Data.count()}")

    logger.info("Removing non-weather related delay records")
    Data = Data.filter(s"ARR_DELAY_NEW <= $delayThreshold or  WEATHER_DELAY + NAS_DELAY >= $delayThreshold")
    logger.info(s"flights dataset without non-weather related delay records: ${Data.count()}")

    logger.info("computing FL_ONTIME")
    Data = Data.withColumn("FL_ONTIME", ($"ARR_DELAY_NEW" <= delayThreshold).cast(DoubleType))
      .drop("WEATHER_DELAY", "NAS_DELAY")

    logger.info("Mapping flights with the weather wban and timezone")
    Data = Data.join(mappingData, $"ORIGIN_AIRPORT_ID" === $"AirportId", "inner")
      .drop("AirportId")
    logger.info(s"flights data after the mapping: ${Data.count()}")

    logger.info("computing FL_DATE, FL_DEP_TIME and FL_ARR_TIME")
    Data = Data.withColumn("FL_DATE", unix_timestamp(concat_ws("", $"FL_DATE", $"DEP_TIME"), "yyyy-MM-ddHHmm"))
      .withColumn("FL_DEP_TIME", $"FL_DATE".minus($"TimeZone"))
      .withColumn("FL_ARR_TIME", ($"FL_DEP_TIME" + $"ACTUAL_ELAPSED_TIME" * 60).cast(LongType))

    logger.info("Flights cleaned data:")
    /*
    Data.select(
      $"ORIGIN_AIRPORT_ID", $"DEST_AIRPORT_ID",
      date_format(from_unixtime($"FL_DATE"), "yyyy-MM-dd HH:mm").as("FL_DATE"),
      ($"TimeZone" / 3600).as("TIMEZONE"),
      date_format(from_unixtime($"FL_DEP_TIME"), "yyyy-MM-dd HH:mm").as("FL_DEP_TIME"),
      ($"FL_ACTUAL_ELAPSED_TIME" / 60).as("FL_ACTUAL_ELAPSED_TIME"),
      date_format(from_unixtime($"FL_ARR_TIME"), "yyyy-MM-dd HH:mm").as("FL_ARR_TIME"),
      $"FL_ONTIME")
      .show(truncate = false)
*/
    Data = Data.select("FL_ID", "FL_DEP_TIME", "FL_ARR_TIME", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_ONTIME")
      .cache()

    //Data.show(truncate = false)
    Data.groupBy("FL_ONTIME").count().show(false)

    Data
  }

}
