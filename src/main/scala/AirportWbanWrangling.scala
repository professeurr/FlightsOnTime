import org.apache.log4j.Logger
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.broadcast
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}

class AirportWbanWrangling(val path: String) {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  import Utils.sparkSession.implicits._

  val schema: StructType = StructType(Array(
    StructField("AirportID", StringType, nullable = true),
    StructField("WBAN", StringType, nullable = true),
    StructField("TimeZone", LongType, nullable = true)))

  var Data: DataFrame = _

  def loadData(): DataFrame = {
    logger.info(s"Loading relationship data between weather and airport data from $path")
    Data = Utils.sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .schema(schema)
      .load(path)
      .withColumn("TimeZone", $"TimeZone" * 3600L)
      .withColumnRenamed("WBAN", "JOIN_WBAN")

    Data = broadcast(Data) // broadcast this dataset which small compare to flights and weather ones. Broadcasting it will significantly speed up the join operations
    logger.info(Data)

    Data
  }
}
