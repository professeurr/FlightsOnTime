import java.nio.file.{Files, Paths}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.io.Source

object Utils {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  var config: Configuration = initialize

  lazy val sparkSession: SparkSession = SparkSession.builder()
    .appName(s"FlightOnTime${scala.util.Random.nextInt()}")
    .getOrCreate()

  def initialize: Configuration = {
    // Import default formats
    implicit val formats: DefaultFormats.type = DefaultFormats

    logger.info("loading configuration...")
    val configFile = "config.json"
    val path = Paths.get(configFile).normalize()
    if (!Files.exists(path))
      throw new Exception(s"configuration file '${configFile}' does not exist. It should be placed in the working directory.")

    logger.info(s"configuration file '${path.toString}' is provided, let's initialize the session.")
    val f = Source.fromFile(configFile)
    // Parse config file into Configuration object
    val json = parse(f.getLines().mkString)
    config = json.camelizeKeys.extract[Configuration]
    f.close()

    val cores =
      if (config.numberOfCores <= 0)
        java.lang.Runtime.getRuntime.availableProcessors * Math.max(sparkSession.sparkContext.statusTracker.getExecutorInfos.length - 1, 1)
      else config.numberOfCores
    val partitions = Math.max(cores - 1, 1)
    logger.info(s"shuffle partitions: $partitions")
    sparkSession.conf.set("spark.sql.shuffle.partitions", partitions)
    logger.info("configuration loaded")
    config
  }

  def destroy(): Unit = {
    if (sparkSession != null) {
      sparkSession.close()
    }
  }

  def log(df: DataFrame, size: Int = 100): Unit = {
    //println(s"partitions: ${df.rdd.getNumPartitions}")
    //df.printSchema()
    //df.explain(false)
    //df.show(size, truncate = false)
  }

  def show(df: DataFrame, size: Int = 100): Unit = {
    //println(df.take(size).mkString("\n"))
    df.show(size, truncate = false)
  }
}
