import java.nio.file.{Files, Paths}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
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
    .master(config.cluster)
    .getOrCreate()

  def initialize: Configuration = {
    // Import default formats
    implicit val formats: DefaultFormats.type = DefaultFormats

    val path = Paths.get("./config.json").normalize()
    val configFile = path.toString
    if (!Files.exists(path))
      throw new Exception(s"configuration file '${configFile}' does not exist. It should be placed in the working directory or provided as argument.")

    Utils.log(s"configuration file '${path.toString}' is provided, let's initialize the session.")
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
    sparkSession.conf.set("spark.sql.shuffle.partitions", partitions)
    sparkSession.conf.set("spark.executor.cores", partitions)
    sparkSession.conf.set("spark.executor.instances", partitions)

    config
  }

  def destroy(): Unit = {
    if (sparkSession != null) {
      sparkSession.close()
    }
  }

  // fill missing weather records over frame-th hours before the flight departure.
  // in strict mode, for a particular missing data point, only the record the precedent record is used to fill up the gap
  // otherwise we use the closest record
  // in the strict mode, the computed record might miss some data points, in that case, null is returned.
  val fillMissingDataUdf: UserDefinedFunction = udf((originTime: Long, times: Seq[Long], weatherConds: Seq[Seq[Double]], frame: Int, step: Int) => {
    var cds: Seq[Double] = null
    val delta = step * 3600
    val enoughRecords = true //times.exists(t => t <= originTime - frame * 3600) //uncomment this to enable the strict filtering
    if (enoughRecords) {
      cds = List[Double]()
      var curTime = originTime
      for (_ <- 1 to frame) {
        val diff = times.map(t => Math.abs(curTime - t))
        val index = diff.indexOf(diff.min)
        cds ++= weatherConds(index)
        curTime -= delta
      }
    }
    cds
  })

  // convert array of double to dense vector in order to feed the regressor
  val toVectorUdf: UserDefinedFunction = udf((data: Seq[Double]) => {
    Vectors.dense(data.toArray)
  })

  val skyConditionCategoryUdf: UserDefinedFunction = udf((items: Seq[String]) => {
    items.toSet.toArray
  })

  val skyConditionPadValueUdf: UserDefinedFunction = udf((items: Seq[String]) => {
    if (items == null)
      (0 until 6).map(_ => "Z")
    else
      items ++ Seq.fill(5 - items.length)("Z")
  })

  def log(df: DataFrame, size: Int = 100): Unit = {
    //println(s"partitions: ${df.rdd.getNumPartitions}")
    df.printSchema()
    //df.explain(false)
    //df.show(size, truncate = false)
  }

  def show(df: DataFrame, size: Int = 100): Unit = {
    //println(df.take(size).mkString("\n"))
    df.show(size, truncate = false)
  }

  def log(str: String): Unit = {
    logger.info(s"$str")
  }

}
