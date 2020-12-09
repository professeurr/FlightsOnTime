import java.nio.file.{Files, Paths}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.io.Source


object Utility {

  @transient lazy val logger: Logger = Logger.getLogger("$")

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)


  var config: Configuration = initialize

  lazy val sparkSession: SparkSession = SparkSession.builder()
    .appName(s"FlightOnTime${scala.util.Random.nextInt()}")
    //.config("spark.sql.autoBroadcastJoinThreshold", -1)
    .getOrCreate()

  def initialize: Configuration = {
    Utility.log("Initializing...")
    if (sparkSession == null) {
      logger.error("Spark session was not created.")
      throw new Exception("Spark session is not created.")
    }
    // Import default formats
    implicit val formats: DefaultFormats.type = DefaultFormats

    Utility.log("loading configuration...")
    val configFile = "config.json"
    val path = Paths.get(configFile).normalize()
    if (!Files.exists(path))
      throw new Exception(s"configuration file '$configFile' does not exist. It should be placed in the working directory.")

    Utility.log(s"configuration file '${path.toString}' is provided, let's initialize the session.")
    val f = Source.fromFile(configFile)
    // Parse config file into Configuration object
    val json = parse(f.getLines().mkString)
    f.close()

    json.camelizeKeys.extract[Configuration].init()
  }

  def destroy(): Unit = {
    if (sparkSession != null)
      sparkSession.close()
  }

  def log(s: Any): Unit = {
    logger.info(s)
  }

  def show(data: DataFrame, truncate: Boolean = false): Unit = {
    if (config.verbose)
      data.show(truncate = truncate)
  }

  def count(data: DataFrame): String = {
    if (config.verbose) data.count().toString else "---"
  }

  def exit(): Unit = {
    scala.sys.exit(0)
  }

  def time[R](msg: String, block: => R): R = {
    val t0 = System.currentTimeMillis()
    val result = block // call-by-name
    val t1 = System.currentTimeMillis()
    println(s"$msg elapsed time: ${(t1 - t0) / 1000} s")
    result
  }

  def readParquet(path: String*): DataFrame = {
    Utility.sparkSession.read.option("inferSchema", "false").parquet(path: _*)
  }

  def readCsv(path: String*): DataFrame = {
    Utility.sparkSession.read.option("header", "true").option("inferSchema", "false").csv(path: _*)

  }
}


object UdfUtility extends Serializable {

  // fill missing weather records over frame-th hours before the flight departure.
  // in strict mode, for a particular missing data point, only the record the precedent record is used to fill up the gap
  // otherwise we use the closest record
  def fillMissingData(weatherConds: Seq[Vector], frame: Int, step: Int): Seq[Double] = {
    if (weatherConds.length < frame) null else weatherConds.toArray.slice(0, frame).flatMap(x => x.toArray).toArray
  }

  def fillMissingData2(originTime: Long, times: Seq[Long], weatherConds: Seq[Vector], frame: Int, step: Int): Seq[Double] = {
    var cds: Seq[Double] = null
    cds = List[Double]()
    var curTime = originTime
    for (_ <- 1 to frame) {
      val diff = times.map(t => Math.abs(curTime - t))
      val index = diff.indexOf(diff.min)
      cds ++= weatherConds(index).toArray
      curTime -= step
    }
    if (cds.isEmpty) null else cds
  }

  val fillWeatherDataUdf: UserDefinedFunction = udf((weatherConds: Seq[Vector], frame: Int, step: Int) => {
    fillMissingData(weatherConds, frame, step)
  })

  val fillWeatherDataUdf2: UserDefinedFunction = udf((originTime: Long, times: Seq[Long], weatherConds: Seq[Vector], frame: Int, step: Int) => {
    fillMissingData2(originTime, times, weatherConds, frame, step)
  })

  val assembleVectors: UserDefinedFunction = udf((cond1: Seq[Double], cond2: Seq[Double]) => {
    Vectors.dense((cond1 ++ cond2).toArray)
  })

  val skys: Array[String] = Array("SKC", "FEW", "SCT", "BKN", "OVC")

  // SKC0200 SCT03011 BKN0400
  // 1 3 4
  def parseSkyCondition(skyCond: String): String = {
    if (skyCond.length < 3) null else skyCond.substring(0, 3)
    //    val conds = skyCond.trim.split(" ")
    //    if (conds(0).length < 3) null else conds(0).substring(0, 3)
    //    val full = (conds ++ Array.fill(3 - conds.length)("ZZZ")).map(x => if (x.length < 3) "ZZZ" else x.substring(0, 3))
    //    full
  }

  val parseSkyConditionUdf: UserDefinedFunction = udf((skyCond: String) => {
    parseSkyCondition(skyCond)
  })

  // SKC0200 SCT03011 BKN0400
  // 1 3 4
  val parseSkyConditionUdf2: UserDefinedFunction = udf((skyCond: String, index: Int) => {
    if (skyCond.isEmpty) -1
    else {
      val skyConds = skyCond.trim.split(" ")
      if (skyConds.length <= index) -1
      else {
        val x = skyConds(index)
        if (x.length >= 3)
          skys.indexOf(x.substring(0, 3))
        else -1
      }
    }
  })

  implicit def strContains(str: Array[String], arr: Array[String]): Double = {
    if (str.intersect(arr).isEmpty) 0.0 else 1.0
  }

  val descriptor: Array[String] = Array("MI", "BC", "PR", "TS", "BL", "SH", "DR", "FZ")
  val precipitation: Array[String] = Array("DZ", "RA", "SN", "SG", "IC", "PL", "GR", "GS")
  val obscuration: Array[String] = Array("BR", "FG", "FU", "VA", "SA", "HZ", "PY", "DU", "SQ", "SS", "DS", "PO", "FC")
  val parseWeatherTypeUdf: UserDefinedFunction = udf((weatherType: String) => {
    val x = weatherType.replace(" ", "").grouped(2).toArray.distinct
    4 * strContains(x, descriptor) + 2 * strContains(x, precipitation) + strContains(x, obscuration)
  })

}
