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
    config = json.camelizeKeys.extract[Configuration]
    f.close()
    config
  }

  def destroy(): Unit = {
    if (sparkSession != null)
      sparkSession.close()
  }

  def log(s: Any) = {
    logger.info(s)
  }

  def show(data: DataFrame, truncate: Boolean = false): Unit = {
    if (config.verbose)
      data.show(truncate = truncate)
  }

  def count(data: DataFrame): String = {
    if (config.verbose) data.count().toString else "---"
  }

  def exit() = {
    scala.sys.exit(0)
  }
}


object UdfUtility extends Serializable {

  // fill missing weather records over frame-th hours before the flight departure.
  // in strict mode, for a particular missing data point, only the record the precedent record is used to fill up the gap
  // otherwise we use the closest record
  def fillMissingData(originTime: Long, times: Seq[Long], weatherConds: Seq[Vector], frame: Int, step: Int): Seq[Double] = {
    var cds: Seq[Double] = null
    val delta = step * 3600
    cds = List[Double]()
    var curTime = originTime
    for (_ <- 1 to frame) {
      val diff = times.map(t => Math.abs(curTime - t))
      val index = diff.indexOf(diff.min)
      cds ++= weatherConds(index).toDense.toArray
      curTime -= delta
    }
    if (cds.isEmpty) null else cds
  }

  val fillWeatherDataUdf: UserDefinedFunction = udf((originTime: Long, depTimes: Seq[Long], depWeatherConds: Seq[Vector],
                                                     destTime: Long, arrTimes: Seq[Long], arrWeatherConds: Seq[Vector],
                                                     frame: Int, step: Int) => {
    var res: Seq[Double] = null
    val depData = fillMissingData(originTime, depTimes, depWeatherConds, frame, step)
    if (depData != null) {
      val arrData = fillMissingData(destTime, arrTimes, arrWeatherConds, frame, step)
      if (arrData != null)
        res = depData ++ arrData
    }
    if (res != null) Vectors.dense(res.toArray) else null
  })

  val skys: Array[String] = Array("SKC", "FEW", "SCT", "BKN", "OVC")

  // SKC0200 SCT03011 BKN0400
  // 1 3 4
  val parseSkyConditionUdf: UserDefinedFunction = udf((skyCond: String, index: Int) => {
    if (skyCond.isEmpty) -1
    else {
      val skyConds = skyCond.trim.split(" ")
      val x = skyConds(Math.min(index, skyConds.length - 1))
      if (x.length >= 3)
        skys.indexOf(x.substring(0, 3))
      else -1
    }
  })

  val parseWeatherTypeUdf: UserDefinedFunction = udf((weatherType: String) => {
    weatherType.trim.replace("+", "").replace("-", "").replace(" ", "").length / 2
  })

  val computeLineUdf: UserDefinedFunction = udf((airport1: String, airport2: String) => {
    try {
      val id1 = airport1.toInt
      val id2 = airport2.toInt
      Math.min(id1, id2) + "_" + Math.max(id1, id2)
    } catch {
      case _: Exception => null
    }
  })
}
