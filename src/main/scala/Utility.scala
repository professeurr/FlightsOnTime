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

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  var config: Configuration = initialize

  lazy val sparkSession: SparkSession = SparkSession.builder()
    .appName(s"FlightOnTime${scala.util.Random.nextInt()}")
    .getOrCreate()

  def initialize: Configuration = {
    logger.info("Initializing...")
    if (sparkSession == null) {
      logger.error("Spark session was not created.")
      throw new Exception("Spark session is not created.")
    }
    // Import default formats
    implicit val formats: DefaultFormats.type = DefaultFormats

    logger.info("loading configuration...")
    val configFile = "config.json"
    val path = Paths.get(configFile).normalize()
    if (!Files.exists(path))
      throw new Exception(s"configuration file '$configFile' does not exist. It should be placed in the working directory.")

    logger.info(s"configuration file '${path.toString}' is provided, let's initialize the session.")
    val f = Source.fromFile(configFile)
    // Parse config file into Configuration object
    val json = parse(f.getLines().mkString)
    config = json.camelizeKeys.extract[Configuration]
    f.close()
    config
  }

  def destroy(): Unit = {
    if (sparkSession != null) {
      sparkSession.close()
    }
  }

  def balanceDataset(data: DataFrame, label: String): DataFrame = {
    logger.info("Balancing the dataset")
    var ontimeFlights = data.filter(s"$label = 1")
    val ontimeFlightsCount = ontimeFlights.count().toDouble
    var delayedFlights = data.filter(s"$label = 0").cache()
    val delayedFlightsCount = delayedFlights.count().toDouble

    logger.info(s"ontimeFlightsCount=$ontimeFlightsCount, delayedFlightsCount=$delayedFlightsCount")
    if (ontimeFlightsCount > delayedFlightsCount)
      ontimeFlights = ontimeFlights.sample(withReplacement = false, delayedFlightsCount / ontimeFlightsCount)
    else
      delayedFlights = delayedFlights.sample(withReplacement = false, ontimeFlightsCount / delayedFlightsCount)

    ontimeFlights.union(delayedFlights).cache()
  }

  def percentUdf: UserDefinedFunction = udf((number: Double) => {
    s"${Math.round(100 * number)}%"
  })

}


object UdfUtility extends Serializable {

  // fill missing weather records over frame-th hours before the flight departure.
  // in strict mode, for a particular missing data point, only the record the precedent record is used to fill up the gap
  // otherwise we use the closest record
  def fillMissingData(originTime: Long, times: Seq[Long], weatherConds: Seq[Vector], frame: Int, step: Int): Seq[Double] = {
    var cds : Seq[Double] = null
    val delta = step * 3600
    cds = List[Double]()
    var curTime = originTime
    for (_ <- 1 to frame) {
      val diff = times.map(t => Math.abs(curTime - t))
      val index = diff.indexOf(diff.min)
      cds ++= weatherConds(index).toDense.toArray
      curTime -= delta
    }
    if(cds.isEmpty) null else cds
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
  def parSkyCondition(skyCond: String): Array[Int] = {
    var result = Array[Int](0, 0, 0)
    var t = skyCond.trim.split(" ").map(x => {
      val f = if (x.length < 3) -1 else skys.indexOf(x.toUpperCase.substring(0, 3))
      if (f == -1) 0 else f + 1
    })
    if (!t.isEmpty) {
      if (t.length == 1) {
        t :+= t(0)
        t :+= t(0)
      } else if (t.length == 2)
        t :+= t(1)
      result = t
    }
    result
  }

  val parseWeatherVariablesUdf: UserDefinedFunction = udf((skyCond: String, dryBulb: String, weatherType: String,
                                                           stationPressure: String, windDirection: String,
                                                           visibility: String, relativeHumidity: String, windSpeed: String) => {
    var result = parSkyCondition(skyCond)

    result :+= evalClass(windDirection, 360.0 / 8.0)

    //between 0 - 10 -> 3 classes
    result :+= Math.min(evalClass(visibility, 4), 4)

    //between 0 - 100 -> 4 classes
    result :+= evalClass(relativeHumidity, 25)

    //between 0 - 100 KT (0 - 50 m/s) -> 4 classes https://www.lmwindpower.com/en/stories-and-press/stories/learn-about-wind/what-is-a-wind-class
    result :+= evalClass(windSpeed, 25)

    result :+= evalClass(stationPressure, 10, -10)

    // between -50 - +50 -> 4 classes
    result :+= evalClass(dryBulb, 25, 50)

    try {
      result :+= weatherType.trim.split(" ").length + 1
    }
    catch {
      case _ : Exception => result :+= 0
    }

    result
  })

  def evalClass(s: String, coeff: Double, bias: Double = 0): Int = {
    var c: Int = 0
    try {
      c = ((s.trim.toDouble + bias) / coeff).toInt + 1
    }
    catch {
      case _: Throwable =>
    }
    c
  }

  val toArrayUdf: UserDefinedFunction = udf((vect: Vector) => {
    vect.toDense.toArray
  })
}
