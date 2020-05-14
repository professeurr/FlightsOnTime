import java.sql.Timestamp
import java.text.SimpleDateFormat

import org.apache.commons.lang3.StringUtils
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.joda.time.DateTime

object Utils {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  // Initialize spark session.
  val sparkSession: SparkSession = SparkSession.builder()
    .appName(s"DescentGradientFETS${scala.util.Random.nextInt()}")
    .master("local")
    .getOrCreate()

  // Get the sparkContext from the session
  val sc: SparkContext = sparkSession.sparkContext

  val numberOfCores: Int = java.lang.Runtime.getRuntime.availableProcessors * Math.max(sc.statusTracker.getExecutorInfos.length - 1, 1)

  // Convert the given date and time into UTC time by applying the time zone
  val convertTimeUdf: UserDefinedFunction = udf((date: Timestamp, time: Int, timeZone: Int) => {
    val timeStr = StringUtils.leftPad(time.toString, 4, "0")
    new Timestamp(date.getTime + ((timeZone + timeStr.substring(0, 2).toInt) * 60 + timeStr.substring(2, 4).toInt) * 60000)
  })

  // Convert the given date and time into UTC time by applying the time zone
  val convertWeatherTimeUdf: UserDefinedFunction = udf((date: Int, time: Int, timeZone: Int) => {
    val d = new SimpleDateFormat("yyyyMMdd").parse(date.toString)
    new Timestamp(d.getTime + (timeZone * 60 + time) * 60000)
  })

  val isOnTimeUdf: UserDefinedFunction = udf((th: Double, cd: Double, wd: Double, nasd: Double, sd: Double, ld: Double) => {
    if (Array(cd, wd, nasd, sd, ld).forall(x => x < th)) 1 else 0
  })

  val fillWeatherMissingDataUdf: UserDefinedFunction = udf((originTime: Timestamp, record: Seq[Timestamp], frame: Int) => {
    var ts = List[Timestamp]()
    if (originTime == null)
      ts
    else {
      var curTime = originTime.getTime
      val tr = record.map(t => t.getTime)
      for (_ <- 1 to frame) {
        val diff = tr.map(t => math.abs(t - curTime))
        val index = diff.indexOf(diff.min)
        ts = ts :+ record(index)
        curTime -= 3600000
      }
      ts.sortBy(t => t.getTime)
    }
  })

  val hasImpactOnFlight: UserDefinedFunction = udf((flightTime: Timestamp, weatherTime: Timestamp, timeDecay: Int) => {
    if (flightTime == null)
      false
    else {
      val delta = flightTime.getTime - weatherTime.getTime
      delta >= 0 && delta <= timeDecay * 3600000
    }
  })

  val fillMissingDataUdf: UserDefinedFunction = udf((originTime: Long, record: Seq[Long], conds: Seq[Seq[String]], frame: Int) => {
    var ts = List[Long]()
    var cds = List[Seq[String]]()
    var curTime = originTime
    val tr = record.map(t => t)
    for (_ <- 1 to frame) {
      val diff = tr.map(t => math.abs(t - curTime))
      val index = diff.indexOf(diff.min)
      ts = ts :+ record(index)
      cds = cds :+ conds(index)
      curTime -= 3600
    }
    cds
  })

  def log(df: DataFrame, size: Int = 100): Unit = {
    //println(s"partitions: ${df.rdd.getNumPartitions}")
    df.printSchema()
    df.explain(false)
  }

  def show(df: DataFrame, size: Int = 100): Unit = {
    //println(df.take(size).mkString("\n"))
    df.show(size, truncate = false)
  }

  def log(str: String): Unit = {
    println(s"[${DateTime.now()}] $str")
  }


}
