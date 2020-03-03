import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SparkSession}
import java.sql.Timestamp
import java.text.SimpleDateFormat

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
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

  val convertTimeUdf: UserDefinedFunction = udf((date: Timestamp, time: Int, timeZone: Int) => {
    if (time.toString.length < 4)
      date
    else {
      val hours = time.toString.substring(0, 2).toInt
      val minutes = time.toString.substring(2, 4).toInt
      new Timestamp(date.getTime + ((timeZone + hours) * 60 + minutes) * 60000)
    }
  })

  val isOnTimeUdf: UserDefinedFunction = udf((th: Double, cd: Double, wd: Double, nasd: Double, sd: Double, ld: Double) => {
    if (Array(cd, wd, nasd, sd, ld).forall(x => x < th)) 1 else 0
  })


  val convertWeatherTimeUdf: UserDefinedFunction = udf((date: Int, time: Int, timeZone: Int) => {
    val d = new SimpleDateFormat("yyyyMMdd").parse(date.toString)
    new Timestamp(d.getTime + (timeZone * 60 + time) * 60000)
  })

  def log(df: DataFrame, size: Int = 100): Unit = {
    df.printSchema()
    df.show(numRows = size, truncate = false)
  }

  def log(str: String): Unit = {
    println(s"[${DateTime.now()}] $str")
  }
}
