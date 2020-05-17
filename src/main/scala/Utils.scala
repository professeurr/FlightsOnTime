import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.joda.time.DateTime
import scala.util.control.Breaks._

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

  val fillMissingDataUdf: UserDefinedFunction = udf((originTime: Long, record: Seq[Long], conds: Seq[Seq[String]], frame: Int) => {
    var cds = List[Seq[String]]()
    var curTime = originTime
    breakable {
      for (_ <- 1 to frame) {
        var diff = record.filter(t => t <= curTime)
        if (diff.isEmpty) {
          cds = null
          break
        }
        diff = diff.map(t => curTime - t)
        val index = diff.indexOf(diff.min)
        cds = cds :+ conds(index)
        curTime -= 3600
      }
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
