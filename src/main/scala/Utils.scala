import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vectors
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

  // fill missing weather records over frame-th hours before the flight departure.
  // in strict mode, for a particular missing data point, only the record the precedent record is used to fill up the gap
  // otherwise we use the closest record
  // in the strict mode, the computed record might miss some data points, in that case, null is returned.
  val fillMissingDataUdf: UserDefinedFunction = udf((originTime: Long, times: Seq[Long], weatherConds: Seq[Seq[Double]], frame: Int) => {
    var cds: Seq[Double] = null
    val enoughRecords = true //times.exists(t => t <= originTime - frame * 3600) //uncomment this to enable the strict filtering
    if (enoughRecords) {
      cds = List[Double]()
      var curTime = originTime
      for (_ <- 1 to frame) {
        val diff = times.map(t => Math.abs(curTime - t))
        val index = diff.indexOf(diff.min)
        cds ++= weatherConds(index)
        curTime -= 3600
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
    println(s"[${DateTime.now()}] $str")
  }


}
