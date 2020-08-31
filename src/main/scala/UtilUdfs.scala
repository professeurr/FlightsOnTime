import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

object UtilUdfs extends Serializable {

  // fill missing weather records over frame-th hours before the flight departure.
  // in strict mode, for a particular missing data point, only the record the precedent record is used to fill up the gap
  // otherwise we use the closest record
  // in the strict mode, the computed record might miss some data points, in that case, null is returned.
  val fillMissingDataUdf: UserDefinedFunction = udf((originTime: Long, times: Seq[Long], weatherConds: Seq[Seq[Double]], frame: Int, step: Int) =>
    fillMissingData(originTime, times, weatherConds, frame, step))

  def fillMissingData(originTime: Long, times: Seq[Long], weatherConds: Seq[Seq[Double]], frame: Int, step: Int): Seq[Double] = {
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
  }

  val fillWeatherDataUdf: UserDefinedFunction = udf((originTime: Long, depTimes: Seq[Long], depWeatherConds: Seq[Seq[Double]],
                                                     destTime: Long, arrTimes: Seq[Long], arrWeatherConds: Seq[Seq[Double]],
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

  // convert array of double to dense vector in order to feed the regressor
  val toVectorUdf: UserDefinedFunction = udf((data: Seq[Double]) => {
    Vectors.dense(data.toArray)
  })


  val padSkyConditionValueUdf: UserDefinedFunction = udf((input: String, padLength: Int, defaultValue: String) => {
    var items: Seq[String] = input.trim().split(" ")
    if (items.isEmpty)
      items = (0 until padLength).map(_ => defaultValue)
    else if (items.length > padLength)
      items = items.slice(0, padLength)
    else if (items.length < padLength)
      items = items ++ Seq.fill(padLength - items.length)(defaultValue)

    items.map(c => c.substring(0, Math.min(c.length, 3)))
  })

  val padWeatherTypeValueUdf: UserDefinedFunction = udf((input: String, padLength: Int, defaultValue: String) => {
    var items: Seq[String] = input.trim().split(" ")
    if (items.isEmpty)
      items = (0 until padLength).map(_ => defaultValue)
    else if (items.length > padLength)
      items = items.slice(0, padLength)
    else if (items.length < padLength)
      items = items ++ Seq.fill(padLength - items.length)(defaultValue)

    items.map(w => {
      val i = if (w.charAt(0) == '+' || w.charAt(0) == '-') 1 else 0
      w.substring(i, Math.min(w.length, 2 + i))
    })
  })


  val timeToHourUdf: UserDefinedFunction = udf((time: String) => {
    time.slice(0, 2)
  })
}
