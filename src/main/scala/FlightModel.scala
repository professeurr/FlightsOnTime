import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.when

abstract class FlightModel(configuration: Configuration, modelName: String, modelPath: String) {

  var pipelineModel: PipelineModel = _

  // our model's name used in the trace
  def getName: String = modelName

  // name of the file where the pipeline is saved or loaded
  def pipelineFilename: String = modelPath

  def getModel: PipelineStage

  // the fit function to be implemented by each concrete class
  def fit(trainingData: DataFrame): FlightModel = {
    val pipeline = new Pipeline().setStages(Array(getModel))
    pipelineModel = pipeline.fit(trainingData)
    this
  }

  // evaluate trained model on test dataset
  def evaluate(testData: DataFrame): DataFrame = {
    pipelineModel = PipelineModel.load(configuration.persistPath + modelPath)
    val predictions = pipelineModel.transform(testData)
    predictions
  }

  def save(): Unit = {
    Utility.log(s"saving the model $modelName...")
    pipelineModel.write.overwrite.save(configuration.persistPath + modelPath)
  }

  def summarize(predictions: DataFrame): Unit = {
    val rdd = predictions.select("prediction", "delayed").rdd.map(row => (row.getDouble(0), row.getDouble(1)))
    val multiclassMetrics = new MulticlassMetrics(rdd)
    val binaryClassMetrics = new BinaryClassificationMetrics(rdd)

    val metricsDF = Seq(
      ("Accuracy         ", multiclassMetrics.accuracy),
      ("Area Under ROC   ", binaryClassMetrics.areaUnderROC()),
      ("Delayed Recall   ", multiclassMetrics.recall(1.0)),
      ("Delayed Precision", multiclassMetrics.precision(1.0)),
      ("OnTime Recall    ", multiclassMetrics.recall(0.0)),
      ("OnTime Precision ", multiclassMetrics.precision(0.0)))
      .map(r => "\t" + r._1 + s": ${Math.round(100 * r._2)}%")
      .mkString("\n")

    Utility.log(s"$getName metrics\n\tThreshold        : ${configuration.flightsDelayThreshold}min\n$metricsDF")
  }

}

abstract class FlightModelClassifier(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModel(configuration, modelName, modelPath) {

  import Utility.sparkSession.implicits._

  override def fit(trainingData: DataFrame): FlightModel = {
    val data = trainingData.withColumn("delayed", when($"delay" > configuration.flightsDelayThreshold, 1.0).otherwise(0.0))
    super.fit(data)
  }
}

abstract class FlightModelRegressor(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModel(configuration, modelName, modelPath) {

  import Utility.sparkSession.implicits._

  override def summarize(predictions: DataFrame): Unit = {
    val predict = predictions.withColumnRenamed("prediction", "delay_prediction")
      .withColumn("delayed", when($"delay" > configuration.flightsDelayThreshold, 1.0).otherwise(0.0))
      .withColumn("prediction", when($"delay_prediction" > configuration.flightsDelayThreshold, 1.0).otherwise(0.0))

//    predict.select("delay", "delay_prediction", "delayed", "prediction")
//      .filter("delay_prediction < 0")
//      .show(truncate = false)
    val evaluator = new RegressionEvaluator()
      .setLabelCol("delay")
      .setPredictionCol("delay_prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predict)
    Utility.log(s"Root Mean Squared Error (RMSE): $rmse")

    super.summarize(predict)
  }
}