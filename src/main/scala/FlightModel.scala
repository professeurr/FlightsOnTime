import org.apache.log4j.Logger
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.DataFrame

trait FlightModel {

  import Utility.sparkSession.implicits._

  def getName: String

  def fit(trainingData: DataFrame): Unit

  def evaluate(testData: DataFrame): DataFrame

  def summarize(predictions: DataFrame): DataFrame = {
    val rdd = predictions.select("FL_ONTIME", "prediction").rdd.map(row => (row.getDouble(0), row.getDouble(1)))
    val metrics = new MulticlassMetrics(rdd)
    val metricsDF = Seq(
      (getName, "All", "Accuracy", metrics.accuracy),
      (getName, "Delayed", "Recall", metrics.recall(0.0)),
      (getName, "Delayed", "Precision", metrics.precision(0.0)),
      (getName, "OnTime", "Recall", metrics.recall(1.0)),
      (getName, "OnTime", "Precision", metrics.precision(1.0)))
      .toDF("model", "label", "metric", "value")
      .withColumn("value", Utility.percentUdf($"value"))

    metricsDF.show(truncate = false)
    metricsDF
  }
}

class FlightWeatherDecisionTree() extends FlightModel {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  var model: DecisionTreeClassificationModel = _

  override def getName: String = {
    "DecisionTree"
  }

  override def fit(trainingData: DataFrame): Unit = {
    logger.info("Training DecisionTreeClassifier model on the training data")
    val dt = new DecisionTreeClassifier().setLabelCol("FL_ONTIME").setFeaturesCol("WEATHER_COND")
    model = dt.fit(trainingData)
  }

  override def evaluate(testData: DataFrame): DataFrame = {
    logger.info("evaluating the model on the test data...")
    model.transform(testData)
  }

}

class FlightWeatherRandomForest() extends FlightModel {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  var model: RandomForestClassificationModel = _

  override def getName: String = {
    "RandomForest"
  }

  override def fit(trainingData: DataFrame): Unit = {
    logger.info("Training RandomForestClassifier model on the training data")
    val rf = new RandomForestClassifier().setNumTrees(10).setLabelCol("FL_ONTIME").setFeaturesCol("WEATHER_COND")
    model = rf.fit(trainingData)
  }

  override def evaluate(testData: DataFrame): DataFrame = {
    logger.info("evaluating the model on the test data...")
    model.transform(testData)
  }


}