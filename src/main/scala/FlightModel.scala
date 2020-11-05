import org.apache.log4j.Logger
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.DataFrame

trait FlightModel {

  var model: PipelineModel = _

  def getName: String

  def fit(trainingData: DataFrame): FlightModel

  def evaluate(testData: DataFrame): DataFrame

  // evaluate trained model
  def evaluate(modelPath: String, testData: DataFrame): DataFrame = {
    model = PipelineModel.load(modelPath + "rf.model")
    model.transform(testData)
  }

  def save(path: String): Unit

  def summarize(predictions: DataFrame): Unit = {
    val rdd = predictions.select("FL_ONTIME", "prediction").rdd.map(row => (row.getDouble(0), row.getDouble(1)))
    val metrics = new MulticlassMetrics(rdd)
    val metricsDF = Seq(
      ("Accuracy         :", metrics.accuracy),
      ("Delayed Recall   :", metrics.recall(0.0)),
      ("Delayed Precision:", metrics.precision(0.0)),
      ("OnTime Recall    :", metrics.recall(1.0)),
      ("OnTime Precision :", metrics.precision(1.0)))
      .map(r => "\t" + r._1 + " " + Utility.percent(r._2))
      .mkString("\n")
    //.toDF("model", "label", "metric", "value")
    //.withColumn("value", Utility.percentUdf($"value"))

    Utility.log(s"$getName metrics\n$metricsDF")
    //metricsDF
  }
}

class FlightWeatherDecisionTree() extends FlightModel {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  override def getName: String = {
    "DecisionTree"
  }

  override def fit(trainingData: DataFrame): FlightModel = {
    Utility.log("Training DecisionTreeClassifier model on the training data")
    val dt = new DecisionTreeClassifier().setLabelCol("FL_ONTIME").setFeaturesCol("WEATHER_COND")
    val pipeline = new Pipeline().setStages(Array(dt))
    model = pipeline.fit(trainingData)
    this
  }

  override def evaluate(testData: DataFrame): DataFrame = {
    Utility.log("evaluating the model on the test data...")
    model.transform(testData)
  }

  override def save(path: String): Unit = {
    model.write.overwrite.save(path + "dt.model")
  }
}

class FlightWeatherRandomForest() extends FlightModel {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  override def getName: String = {
    "RandomForest"
  }

  override def fit(trainingData: DataFrame): FlightModel = {
    Utility.log("Training RandomForestClassifier model on the training data")
    val rf = new RandomForestClassifier().setNumTrees(10).setLabelCol("FL_ONTIME").setFeaturesCol("WEATHER_COND")
    val pipeline = new Pipeline().setStages(Array(rf))
    model = pipeline.fit(trainingData)
    this
  }

  override def evaluate(testData: DataFrame): DataFrame = {
    Utility.log("evaluating the model on the test data...")
    model.transform(testData)
  }

  override def save(path: String): Unit = {
    model.write.overwrite.save(path + "rf.model")
  }
}