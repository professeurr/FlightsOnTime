import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.DataFrame

trait FlightModel {

  var pipelineModel: PipelineModel = _

  def getName: String

  def fit(trainingData: DataFrame): FlightModel

  def evaluate(testData: DataFrame): DataFrame

  // evaluate trained model
  def evaluate(modelPath: String, testData: DataFrame): DataFrame = {
    pipelineModel = PipelineModel.load(modelPath)
    pipelineModel.transform(testData)
  }

  def save(path: String): Unit = {
    Utility.log(s"saving the model $getName...")
    pipelineModel.write.overwrite.save(path)
  }

  def summarize(predictions: DataFrame): Unit = {
    //predictions.select("FL_ONTIME", "prediction").show()
    val rdd = predictions.select("prediction", "FL_ONTIME").rdd.map(row => (row.getDouble(0), row.getDouble(1)))
    val multiclassMetrics = new MulticlassMetrics(rdd)
    val binaryClassMetrics = new BinaryClassificationMetrics(rdd)
    val metricsDF = Seq(
      ("Accuracy         ", multiclassMetrics.accuracy),
      ("Area Under ROC   ", binaryClassMetrics.areaUnderROC()),
      ("Delayed Recall   ", multiclassMetrics.recall(0.0)),
      ("Delayed Precision", multiclassMetrics.precision(0.0)),
      ("OnTime Recall    ", multiclassMetrics.recall(1.0)),
      ("OnTime Precision ", multiclassMetrics.precision(1.0)))
      .map(r => "\t" + r._1 + s": ${Math.round(100 * r._2)}%")
      .mkString("\n")

    Utility.log(s"$getName metrics\n$metricsDF")
    //metricsDF
  }
}

class FlightDelayCrossValidation(configuration: Configuration) extends FlightModel {
  override def getName: String = {
    "CrossValidation"
  }

  override def fit(trainingData: DataFrame): FlightModel = {
    // create decision tree model
    val dt = new DecisionTreeClassifier()
      .setLabelCol("FL_ONTIME")
      .setFeaturesCol("WEATHER_COND")

    // create random forest model
    val rf = new RandomForestClassifier()
      .setLabelCol("FL_ONTIME")
      .setFeaturesCol("WEATHER_COND")

    // configure an ML pipeline by adding the 2 models above
    val pipeline = new Pipeline().setStages(Array(rf))

    // use ParamGridBuilder to specify the parameters to search over and their range of values
    val paramGrid = new ParamGridBuilder()
      //      .addGrid(dt.maxDepth, Array(3, 5, 7))
      //      .addGrid(dt.maxBins, Array(5, 10, 20))
      //      .addGrid(dt.impurity, Array("gini", "entropy"))
      .addGrid(rf.maxDepth, Array(3, 5, 10))
      .addGrid(rf.numTrees, Array(5, 10))
      .addGrid(rf.maxBins, Array(5, 10))
      .addGrid(rf.impurity, Array("gini", "entropy"))
      .build()

    // create the evaluator
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("FL_ONTIME")
      .setRawPredictionCol("prediction") // both decision tree and random forest produce a `prediction` column as output

    // create the cross-validator instance where the pipeline is provided as an estimator,
    // the evaluator and the paramMap. The number of fold is set to 3
    // and we tell cross-validator engine to paralyze the search over 3 threads
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
      .setParallelism(3)

    // run the cross-validation engine
    val cvModel = cv.fit(trainingData)

    // explain the parameters
    cvModel.explainParams()

    // get the best model with its optimal parameters for persistence and testing
    pipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    Utility.log("best model: " + pipelineModel.toString())
    this
  }

  override def evaluate(testData: DataFrame): DataFrame = {
    super.evaluate(configuration.persistPath + "cv.model", testData)
  }

  override def save(path: String): Unit = {
    super.save(path + "cv.model")
  }
}

class FlightWeatherDecisionTree(configuration: Configuration) extends FlightModel {

  override def getName: String = {
    "DecisionTree"
  }

  override def fit(trainingData: DataFrame): FlightModel = {
    val dt = new DecisionTreeClassifier()
      .setLabelCol("FL_ONTIME")
      .setFeaturesCol("WEATHER_COND")
    val pipeline = new Pipeline().setStages(Array(dt))
    pipelineModel = pipeline.fit(trainingData)
    this
  }

  override def evaluate(testData: DataFrame): DataFrame = {
    super.evaluate(configuration.persistPath + "df.model", testData)
  }

  override def save(path: String): Unit = {
    super.save(path + "dt.model")
  }
}

class FlightWeatherRandomForest(configuration: Configuration) extends FlightModel {

  override def getName: String = {
    "RandomForest"
  }

  override def fit(trainingData: DataFrame): FlightModel = {
    val rf = new RandomForestClassifier()
      .setLabelCol("FL_ONTIME")
      .setFeaturesCol("WEATHER_COND")
    val pipeline = new Pipeline().setStages(Array(rf))
    pipelineModel = pipeline.fit(trainingData)
    this
  }

  override def evaluate(testData: DataFrame): DataFrame = {
    super.evaluate(configuration.persistPath + "rf.model", testData)
  }

  override def save(path: String): Unit = {
    super.save(path + "rf.model")
  }
}