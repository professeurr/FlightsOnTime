import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.DataFrame

abstract class FlightModel(configuration: Configuration, modelName: String, modelPath: String) {

  var pipelineModel: PipelineModel = _

  // our model's name used in the trace
  def getName: String = modelName

  // name of the file where the pipeline is saved or loaded
  def pipelineFilename: String = modelPath

  // the fit function to be implemented by each concrete class
  def fit(trainingData: DataFrame): FlightModel

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

class FlightDelayCrossValidation(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModel(configuration, modelName, modelPath) {

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
      .addGrid(rf.maxDepth, Array(10, 15, 20))
      .addGrid(rf.numTrees, Array(5, 10, 20))
      .addGrid(rf.maxBins, Array(3, 5, 9))
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
      .setNumFolds(5)
      .setParallelism(configuration.partitions)

    // run the cross-validation engine
    val cvModel = cv.fit(trainingData)

    // explain the parameters
    cvModel.explainParams()

    // get the best model with its optimal parameters for persistence and testing
    pipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    Utility.log("best model: " + pipelineModel.toString())
    this
  }
}

class FlightDelayDecisionTree(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModel(configuration, modelName, modelPath) {

  override def fit(trainingData: DataFrame): FlightModel = {
    val dt = new DecisionTreeClassifier()
      .setMaxBins(7)
      .setMaxDepth(20)
      .setLabelCol("FL_ONTIME")
      .setFeaturesCol("WEATHER_COND")
    val pipeline = new Pipeline().setStages(Array(dt))
    pipelineModel = pipeline.fit(trainingData)
    this
  }
}

class FlightDelayRandomForest(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModel(configuration, modelName, modelPath) {

  override def fit(trainingData: DataFrame): FlightModel = {
    val rf = new RandomForestClassifier()
      .setMaxBins(8)
      .setMaxDepth(25)
      .setNumTrees(25)
      .setImpurity("gini")
      .setLabelCol("FL_ONTIME")
      .setFeaturesCol("WEATHER_COND")
    val pipeline = new Pipeline().setStages(Array(rf))
    pipelineModel = pipeline.fit(trainingData)
    this
  }

}

class FlightDelayLogisticRegression(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModel(configuration, modelName, modelPath) {

  override def fit(trainingData: DataFrame): FlightModel = {
    val lr = new LogisticRegression()
      .setMaxIter(1000)
      .setRegParam(0.1)
      .setElasticNetParam(0.1)
      .setTol(1e-6)
      .setFamily("binomial")
      .setLabelCol("FL_ONTIME")
      .setFeaturesCol("WEATHER_COND")

    val pipeline = new Pipeline().setStages(Array(lr))
    pipelineModel = pipeline.fit(trainingData)
    this
  }

}