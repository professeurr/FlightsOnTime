import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame

class FlightDelayRandomForestClassifier(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModel(configuration, modelName, modelPath) {

  override def fit(trainingData: DataFrame): FlightModel = {
    val rf = new RandomForestClassifier()
      .setMaxBins(10)
      .setMaxDepth(15)
      .setNumTrees(20)
      .setImpurity("gini")
      .setLabelCol("delayed")
      .setFeaturesCol("features")
    val pipeline = new Pipeline().setStages(Array(rf))
    pipelineModel = pipeline.fit(trainingData)
    this
  }

}

class FlightDelayCrossValidation(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModel(configuration, modelName, modelPath) {

  override def fit(trainingData: DataFrame): FlightModel = {
    // create decision tree model
    val dt = new DecisionTreeClassifier()
      .setLabelCol("delayed")
      .setFeaturesCol("features")

    // create random forest model
    val rf = new RandomForestClassifier()
      .setLabelCol("delayed")
      .setFeaturesCol("features")

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
      .setLabelCol("delayed")
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
