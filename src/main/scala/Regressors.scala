import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.{GBTRegressor, GeneralizedLinearRegression, RandomForestRegressor}
import org.apache.spark.sql.DataFrame

class FlightDelayRandomForestRegressor(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModel(configuration, modelName, modelPath) {

  override def fit(trainingData: DataFrame): FlightModel = {
    val rf = new RandomForestRegressor().setSeed(42L)
      .setMaxBins(10)
      .setMaxDepth(25)
      .setNumTrees(25)
      .setImpurity("variance")
      .setLabelCol("delay")
      .setFeaturesCol("features")
    val pipeline = new Pipeline().setStages(Array(rf))
    pipelineModel = pipeline.fit(trainingData)
    this
  }

}

class FlightDelayLinearRegressor(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModel(configuration, modelName, modelPath) {

  override def fit(trainingData: DataFrame): FlightModel = {
    val glr = new GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setMaxIter(10000)
      .setRegParam(0.1)
      .setTol(1e-17)
      .setLabelCol("delay")
      .setFeaturesCol("features")
    val pipeline = new Pipeline().setStages(Array(glr))
    pipelineModel = pipeline.fit(trainingData)
    this
  }
}


class FlightDelayGBTRegressor(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModel(configuration, modelName, modelPath) {

  override def fit(trainingData: DataFrame): FlightModel = {
    val gbt = new GBTRegressor()
      .setMaxIter(10)
      .setMaxBins(10)
      .setMaxDepth(10)
      .setStepSize(0.1).setSeed(42L)
      .setLossType("squared")
      .setLabelCol("delay")
      .setFeaturesCol("features")
    val pipeline = new Pipeline().setStages(Array(gbt))
    pipelineModel = pipeline.fit(trainingData)
    this
  }
}

