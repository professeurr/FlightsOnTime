import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.regression.{GBTRegressor, GeneralizedLinearRegression, RandomForestRegressor}

class FlightDelayRandomForestRegressor(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModelRegressor(configuration, modelName, modelPath) {

  override def getModel: PipelineStage = {
    new RandomForestRegressor().setSeed(42L)
      .setMaxBins(10)
      .setMaxDepth(15)
      .setNumTrees(10)
      .setImpurity("variance")
      .setLabelCol("delay")
      .setFeaturesCol("features")
  }
}

class FlightDelayLinearRegressor(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModelRegressor(configuration, modelName, modelPath) {

  override def getModel: PipelineStage = {
    new GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setMaxIter(10000)
      .setRegParam(0.1)
      .setTol(1e-17)
      .setLabelCol("delay")
      .setFeaturesCol("features")
  }
}

class FlightDelayGBTRegressor(configuration: Configuration, modelName: String, modelPath: String)
  extends FlightModelRegressor(configuration, modelName, modelPath) {

  override def getModel: PipelineStage = {
    new GBTRegressor()
      .setMaxIter(10)
      .setMaxBins(10)
      .setMaxDepth(10)
      .setStepSize(0.1).setSeed(42L)
      .setLossType("squared")
      .setLabelCol("delay")
      .setFeaturesCol("features")
  }
}

