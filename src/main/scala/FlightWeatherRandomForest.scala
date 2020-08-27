import org.apache.log4j.Logger
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.sql.DataFrame

class FlightWeatherRandomForest(trainingData: DataFrame, testData: DataFrame, config: Configuration) {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  def evaluate(): DataFrame = {

    logger.info("Training RandomForest model on the training data")
    val model = new RandomForestClassifier()
      .setNumTrees(10)
      .setLabelCol("FL_ONTIME")
      .setFeaturesCol("WEATHER_COND")

    val trainedModel = model.fit(trainingData)
    trainedModel.write.overwrite().save(config.modelPath)

    logger.info("evaluating the model on the test data...")
    trainedModel.transform(testData)
  }

}
