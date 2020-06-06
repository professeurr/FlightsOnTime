import org.apache.log4j.Logger
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

class FlightWeatherDecisionTree(flightWeatherWrangling: FlightWeatherWrangling, outputPath: String) {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  def evaluate(): Unit = {

    logger.info("Balancing the dataset")
    var data = flightWeatherWrangling.Data
    var ontimeFlights = data.where("Fl_ONTIME = 1").cache()
    val ontimeFlightsCount = ontimeFlights.count().toDouble
    var delayedFlights = data.where("FL_ONTIME = 0").cache()
    val delayedFlightsCount = delayedFlights.count().toDouble

    logger.info(s"ontimeFlightsCount=$ontimeFlightsCount, delayedFlightsCount=$delayedFlightsCount")
    if (ontimeFlightsCount > delayedFlightsCount)
      ontimeFlights = ontimeFlights.sample(withReplacement = false, delayedFlightsCount / ontimeFlightsCount)
    else
      delayedFlights = delayedFlights.sample(withReplacement = false, ontimeFlightsCount / delayedFlightsCount)

    data = ontimeFlights.union(delayedFlights).cache()
    logger.info(data)

    logger.info("split the dataset into training and test data")
    var Array(trainingData, testData) = data.randomSplit(Array(0.75, 0.25))

    trainingData = trainingData.cache()
    testData = testData.cache()

    logger.info("Training DecisionTreeClassifier model on the training data")
    val model = new DecisionTreeClassifier()
      .setLabelCol("FL_ONTIME")
      .setFeaturesCol("WEATHER_COND")
    val trainedModel = model.fit(trainingData)

    trainedModel.write.overwrite().save(outputPath)

    logger.info("evaluating the model on the test data...")
    var predictions = trainedModel.transform(testData)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("FL_ONTIME")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)

    predictions = predictions.select("FL_ONTIME", "prediction", "rawPrediction", "probability", "FL_ID", "WEATHER_COND")
    logger.info(predictions)

    logger.info("computing metrics...")
    val ontimePositive = predictions.where("FL_ONTIME=1 and prediction=1").count().toDouble
    val ontimeNegative = predictions.where("FL_ONTIME=1 and prediction=0").count().toDouble
    val delayedPositive = predictions.where("FL_ONTIME=0 and prediction=0").count().toDouble
    val delayedNegative = predictions.where("FL_ONTIME=0 and prediction=1").count().toDouble

    val ontimePrecision = ontimePositive / (ontimePositive + delayedNegative)
    val delayedPrecision = delayedPositive / (delayedPositive + ontimeNegative)
    val ontimeRecall = ontimePositive / (ontimePositive + ontimeNegative)
    val delayedRecall = delayedPositive / (delayedPositive + delayedNegative)

    logger.info(s"Accuracy = $accuracy")
    logger.info(s"Ontime Precision = $ontimePrecision")
    logger.info(s"Delayed Precision = $delayedPrecision")
    logger.info(s"Ontime Recall = $ontimeRecall")
    logger.info(s"Delayed Recall = $delayedRecall")
  }

}
