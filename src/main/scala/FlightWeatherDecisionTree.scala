import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

class FlightWeatherDecisionTree(flightWeatherWrangling: FlightWeatherWrangling) {

  def evaluate(): Unit = {
    Utils.log("Balance the dataset")
    var ontimeFlights = flightWeatherWrangling.Data.where("Fl_ONTIME = 1").cache()
    val ontimeFlightsCount = ontimeFlights.count().toDouble
    var delayedFlights = flightWeatherWrangling.Data.where("FL_ONTIME = 0").cache()
    val delayedFlightsCount = delayedFlights.count().toDouble

    Utils.log(s"ontimeFlightsCount=$ontimeFlightsCount, delayedFlightsCount=$delayedFlightsCount")
    if (ontimeFlightsCount > delayedFlightsCount)
      ontimeFlights = ontimeFlights.sample(withReplacement = false, delayedFlightsCount / ontimeFlightsCount)
    else
      delayedFlights = delayedFlights.sample(withReplacement = false, ontimeFlightsCount / delayedFlightsCount)

    flightWeatherWrangling.Data = ontimeFlights.union(delayedFlights).cache()
    Utils.log(flightWeatherWrangling.Data)

    Utils.log("split the dataset into training and test data")
    var Array(trainingData, testData) = flightWeatherWrangling.Data.randomSplit(Array(0.9, 0.1), 42L)

    trainingData = trainingData.cache()
    testData = testData.cache()

    Utils.log("Creating DecisionTreeRegressor")
    val dt = new DecisionTreeClassifier()
      .setLabelCol("FL_ONTIME")
      .setFeaturesCol("WEATHER_COND")

    Utils.log("Fitting the model")
    val dtModel = dt.fit(trainingData)

    Utils.log("predicting...")
    val predictions = dtModel.transform(testData)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("FL_ONTIME")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    Utils.log("Evaluate the prediction")
    val accuracy = evaluator.evaluate(predictions)

    Utils.log("prediction result")
    val predictionData = predictions.select("FL_ONTIME", "prediction", "rawPrediction", "probability", "FL_ID", "WEATHER_COND")
    Utils.log(predictionData)

    Utils.log("computing metrics...")
    val truePositive = predictionData.where("FL_ONTIME=1 and prediction=1").count().toDouble
    val falseNegative = predictionData.where("FL_ONTIME=1 and prediction=0").count().toDouble
    val trueNegative = predictionData.where("FL_ONTIME=0 and prediction=0").count().toDouble
    val falsePositive = predictionData.where("FL_ONTIME=0 and prediction=1").count().toDouble

    val ontimeRecall = truePositive / (truePositive + falseNegative)
    val delayedRecall = trueNegative / (trueNegative + falsePositive)

    Utils.log(s"Accuracy = $accuracy")
    Utils.log(s"Ontime Recall = $ontimeRecall")
    Utils.log(s"Delayed Recall = $delayedRecall")
  }

}
