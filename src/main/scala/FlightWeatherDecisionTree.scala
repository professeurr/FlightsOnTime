import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressor

class FlightWeatherDecisionTree(flightWeatherWrangling: FlightWeatherWrangling) {

  def evaluate(): Unit = {
    Utils.log("Balance the dataset")
    var ontimeFlights = flightWeatherWrangling.Data.where("Fl_ONTIME = 1").cache()
    val ontimeFlightsCount = ontimeFlights.count().toDouble
    var delayedFlights = flightWeatherWrangling.Data.where("FL_ONTIME = 0").cache()
    val delayedFlightsCount = delayedFlights.count().toDouble

    Utils.log(s"nbFlights=${delayedFlightsCount + ontimeFlightsCount}")
    if (ontimeFlightsCount > delayedFlightsCount) {
      ontimeFlights = ontimeFlights.sample(withReplacement = false, delayedFlightsCount / ontimeFlightsCount)
      Utils.log(s"nbFlights=${delayedFlightsCount * 2}")
    }
    else {
      delayedFlights = delayedFlights.sample(withReplacement = false, ontimeFlightsCount / delayedFlightsCount)
      Utils.log(s"nbFlights=${ontimeFlightsCount * 2}")
    }

    flightWeatherWrangling.Data = ontimeFlights.union(delayedFlights).cache()
    Utils.log(flightWeatherWrangling.Data)

    Utils.log("split the dataset into training and test data")
    val Array(trainingData, testData) = flightWeatherWrangling.Data.randomSplit(Array(0.9, 0.1), 42L)

    Utils.log("Creating DecisionTreeRegressor")
    val dt = new DecisionTreeRegressor()
      .setLabelCol("FL_ONTIME")
      .setFeaturesCol("WEATHER_COND")

    Utils.log("Fitting the model")
    val dtModel = dt.fit(trainingData)

    Utils.log("predicting")
    val predictions = dtModel.transform(testData)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("FL_ONTIME")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    Utils.log("Evaluate the prediction")
    val rmse = evaluator.evaluate(predictions)

    Utils.log(s"Root Mean Squared Error (RMSE) on test data = $rmse")

  }

}
