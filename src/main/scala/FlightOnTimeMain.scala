import org.apache.log4j.Logger
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.DataFrame

object FlightOnTimeMain {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {

    val t0 = System.nanoTime()
    logger.info("[START]")

    try {

      val config = Utils.config

      val airportWbanWrangling = new AirportWbanWrangling(config.wbanAirportsPath)
      airportWbanWrangling.loadData()

      val flightWrangling = new FlightWrangling(config.flightsPath, airportWbanWrangling, config.flightsDelayThreshold)
      flightWrangling.loadData()

      val weatherWrangling = new WeatherWrangling(config.weatherPath, airportWbanWrangling, config)
      weatherWrangling.loadData()
      logger.info(s"weather data after wrangling: ${weatherWrangling.Data.count()}")


      val flightWeatherWrangling = new FlightWeatherWrangling(flightWrangling, weatherWrangling, config)
      flightWeatherWrangling.loadData()

      val Array(trainingData, testData) = balanceData(flightWeatherWrangling.Data, config.mlBalanceDataset)

      logger.info("Evaluating with Decision Tree...")
      val flightWeatherDT = new FlightWeatherDecisionTree(trainingData, testData, config)
      val predictionDT = flightWeatherDT.evaluate()
      evaluatePrediction(predictionDT)

      logger.info("Evaluating with Random Forest...")
      val flightWeatherRF = new FlightWeatherRandomForest(trainingData, testData, config)
      val predictionRF = flightWeatherRF.evaluate()
      evaluatePrediction(predictionRF)

    } catch {
      case e: Exception =>
        logger.info(e.toString)
        Utils.destroy()
    }

    logger.info(s"[END: ${(System.nanoTime() - t0) / 1000000000} s]")
  }


  def balanceData(cleanedData: DataFrame, balance: Boolean): Array[DataFrame] = {
    var data = cleanedData
    logger.info(s"cleaned data: ${data.count()}")
    var ontimeFlights = data.where("Fl_ONTIME = 1").cache()
    val ontimeFlightsCount = ontimeFlights.count().toDouble
    var delayedFlights = data.where("FL_ONTIME = 0").cache()
    val delayedFlightsCount = delayedFlights.count().toDouble
    logger.info(s"ontimeFlightsCount=$ontimeFlightsCount, delayedFlightsCount=$delayedFlightsCount")
    if (balance && false) {
      logger.info("Balancing the dataset")
      if (ontimeFlightsCount > delayedFlightsCount)
        ontimeFlights = ontimeFlights.sample(withReplacement = false, delayedFlightsCount / ontimeFlightsCount)
      else
        delayedFlights = delayedFlights.sample(withReplacement = false, ontimeFlightsCount / delayedFlightsCount)

      data = ontimeFlights.union(delayedFlights).cache()
      logger.info(s"data after balancing: ${data.count()}")
    }
    logger.info("split the dataset into training and test data")
    var Array(trainingData, testData) = data.randomSplit(Array(0.75, 0.25), seed = 42)
    trainingData = trainingData.cache()
    testData = testData.cache()

    Array(trainingData, testData)
  }


  def evaluatePrediction(predictions: DataFrame): Unit = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("FL_ONTIME")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)

    logger.info("computing metrics...")
    val ontimePositive = predictions.where("FL_ONTIME=1 and prediction=1").count().toDouble
    val ontimeNegative = predictions.where("FL_ONTIME=1 and prediction=0").count().toDouble
    val delayedPositive = predictions.where("FL_ONTIME=0 and prediction=0").count().toDouble
    val delayedNegative = predictions.where("FL_ONTIME=0 and prediction=1").count().toDouble

    val ontimePrecision = ontimePositive / (ontimePositive + delayedNegative)
    val delayedPrecision = delayedPositive / (delayedPositive + ontimeNegative)
    val ontimeRecall = ontimePositive / (ontimePositive + ontimeNegative)
    val delayedRecall = delayedPositive / (delayedPositive + delayedNegative)

    //predictions.show(truncate = false)
    logger.info(s"\nMETRICS:\nAccuracy = $accuracy\n" +
      s"Ontime Precision = $ontimePrecision\n" +
      s"Delayed Precision = $delayedPrecision\n" +
      s"Ontime Recall = $ontimeRecall\n" +
      s"Delayed Recall = $delayedRecall")
  }
}
