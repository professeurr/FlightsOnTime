import org.apache.log4j.Logger
import org.apache.spark.mllib.evaluation.MulticlassMetrics
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

      val flightWrangling = new FlightWrangling(config.flightsPath, airportWbanWrangling.Data, config.flightsDelayThreshold)
      flightWrangling.loadData()

      val weatherWrangling = new WeatherWrangling(config.weatherPath, airportWbanWrangling.Data, config)
      weatherWrangling.loadData()
      logger.info(s"weather data after wrangling: ${weatherWrangling.Data.count()}")

      val flightWeatherWrangling = new FlightWeatherWrangling(flightWrangling.Data, weatherWrangling.Data, config)
      flightWeatherWrangling.loadData()

      val Array(trainingData, testData) = balanceData(flightWeatherWrangling.Data)

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


  def balanceData(cleanedData: DataFrame): Array[DataFrame] = {
    logger.info("split the dataset into training and test data")
    var Array(trainingData, testData) = cleanedData.randomSplit(Array(0.80, 0.20), seed = 42L)
    trainingData = trainingData.cache()
    testData = testData.cache()
    Array(trainingData, testData)
  }

  def evaluatePrediction(predictions: DataFrame): Unit = {
    logger.info("evaluating the prediction")
    val rdd = predictions.select("FL_ONTIME", "prediction").rdd.map(row ⇒ (row.getDouble(0), row.getDouble(1)))
    val metrics = new MulticlassMetrics(rdd)
    println("Overall accuracy: " + metrics.accuracy)
    println("Recall for delayed flights: " + metrics.recall(0.0))
    println("Precision for delayed flights: " + metrics.precision(0.0))
    println("Recall for on-time flights: " + metrics.recall(1.0))
    println("Precision for on-time flights: " + metrics.precision(1.0))

  }

}
