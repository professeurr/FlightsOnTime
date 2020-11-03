import org.apache.log4j.Logger
import org.apache.spark.sql.functions.broadcast

object Main {
  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {

    val t0 = System.nanoTime()
    logger.info("[START]")

    try {
      val config = Utility.config

      logger.info("[DATA PREPARATION]")
      val dataLoader = new DataLoader(config)
      // broadcast this dataset which is small compare to flights and weather ones. Broadcasting it will significantly speed up the join operations
      val airportWbanData = broadcast(dataLoader.loadMappingData())
      val weatherData = dataLoader.loadWeatherData(airportWbanData).cache()
      val flightData = dataLoader.loadFlightData(airportWbanData).cache()
      val data = dataLoader.combineData(flightData, weatherData).cache()

      var Array(trainingData, testData) = data.randomSplit(Array(0.70, 0.30), seed = 42L)
      trainingData = trainingData.cache()
      testData = testData.cache()

      logger.info("[MACHINE LEARNING]")
      val models = List[FlightModel](new FlightWeatherDecisionTree(), new FlightWeatherRandomForest())
      models.foreach(model => {
        logger.info(s"Training the model ${model.getName} on training data...")
        model.fit(trainingData)
        logger.info(s"Evaluating the model ${model.getName} on training data...")
        var prediction = model.evaluate(trainingData)
        logger.info(s"Performance of the model ${model.getName} on training data...")
        model.summarize(prediction)
        logger.info(s"Evaluating the model ${model.getName} on test data...")
        prediction = model.evaluate(testData)
        logger.info(s"Performance of the model ${model.getName} on test data...")
        model.summarize(prediction)
      })

    } catch {
      case e: Exception =>
        logger.info(e.toString)
    } finally {
      Utility.destroy()
    }

    logger.info(s"[END: ${(System.nanoTime() - t0) / 1000000000} s]")
  }
}
