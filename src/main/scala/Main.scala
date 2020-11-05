import org.apache.spark.sql.functions.broadcast

object Main {

  def main(args: Array[String]): Unit = {

    val t0 = System.nanoTime()
    Utility.log("[START]")

    try {
      val config = Utility.config

      val dataLoader = new DataLoader(config)
      // broadcast this dataset which is small compare to flights and weather ones. Broadcasting it will significantly speed up the join operations
      val airportWbanData = broadcast(dataLoader.loadStationsData())

      if (config.trainModel) {
        Utility.log("[TRAINING DATA PREPARATION]")
        val weatherData = dataLoader.loadWeatherData(config.weatherPath, airportWbanData, false).cache()
        val flightData = dataLoader.loadFlightData(config.flightsPath, airportWbanData).cache()
        val data = dataLoader.combineData(flightData, weatherData).cache()

        var Array(trainingData, testData) = data.randomSplit(Array(0.70, 0.30))
        trainingData = trainingData.cache()
        testData = testData.cache()

        Utility.log("[TRAINING MACHINE LEARNING]")
        val models = List[FlightModel](new FlightWeatherDecisionTree(), new FlightWeatherRandomForest())
        models.foreach(model => {
          Utility.log(s"Training the model ${model.getName} on training data...")
          model.fit(trainingData)
          model.save(config.modelPath)
          Utility.log(s"Evaluating the model ${model.getName} on training data...")
          var prediction = model.evaluate(trainingData)
          Utility.log(s"Performance of the model ${model.getName} on training data...")
          model.summarize(prediction)
          Utility.log(s"Evaluating the model ${model.getName} on test data...")
          prediction = model.evaluate(testData)
          Utility.log(s"Performance of the model ${model.getName} on test data...")
          model.summarize(prediction)
        })
        Utility.sparkSession.sqlContext.clearCache()
      }

      if (config.testModel) {
        Utility.log("[OFFLINE TESTING]")
        Utility.log("[TESTING DATA PREPARATION]")
        // broadcast this dataset which is small compare to flights and weather ones. Broadcasting it will significantly speed up the join operations
        val testingWeatherData = dataLoader.loadWeatherData(config.weatherTestPath, airportWbanData, true).cache()
        val testingFlightData = dataLoader.loadFlightData(config.flightsTestPath, airportWbanData).cache()
        val testingData = dataLoader.combineData(testingFlightData, testingWeatherData).cache()

        Utility.log("[TESTING MACHINE LEARNING]")
        val testingModels = List[FlightModel](new FlightWeatherDecisionTree(), new FlightWeatherRandomForest())
        testingModels.foreach(model => {
          Utility.log(s"Evaluating the model ${model.getName} on testing data...")
          val prediction = model.evaluate(config.modelPath, testingData)
          Utility.log(s"Performance of the model ${model.getName} on testing data...")
          model.summarize(prediction)
        })
        Utility.sparkSession.sqlContext.clearCache()
      }
    } catch {
      case e: Exception =>
        Utility.log(e.toString)
    } finally {
      Utility.destroy()
    }

    Utility.log(s"[END: ${(System.nanoTime() - t0) / 1000000000} s]")
  }
}
