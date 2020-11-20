import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.broadcast

object Main {

  def main(args: Array[String]): Unit = {

    val dt = 1000000000
    val t0 = System.nanoTime()
    Utility.log("[START]")

    try {
      val config = Utility.config

      var data: DataFrame = null

      val dataLoader = new DataLoader(config)
      // we define here the set of models we want to train or test (in production mode)
      val models = List[FlightModel](
        new FlightDelayRandomForest(config)
        //, new FlightDelayCrossValidation(config)
      ) //(new FlightDelayCrossValidation, new FlightDelayDecisionTree(), new FlightDelayRandomForest() /*, new FlightWeatherLogisticRegression()*/)

      if (config.mlMode.contains("data")) {
        Utility.log("[DATA PREPARATION]")
        // broadcast this dataset which is small compare to flights and weather ones. Broadcasting it will significantly speed up the join operations
        val airportWbanData = broadcast(dataLoader.loadStationsData())
        new DataFeaturing(config)
          .preloadFlights(airportWbanData)
        .preloadWeather(airportWbanData)
      }
      else if (config.mlMode.contains("train")) {
        val flightData = dataLoader.loadFlightData().cache()
        Utility.log(s"[flightDataLoad elapsed: ${(System.nanoTime() - t0) / dt} s]")

        val t1 = System.nanoTime()
        val weatherData = dataLoader.loadWeatherData().cache()
        Utility.log(s"[weatherDataLoad elapsed: ${(t1 - t0) / dt} s]")

        data = dataLoader.combineData(flightData, weatherData).cache()
        Utility.log(s"[data combining elapsed: ${(System.nanoTime() - t1) / dt} s]")
        Utility.log(s"[data preparation elapsed: ${(System.nanoTime() - t0) / dt} s]")

        // split the dataset into training and testing set
        var Array(trainingData, testData) = data.randomSplit(Array(0.70, 0.30))
        trainingData = trainingData.cache()
        testData = testData.cache()

        Utility.log("[MACHINE LEARNING MODEL]")
        models.foreach(model => {
          Utility.log(s"Training the model ${model.getName} on training data...")
          model.fit(trainingData).save()

          Utility.log(s"Evaluating the model ${model.getName} on training data...")
          var prediction = model.evaluate(trainingData)
          Utility.log(s"Performance of the model ${model.getName} on training data...")
          model.summarize(prediction)

          Utility.log(s"Evaluating the model ${model.getName} on test data...")
          prediction = model.evaluate(testData)
          Utility.log(s"Performance of the model ${model.getName} on test data...")
          model.summarize(prediction)
        })
      }

      //      if (config.mlMode.contains("test")) {
      //        Utility.log("[OFFLINE TESTING]")
      //
      //        Utility.log("[TESTING DATA PREPARATION]")
      //        // broadcast this dataset which is small compare to flights and weather ones. Broadcasting it will significantly speed up the join operations
      //        val testingWeatherData = dataLoader.loadWeatherData(config.weatherPath, airportWbanData).cache()
      //        val testingFlightData = dataLoader.loadFlightData(config.flightsPath, airportWbanData).cache()
      //        val testingData = dataLoader.combineData(testingFlightData, testingWeatherData).cache()
      //
      //        Utility.log("[TESTING MACHINE LEARNING]")
      //        models.foreach(model => {
      //          Utility.log(s"Evaluating the model ${model.getName} on offline testing data...")
      //          val prediction = model.evaluate(testingData)
      //          Utility.log(s"Performance of the model ${model.getName} on offline testing data...")
      //          model.summarize(prediction)
      //        })
      //      }
    }

    catch {
      case e: Exception =>
        Utility.log(e.toString)
    }
    finally {
      Utility.destroy()
    }

    Utility.log(s"[END: ${(System.nanoTime() - t0) / 1000000000} s]")
  }

}
