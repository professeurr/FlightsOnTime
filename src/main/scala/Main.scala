import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.broadcast

object Main {

  def main(args: Array[String]): Unit = {

    val t0 = System.nanoTime()
    Utility.log("[START]")

    try {
      val config = Utility.config

      var data: DataFrame = null

      val dataLoader = new DataLoader(config)

      if (config.mlMode.contains("data")) {
        Utility.log("[DATA PREPARATION]")
        // broadcast this dataset which is small compare to flights and weather ones. Broadcasting it will significantly speed up the join operations
        val airportWbanData = broadcast(dataLoader.loadStationsData())
        new DataFeaturing(config)
          .preloadFlights(airportWbanData)
          .preloadWeather(airportWbanData)
      }
      if (config.mlMode.contains("train")) {
        var t1 = System.nanoTime()
        val flightData = dataLoader.loadFlightData().cache()
        Utility.log(s"[flightDataLoad elapsed: ${elapsed(t1)}]")

        t1 = System.nanoTime()
        val weatherData = dataLoader.loadWeatherData().cache()
        Utility.log(s"[weatherDataLoad elapsed: ${elapsed(t1)}]")

        data = dataLoader.combineData(flightData, weatherData).cache()
        Utility.log(s"[data combining elapsed: ${elapsed(t1)}]")
        Utility.log(s"[data preparation elapsed: ${elapsed(t0)}]")

        // split the dataset into training and testing set
        var Array(trainingData, testData) = data.randomSplit(Array(0.75, 0.25))
        trainingData = trainingData.cache()
        testData = testData.cache()

        Utility.log("[MACHINE LEARNING MODEL]")
        // we define here the set of models we want to train or test (in production mode)
        val models = List[FlightModel](
          new FlightDelayRandomForest(config)
          //, new FlightDelayDecisionTree(config)
          //, new FlightDelayCrossValidation(config)
        )
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
    }

    catch {
      case e: Exception =>
        Utility.log(e.toString)
    }
    finally {
      Utility.destroy()
    }

    Utility.log(s"[END: ${elapsed(t0)}]")
  }

  def elapsed(t: Long): String = {
    s"${(System.nanoTime() - t) / 1000000000} s "
  }
}
