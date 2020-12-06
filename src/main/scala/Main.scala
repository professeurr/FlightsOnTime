import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.broadcast

object Main {

  def main(args: Array[String]): Unit = {

    val t0 = System.nanoTime()
    Utility.log("[START]")

    try {
      val config = Utility.config

      // we define here the set of models we want to train or test (in production mode)
      val models = List[FlightModel](
        //new FlightDelayLogisticRegression(config, "Logistic Regression", "ml/lr.model"),
        new FlightDelayRandomForest(config, "Random Forest", "ml/rf.model")
        //, new FlightDelayDecisionTree(config, "Decision Tree", "ml/dt.model")
        //new FlightDelayCrossValidation(config, "Cross Validation", "ml/cv.model")
      )

      for (mode <- config.executionModes) {
        config.mlMode = mode
        if (mode.equals("extract")) {
          config.flightsPath = config.flightsDataPath
          config.weatherPath = config.weatherDataPath
          extractFeatures(config)
        }
        else if (mode.equals("transform")) {
          config.flightsPath = config.flightsExtractPath
          config.weatherPath = config.weatherExtractPath
          transformData(config)
        }
        else if (mode.equals("train")) {
          val path = config.persistPath + "transform/data.balanced"
          val data = Utility.sparkSession.read.parquet(path)
          trainModel(models, data)
        }
        else if (mode.equals("evaluate")) {
          Utility.log("[MODEL EVALUATION]")

          config.flightsPath = config.flightsTestPath
          config.weatherPath = config.weatherTestPath
          extractFeatures(config)

          config.flightsPath = Array(config.persistPath + "evaluate/flights")
          config.weatherPath = Array(config.persistPath + "evaluate/weather")
          val data: DataFrame = transformData(config)

          evaluateModel(models, data)
        }
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

  def extractFeatures(configuration: Configuration): Unit = {
    val t0 = System.nanoTime()
    var t1 = System.nanoTime()
    Utility.log("[FEATURES EXTRACTION]")

    val featuresExtractor = new DataFeaturing(configuration)
    val stationsData = broadcast(featuresExtractor.extractStationsFeatures()) // loading weather stations

    t1 = System.nanoTime()
    featuresExtractor.extractWeatherFeatures(stationsData) // loading weather data
    Utility.log(s"[weatherDataExtraction elapsed in: ${elapsed(t1)}]")

    t1 = System.nanoTime()
    featuresExtractor.extractFlightsFeatures(stationsData) // loading flights data
    Utility.log(s"[flightDataExtraction elapsed in: ${elapsed(t1)}]")

    Utility.log(s"[data featuring elapsed in: ${elapsed(t0)}]")
  }

  def transformData(configuration: Configuration): DataFrame = {
    Utility.log("[DATA TRANSFORMATION]")

    val t0 = System.nanoTime()
    val dataLoader = new DataTransformer(configuration)

    var t1 = System.nanoTime()
    val flightData = dataLoader.transformFlightData()
    Utility.log(s"[flightDataLoad elapsed in: ${elapsed(t1)}]")

    t1 = System.nanoTime()
    val weatherData = dataLoader.transformWeatherData().cache()
    Utility.log(s"[weatherDataLoad elapsed in: ${elapsed(t1)}]")

    val data = dataLoader.joinData(flightData, weatherData)
    Utility.log(s"[data combining elapsed in: ${elapsed(t1)}]")

    Utility.log(s"[data transformation elapsed in: ${elapsed(t0)}]")

    data
  }

  def trainModel(models: List[FlightModel], data: DataFrame): Unit = {
    Utility.log("[MODEL TRAINING]")
    val t0 = System.nanoTime()

    // split the dataset into training and testing set
    var Array(trainingData, testData) = data.randomSplit(Array(0.70, 0.3), 42L)
    trainingData = trainingData.cache()
    testData = testData

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
    trainingData.unpersist(blocking = false)
    Utility.log(s"[model training elapsed in: ${elapsed(t0)}]")
  }

  def evaluateModel(models: List[FlightModel], data: DataFrame): Unit = {
    Utility.log("[MODEL PREDICTION]")
    val t0 = System.nanoTime()

    models.foreach(model => {
      Utility.log(s"Evaluating the model ${model.getName} on test data...")
      val prediction = model.evaluate(data)
      Utility.log(s"Performance of the model ${model.getName} on test data...")
      model.summarize(prediction)
    })
    Utility.log(s"[prediction elapsed in: ${elapsed(t0)}]")
  }

  def elapsed(t: Long): String = {
    s"${(System.nanoTime() - t) / 1000000000} s "
  }

}
