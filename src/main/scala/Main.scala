import org.apache.spark.sql.functions.{broadcast, when}
import org.apache.spark.sql.{DataFrame, SaveMode}

object Main {

  import Utility.sparkSession.implicits._

  def main(args: Array[String]): Unit = {

    val t0 = System.nanoTime()
    Utility.log("[START]")

    try {
      val config = Utility.config

      // we define here the set of models we want to train or test (in production mode)
      val models = List[FlightModel](
        //new FlightDelayLogisticRegression(config, "Logistic Regression", "ml/lr.model"),
        //new FlightDelayLinearRegressor(config, "Linear Regressor", "ml/glr.model"),
        new FlightDelayRandomForestRegressor(config, "Random Forest Regressor", "ml/rfr.model"),
        //new FlightDelayGBTRegressor(config, "GBT Regressor", "ml/gbt.model"),
        new FlightDelayRandomForestClassifier(config, "Random Forest Classifier", "ml/rfc.model")
        //, new FlightDelayDecisionTree(config, "Decision Tree", "ml/dt.model")
        //new FlightDelayCrossValidation(config, "Cross Validation", "ml/cv.model")
      )

      for (mode <- config.executionModes) {
        config.mlMode = mode
        if (mode.equals("extract")) {
          extractFeatures(config, config.flightsTrainPath, config.weatherTrainPath)
        }
        else if (mode.equals("transform")) {
          transformData(config, config.persistPath + "/extract")
        }
        else if (mode.equals("train")) {
          var data = Utility.readParquet(config.persistPath + "transform/data")
          data = balanceData(config, data)
          trainModel(models, data)
        }
        else if (mode.equals("evaluate")) {
          Utility.log("[MODEL EVALUATION]")
          evaluateModel(config, models)
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

  def extractFeatures(configuration: Configuration, flightPaths: Array[String], weatherPaths: Array[String]): Unit = {
    val t0 = System.nanoTime()
    var t1 = System.nanoTime()
    Utility.log("[FEATURES EXTRACTION]")

    val featuresExtractor = new DataFeaturing(configuration)
    Utility.log("extracting stations data...")
    val stationsData = broadcast(Utility.readCsv(configuration.wbanAirportsPath)) // loading weather stations

    t1 = System.nanoTime()
    Utility.log("extracting weather data...")
    val weatherData = Utility.readCsv(weatherPaths: _*)
    featuresExtractor.extractWeatherFeatures(weatherData, stationsData) // loading weather data
    Utility.log(s"[weatherDataExtraction elapsed in: ${elapsed(t1)}]")

    t1 = System.nanoTime()
    Utility.log("extracting flights data...")
    val flightData = Utility.readCsv(flightPaths: _*)
    featuresExtractor.extractFlightsFeatures(flightData, stationsData) // loading flights data
    Utility.log(s"[flightDataExtraction elapsed in: ${elapsed(t1)}]")

    Utility.log(s"[data featuring elapsed in: ${elapsed(t0)}]")
  }

  def transformData(configuration: Configuration, root:String): DataFrame = {
    Utility.log("[DATA TRANSFORMATION]")

    val t0 = System.nanoTime()
    val dataLoader = new DataTransformer(configuration)
    val flightDataDep = Utility.readParquet(root + "/flights.dep")
    val flightDataArr = Utility.readParquet(root + "/flights.arr")
    val weatherData = Utility.readParquet(root + "/weather")
    val data = dataLoader.joinData(flightDataDep, flightDataArr, weatherData)

    Utility.log(s"[data transformation elapsed in: ${elapsed(t0)}]")

    data
  }

  def trainModel(models: List[FlightModel], data: DataFrame): Unit = {
    Utility.log("[MODEL TRAINING]")
    val t0 = System.nanoTime()

    // split the dataset into training and testing set
    var Array(trainingData, testData) = data.randomSplit(Array(0.75, 0.25), 42L)
    trainingData = trainingData.cache()

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

  def evaluateModel(config: Configuration, models: List[FlightModel]): Unit = {
    Utility.log("[MODEL PREDICTION]")
    val t0 = System.nanoTime()

    extractFeatures(config, config.flightsTestPath, config.weatherTestPath)
    var data: DataFrame = transformData(config, config.persistPath + "/evaluate")
    data = balanceData(config, data)

    models.foreach(model => {
      Utility.log(s"Evaluating the model ${model.getName} on test data...")
      val prediction = model.evaluate(data)
      Utility.log(s"Performance of the model ${model.getName} on test data...")
      model.summarize(prediction)
    })
    Utility.log(s"[prediction elapsed in: ${elapsed(t0)}]")
  }

  def balanceData(configuration: Configuration, dataFrame: DataFrame): DataFrame = {
    Utility.log("[DATA BALANCING]")
    val t0 = System.nanoTime()

    var data = dataFrame.withColumn("delayed", when($"DELAY" > configuration.flightsDelayThreshold, 1.0).otherwise(0.0))

    val ontimeCount = data.filter(s"delayed = 0.0").count().toDouble
    val delayedCount = data.filter(s"delayed = 1.0").count().toDouble
    Utility.log(s"ontime=$ontimeCount, delayed=$delayedCount")

    Utility.log("balancing...")
    val outputPath = configuration.persistPath + configuration.mlMode + "/data"
    val fractions = if (ontimeCount >= delayedCount) Map(1.0 -> 1.0, 0.0 -> delayedCount / ontimeCount) else Map(1.0 -> ontimeCount / delayedCount, 0.0 -> 1.0)
    data = data.stat.sampleBy("delayed", fractions, 42L)
    data.repartition(configuration.partitions)
      .write.mode(SaveMode.Overwrite)
      .parquet(outputPath + ".balanced")

    Utility.log(s"saving balanced dataset into $outputPath")
    data = Utility.readParquet(outputPath + ".balanced")
    Utility.log(s"[data balancing elapsed in: ${elapsed(t0)}]")
    data
  }

  def elapsed(t: Long): String = {
    s"${(System.nanoTime() - t) / 1000000000} s "
  }

}
