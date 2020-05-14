import java.io.File

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, _}

object DataWrangling {

  import Utils.sparkSession.implicits._

  def loadData(): DataFrame = {

    val weatherTimeDecay = 12; // 12h before the fight event (departure and arrival)
    val delayThreshold = 15.0 // the threshold of the flight delay is set to 15 minutes by default

    val flightsPath: String = getClass.getResource("ontime_reporting_2013_12_0.csv").getPath
    val weatherPath = getClass.getResource("weather_2013_12_0.csv").getPath
    val wbanAirportsPath = getClass.getResource("wban_airport_timezone.csv").getPath
    val outputPath = new File(flightsPath).getParent + "/output_2013_12.parquet"

    Utils.log("Loading relationship data between weather and airport data")
    val wbanAirportsData = Utils.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "true")
      .load(wbanAirportsPath)
      .cache()
    Utils.log(wbanAirportsData)

    Utils.log("preparing flights data...")
    val flightsData = prepareFlightsData(flightsPath, wbanAirportsData, delayThreshold)

    Utils.log("preparing weather data...")
    val weatherData = prepareWeatherData(weatherPath, wbanAirportsData)

    Utils.log("combining weather data with departure flight data...")
    val airportOriginWeatherData = prepareData(weatherData, flightsData, weatherTimeDecay, "ORIGIN_AIRPORT_ID", "DEP_TIME")
    Utils.log(airportOriginWeatherData)

    Utils.log("combining weather data with destination flight data...")
    val airportDestWeatherData = prepareData(weatherData, flightsData, weatherTimeDecay, "DEST_AIRPORT_ID", "ARR_TIME")
    Utils.log(airportDestWeatherData)

    Utils.log("joining the departure and the destination data...")
    var data = airportOriginWeatherData.as("origin").join(airportDestWeatherData.as("dest"), $"origin.FL_ID" === $"dest.FL_ID", "inner")
      .select(
        $"origin.FL_ID".as("FL_ID"),
        $"origin.FL_ONTIME".as("FL_ONTIME"),
        $"origin.WEATHER_COND".as("ORIGIN_WEATHER_COND"),
        $"dest.WEATHER_COND".as("DEST_WEATHER_COND"))
    Utils.log(data)
    //Utils.log(s"data.count(): ${data.count()}")

    Utils.log("balancing the dataset by selecting the same number of records for the on time flights and delayed ones...")
    val nbFlightsPerCategory = data.groupBy("FL_ONTIME").agg(count("FL_ID").as("FL_ONTIME_COUNT")).orderBy("FL_ONTIME")
    Utils.log(nbFlightsPerCategory)

    if (nbFlightsPerCategory.count() == 2) {
      nbFlightsPerCategory.show(numRows = 10, truncate = false)
      val nbDelayed = nbFlightsPerCategory.first().getAs[Long](1)
      val nbOnTime = data.count() - nbDelayed // TODO Replace count()
      if (nbOnTime > nbDelayed) data = data.where($"FL_ONTIME" === 0).union(data.where($"FL_ONTIME" === 1).limit(nbDelayed.toInt))
      else if (nbDelayed > nbOnTime) data = data.where($"FL_ONTIME" === 1).union(data.where($"FL_ONTIME" === 0).limit(nbDelayed.toInt))
      data = data.orderBy("FL_ONTIME")
      Utils.log(data)
    }

    // save the resulting data into json file
    Utils.log(s"saving the output to $outputPath ...")
    data.coalesce(1).write.format("parquet").mode("overwrite").save(outputPath)

    data
  }

  def prepareData(weatherData: DataFrame, flightsData: DataFrame, weatherTimeDecay: Int, airportLabel: String, flightTimeLabel: String):
  DataFrame = {
    val airportWeatherData = flightsData.join(weatherData, col(airportLabel) === $"AirportID", "inner")
      .where(Utils.hasImpactOnFlight(col(flightTimeLabel), $"WEATHER_TIME", lit(weatherTimeDecay))) // take the last weatherTimeDecay-th record
      .coalesce(2)
      .cache()
    Utils.log(airportWeatherData)

    Utils.log("grouping weather record timeserie...")
    var airportWeatherDataTimeserie = airportWeatherData
      //.where(col(flightTimeLabel) =!= null)
      .groupBy($"FL_ID")
      .agg(max(col(flightTimeLabel)).as(flightTimeLabel), collect_list($"WEATHER_TIME").as("WEATHER_TIMESERIE"))
      .coalesce(2)
    Utils.log(airportWeatherDataTimeserie)

    Utils.log("Filling up weather missing data...")
    var airportWeatherDataFilled = airportWeatherDataTimeserie
      .withColumn("WEATHER_TIMESERIE", Utils.fillWeatherMissingDataUdf(col(flightTimeLabel), $"WEATHER_TIMESERIE", lit(weatherTimeDecay)))
    Utils.log(airportWeatherDataFilled)

    Utils.log("Assembling airport+weather data...")
    airportWeatherDataFilled = airportWeatherDataTimeserie
      .withColumn("WEATHER_TIME", explode($"WEATHER_TIMESERIE")).as("filled")
      .join(airportWeatherData.as("original"), $"filled.WEATHER_TIME" === $"original.WEATHER_TIME", "inner")
      .groupBy("original.FL_ID").agg(max($"FL_ONTIME").as("FL_ONTIME"), collect_list("original.WEATHER_COND").as("WEATHER_COND"))
      .coalesce(1)
      .cache()
    Utils.log(airportWeatherDataFilled)
    airportWeatherDataFilled.show(truncate = false)

    airportWeatherDataFilled
  }

  def prepareFlightsData(flightsPath: String, wbanAirportsData: DataFrame, delayThreshold: Double): DataFrame = {

    Utils.log("Loading flights data")
    val flightsFeatures = Array("FL_ID", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "DEP_TIME", "ARR_TIME", "FL_ONTIME")
    val delayFields = Array("CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY")
    var flightsData = Utils.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "true")
      .load(flightsPath)
      .limit(10)
      .withColumn("FL_ID", concat($"ORIGIN", lit("_"), $"DEST", lit("_"), $"FL_DATE", lit("_"), $"DEP_TIME", lit("_"), $"CARRIER"))
      .sort("FL_ID")
      .drop("ORIGIN", "DEST", "_c14")
      .na.fill(0.0, delayFields)
      .cache()
    Utils.log(flightsData)

    Utils.log("Joining airports data with airport_wban_tz data...")
    // assign time zones in order to normalize the times
    flightsData = flightsData
      .join(wbanAirportsData, $"ORIGIN_AIRPORT_ID" === $"AirportID", "inner") // join with origin flights data
      .withColumn("DEP_TIME", Utils.convertTimeUdf($"FL_DATE", $"DEP_TIME", $"TimeZone")) // convert all times to UTC
      .drop("TimeZone", "AirportID", "WBAN")
      .join(wbanAirportsData, $"DEST_AIRPORT_ID" === $"AirportID", "inner") // join with destination flights data
      .withColumn("ARR_TIME", Utils.convertTimeUdf($"FL_DATE", $"ARR_TIME", $"TimeZone"))
      .cache()
    flightsData = flightsData.coalesce(1)
    Utils.log(flightsData)

    Utils.log("Computing the ontime flag based on the delay threshold " + delayThreshold + "...")
    flightsData = flightsData
      .withColumn("FL_ONTIME", Utils.isOnTimeUdf(lit(delayThreshold), $"CARRIER_DELAY", $"WEATHER_DELAY", $"NAS_DELAY", $"SECURITY_DELAY", $"LATE_AIRCRAFT_DELAY"))
      .select(flightsFeatures.map(x => col(x)): _*)
      .cache()
    Utils.log(flightsData)

    flightsData
  }

  def prepareWeatherData(weatherPath: String, wbanAirportsData: DataFrame): DataFrame = {
    Utils.log("Loading weather data...")
    val weatherCategoricalFeaturesCols = Array("SkyCondition", "WeatherType")
    val weatherNumericalFeaturesCols = Array("DryBulbCelsius", "RelativeHumidity", "WindDirection", "WindSpeed", "StationPressure", "Visibility")
    var weatherFeatures = weatherNumericalFeaturesCols ++ weatherCategoricalFeaturesCols

    var weatherData = Utils.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "true")
      .load(weatherPath)
      .cache()

    Utils.log("Converting some weather conditions values to double")
    weatherData = weatherNumericalFeaturesCols.foldLeft(weatherData) {
      case (acc, col) => acc.withColumn(col, weatherData(col).cast("double"))
    }
    weatherData = weatherData.coalesce(1)
    Utils.log(weatherData)

    Utils.log("Joining weather data with airport_wban_tz data and normalizing times")
    val cols = weatherFeatures ++ Array("WEATHER_TIME", "AirportID")
    weatherData = weatherData.as("w")
      .join(wbanAirportsData.as("a"), $"w.WBAN" === $"a.WBAN", "inner")
      .withColumn("WEATHER_TIME", Utils.convertWeatherTimeUdf($"Date", $"Time", $"TimeZone")) // convert record times to UTC
      .select(cols.map(c => col(c)): _*)
      .cache()
    weatherData = weatherData.coalesce(1)
    Utils.log(weatherData)

    // And then transformed the categorical variables into one hot encoder vectors
    val indexers = weatherCategoricalFeaturesCols.map(colName => new StringIndexer().setInputCol(colName)
      .setOutputCol(colName + "Category"))

    weatherFeatures = weatherNumericalFeaturesCols ++ indexers.map(c=>c.getOutputCol)
    val assembler = new VectorAssembler().setHandleInvalid("skip")
      .setInputCols(weatherFeatures).setOutputCol("WEATHER_COND")

    Utils.log("Applying transformers on the weather data")
    // We chain all transformers using a pipeline object
    val pipeline = new Pipeline().setStages(indexers ++ Array(assembler))
    // We create here our model by fitting the pipeline with the input data
    Utils.log("Fitting the model with the original weather data...")
    val model = pipeline.fit(weatherData)
    // Apply the transformation
    Utils.log("Applying the model on the weather data...")
    weatherData = model.transform(weatherData).select("AirportId", "WEATHER_TIME", "WEATHER_COND")
      .cache()
    Utils.log(weatherData)

    weatherData
  }

}
