import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions.{col, lit}

object DataWrangling {

  import Utils.sparkSession.implicits._

  def loadData() : Unit = {

    val flightsPath = getClass.getResource("ontime_reporting_2013_12.csv").getPath
    val wbanAirportsPath = getClass.getResource("wban_airport_timezone.csv").getPath
    val weatherPath = getClass.getResource("weather_2013_12.csv").getPath


    var wbanAirportsData = Utils.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "true")
      .load(wbanAirportsPath)
      .cache()
    //Utils.log(wbanAirportsData)

    var flightsData = Utils.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "true")
      .load(flightsPath)
      //.limit(100)
      .drop("ORIGIN", "DEST", "DEST", "_c14")
      .na.fill(0.0, Seq("CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"))
      .cache()
    //Utils.log(flightsData)

    // assign time zones in order to normalize the times
    flightsData = flightsData.join(wbanAirportsData, $"ORIGIN_AIRPORT_ID" === $"AirportID", "inner")
      .drop("WBAN").withColumnRenamed("TimeZone", "ORIGIN_TZ").drop("AirportID")
      .join(wbanAirportsData, $"DEST_AIRPORT_ID" === $"AirportID", "inner")
      .drop("WBAN").withColumnRenamed("TimeZone", "DEST_TZ").drop("AirportID")
    //Utils.log(flightsData)

    // convert datetimes and normalize them to UTC
    flightsData = flightsData.withColumn("CRS_DEP_TIME", Utils.convertTimeUdf($"FL_DATE", $"CRS_DEP_TIME", $"ORIGIN_TZ"))
      .withColumn("DEP_TIME", Utils.convertTimeUdf($"FL_DATE", $"DEP_TIME", $"ORIGIN_TZ"))
      .withColumn("CRS_ARR_TIME", Utils.convertTimeUdf($"FL_DATE", $"CRS_ARR_TIME", $"DEST_TZ"))
      .withColumn("ARR_TIME", Utils.convertTimeUdf($"FL_DATE", $"ARR_TIME", $"DEST_TZ"))
      .drop("FL_DATE", "ORIGIN_TZ", "DEST_TZ")
    //Utils.log(flightsData)

    // compute OnTime column
    val threshold = 15.0 // the threshold is set to 15 minutes
    flightsData = flightsData.withColumn("FL_ONTIME", Utils.isOnTimeUdf(lit(threshold), $"CARRIER_DELAY", $"WEATHER_DELAY", $"NAS_DELAY", $"SECURITY_DELAY", $"LATE_AIRCRAFT_DELAY"))
      .drop("CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY")

    Utils.log(flightsData)

    var weatherData = Utils.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "true")
      .load(weatherPath)
      //.limit(10000)
      .cache()

    weatherData = weatherData.select("WBAN", "Date", "Time", "DryBulbCelsius", "RelativeHumidity", "WindDirection", "WindSpeed",
      "StationPressure", "SkyCondition", "Visibility", "WeatherType").as("w")
      .join(wbanAirportsData.as("a"), $"w.WBAN" === $"a.WBAN", "inner")
      .withColumn("WEATHER_TIME", Utils.convertWeatherTimeUdf($"Date", $"Time", $"TimeZone")) // normalize times
      .drop("Date", "WBAN", "TimeZone")

    Utils.log(weatherData)

    //cast
    val strColumns = Array("DryBulbCelsius", "RelativeHumidity", "WindDirection", "WindSpeed", "StationPressure", "Visibility")
    strColumns.foreach(c => {
      weatherData = weatherData.withColumn(c, col(c).cast("double"))
    })

    // And then transformed the categorical variables into one hot encoder vectors
    val categoricalFeaturesCols = Array("SkyCondition", "WeatherType")
    val indexers = categoricalFeaturesCols.map(colName =>
      new StringIndexer().setInputCol(colName)
        .setOutputCol(colName + "Category"))

    val assembler = new VectorAssembler()
      .setInputCols(Array("DryBulbCelsius", "RelativeHumidity", "WindDirection",
        "WindSpeed", "StationPressure", "SkyConditionCategory", "Visibility", "WeatherTypeCategory"))
      .setOutputCol("WEATHER_COND")
      .setHandleInvalid("skip")

    // We chain all transformers using a pipeline object
    val pipeline = new Pipeline()
      .setStages(indexers ++ Array(assembler))
    // We create here our model by fitting the pipeline with the input data
    val model = pipeline.fit(weatherData)
    // Apply the transformation
    weatherData = model.transform(weatherData).select("AirportId", "WEATHER_TIME", "WEATHER_COND")
    Utils.log(weatherData)

    //combine flights and weather
    val originWeatherCols = weatherData.columns.map(n => col(n).as(s"ORIGIN_$n"))
    val originWeatherData = weatherData.select(originWeatherCols: _*)
    val destWeatherCols = weatherData.columns.map(n => col(n).as(s"DEST_$n"))
    val destWeatherData = weatherData.select(destWeatherCols: _*)
    val airportWeatherData = flightsData.join(originWeatherData, $"ORIGIN_AIRPORT_ID" === $"ORIGIN_AirportID", "inner").drop("ORIGIN_AirportID")
      .join(destWeatherData, $"DEST_AIRPORT_ID" === $"DEST_AirportID", "inner").drop("DEST_AirportID")

    Utils.log(airportWeatherData)

    airportWeatherData
  }

}
