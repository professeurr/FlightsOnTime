import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object DataWrangling {

  import Utils.sparkSession.implicits._

  def loadData(): DataFrame = {

    val flightsPath = getClass.getResource("ontime_reporting_2013_12.csv").getPath
    val wbanAirportsPath = getClass.getResource("wban_airport_timezone.csv").getPath
    val weatherPath = getClass.getResource("weather_2013_12.csv").getPath

    Utils.log("Loadind relationship between weather and airport data")
    var wbanAirportsData = Utils.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "true")
      .load(wbanAirportsPath)
      .cache()
    //Utils.log(wbanAirportsData)

    Utils.log("Loading flights data")
    var flightsData = Utils.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "true")
      .load(flightsPath)
      //.limit(100)
      .withColumn("FL_ID", monotonically_increasing_id())
      .drop("ORIGIN", "DEST", "DEST", "_c14")
      .na.fill(0.0, Seq("CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"))
      .cache()
    //Utils.log(flightsData)

    Utils.log("Joining airports data with airport_wban_tz data")
    // assign time zones in order to normalize the times
    flightsData = flightsData.join(wbanAirportsData, $"ORIGIN_AIRPORT_ID" === $"AirportID", "inner")
      .drop("WBAN").withColumnRenamed("TimeZone", "ORIGIN_TZ").drop("AirportID")
      .join(wbanAirportsData, $"DEST_AIRPORT_ID" === $"AirportID", "inner")
      .drop("WBAN").withColumnRenamed("TimeZone", "DEST_TZ").drop("AirportID")
      .cache()
    //Utils.log(flightsData)

    Utils.log("Normalizing times")
    // convert datetimes and normalize them to UTC
    flightsData = flightsData.withColumn("CRS_DEP_TIME", Utils.convertTimeUdf($"FL_DATE", $"CRS_DEP_TIME", $"ORIGIN_TZ"))
      .withColumn("DEP_TIME", Utils.convertTimeUdf($"FL_DATE", $"DEP_TIME", $"ORIGIN_TZ"))
      .withColumn("CRS_ARR_TIME", Utils.convertTimeUdf($"FL_DATE", $"CRS_ARR_TIME", $"DEST_TZ"))
      .withColumn("ARR_TIME", Utils.convertTimeUdf($"FL_DATE", $"ARR_TIME", $"DEST_TZ"))
      .drop("FL_DATE", "ORIGIN_TZ", "DEST_TZ")
      .cache()
    //Utils.log(flightsData)

    Utils.log("Computing the ontime flag")
    // compute OnTime column
    val threshold = 15.0 // the threshold is set to 15 minutes
    flightsData = flightsData.withColumn("FL_ONTIME", Utils.isOnTimeUdf(lit(threshold), $"CARRIER_DELAY", $"WEATHER_DELAY", $"NAS_DELAY", $"SECURITY_DELAY", $"LATE_AIRCRAFT_DELAY"))
      .drop("CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY")
      .cache()

    //Utils.log(flightsData)

    Utils.log("Loading weather data")
    var weatherData = Utils.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "true")
      .load(weatherPath)
      //.limit(10000)
      .cache()

    Utils.log("Joining weather data with airport_wban_tz data and normalizing times")
    weatherData = weatherData.select("WBAN", "Date", "Time", "DryBulbCelsius", "RelativeHumidity", "WindDirection", "WindSpeed",
      "StationPressure", "SkyCondition", "Visibility", "WeatherType").as("w")
      .join(wbanAirportsData.as("a"), $"w.WBAN" === $"a.WBAN", "inner")
      .withColumn("WEATHER_TIME", Utils.convertWeatherTimeUdf($"Date", $"Time", $"TimeZone")) // normalize times
      .drop("Date", "WBAN", "TimeZone")
      .cache()

    //Utils.log(weatherData)
    Utils.log("Converting weather conditions values to double")
    //cast
    val strColumns = Array("DryBulbCelsius", "RelativeHumidity", "WindDirection", "WindSpeed", "StationPressure", "Visibility")
    strColumns.foreach(c => {
      weatherData = weatherData.withColumn(c, col(c).cast("double"))
    })
    weatherData = weatherData.cache()

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

    Utils.log("Applying transformers on the weather data")
    // We chain all transformers using a pipeline object
    val pipeline = new Pipeline()
      .setStages(indexers ++ Array(assembler))
    // We create here our model by fitting the pipeline with the input data
    val model = pipeline.fit(weatherData)
    // Apply the transformation
    weatherData = model.transform(weatherData).select("AirportId", "WEATHER_TIME", "WEATHER_COND")
      .cache()
    //Utils.log(weatherData)

    Utils.log("joining weather data with airport data")
    //combine flights and weather
    val originWeatherCols = weatherData.columns.map(n => col(n).as(s"ORIGIN_$n"))
    val originWeatherData = weatherData.select(originWeatherCols: _*)
      .cache()
    val destWeatherCols = weatherData.columns.map(n => col(n).as(s"DEST_$n"))
    val destWeatherData = weatherData.select(destWeatherCols: _*)
      .cache()
    var airportWeatherData = flightsData.join(originWeatherData, $"ORIGIN_AIRPORT_ID" === $"ORIGIN_AirportID", "inner").drop("ORIGIN_AirportID")
      .join(destWeatherData, $"DEST_AIRPORT_ID" === $"DEST_AirportID", "inner").drop("DEST_AirportID")
      .cache()

    Utils.log("selecting only the weather registered times before the departure and arrival")
    // remove unnecessary weather data after the actual departure and arrival times
    airportWeatherData = airportWeatherData.where("DEP_TIME >= ORIGIN_WEATHER_TIME AND ARR_TIME >= DEST_WEATHER_TIME")
      .cache()

    //Utils.log(airportWeatherData)

    Utils.log("remove unnecessary features")
    // for the delay (on time) analysis, we just need the weather conditions,
    // the ontime flag and the flight id (FL_ID) in order to combine later the weather conditions of the same flight.
    // We should remove the others columns
    airportWeatherData = airportWeatherData.drop("ORIGIN_WEATHER_TIME", "DEST_WEATHER_TIME", "ORIGIN_AIRPORT_ID",
      "DEST_AIRPORT_ID", "CRS_DEP_TIME", "DEP_TIME", "CRS_ARR_TIME", "ARR_TIME", "")
      .cache()

    //Utils.log(airportWeatherData)

    airportWeatherData = airportWeatherData.limit(2).cache()

    Utils.log("Group together the weather conditions for the same flight")
    /*val timeWindow = Window.partitionBy("FL_ID")
    airportWeatherData = airportWeatherData
      .withColumn("ORIGIN_WEATHER_COND", collect_list($"ORIGIN_WEATHER_COND").over(timeWindow))
      .withColumn("DEST_WEATHER_COND", collect_list($"DEST_WEATHER_COND").over(timeWindow))
      .cache()*/

    airportWeatherData = airportWeatherData.groupBy("FL_ID", "ORIGIN_WEATHER_COND", "DEST_WEATHER_COND")
      .agg(max("FL_ONTIME"))
      //.agg(collect_list($"ORIGIN_WEATHER_COND").as("ORIGIN_WEATHER_COND"))
      //.agg(collect_list($"DEST_WEATHER_COND").as("DEST_WEATHER_COND"))
      .cache()

    Utils.log(airportWeatherData)

    Utils.log("Remove the flight identifier")
    airportWeatherData = airportWeatherData.drop("FL_ID").cache()

    airportWeatherData
  }

}
