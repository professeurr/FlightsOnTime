import UdfUtility._
import org.apache.log4j.Logger
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class DataLoader(config: Configuration) {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  import Utility.sparkSession.implicits._

  def loadMappingData(): DataFrame = {
    val schema: StructType = StructType(Array(
      StructField("AirportID", StringType, nullable = true),
      StructField("WBAN", StringType, nullable = true),
      StructField("TimeZone", LongType, nullable = true)))
    logger.info(s"Loading relationship data between weather and airport data from ${config.wbanAirportsPath}")
    Utility.sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .schema(schema)
      .load(config.wbanAirportsPath)
      .withColumn("TimeZone", $"TimeZone" * 3600L)
      .withColumnRenamed("WBAN", "MAPPING_WBAN")
  }

  def loadFlightData(mappingData: DataFrame): DataFrame = {
    val s = config.flightsPath.mkString(",")
    logger.info(s"Loading flights data from $s")
    var data = config.flightsPath.map(Utility.sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load(_)).reduce(_ union _)
      .na.fill("0.0", Array("ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY", "CANCELLED", "DIVERTED"))

    // remove empty columns
    val cols = data.columns.toSet.toList.filter(c => !c.trim.contains("_c"))
    data = data.select(cols.map(c => col(c)): _*)
    logger.info("Flights columns: " + cols)

    //logger.info(Data.schema.treeString)
    logger.info(s"full flights dataset: ${data.count()}")
    //Data.show(truncate = false)

    logger.info("computing FL_ID")
    data = data.withColumn("FL_ID",
      concat_ws("_", $"OP_CARRIER_AIRLINE_ID", $"OP_CARRIER_FL_NUM", $"FL_DATE", $"ORIGIN_AIRPORT_ID", $"DEST_AIRPORT_ID"))
      .drop("OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM")
    //logger.info(Data.schema.treeString)

    // remove cancelled and diverted data
    logger.info("Removing cancelled, diverted")
    data = data.filter("CANCELLED + DIVERTED = 0").drop("CANCELLED", "DIVERTED")
    logger.info(s"flights dataset without cancelled and diverted flights: ${data.count()}")

    logger.info("Removing non-weather related delay records")
    data = data.filter(s"ARR_DELAY_NEW <= ${config.flightsDelayThreshold} or  WEATHER_DELAY + NAS_DELAY >= ${config.flightsDelayThreshold} ")
    logger.info(s"flights dataset without non-weather related delay records: ${data.count()}")

    logger.info("computing FL_ONTIME")
    data = data.withColumn("FL_ONTIME", ($"ARR_DELAY_NEW" <= config.flightsDelayThreshold).cast(DoubleType))
      .drop("WEATHER_DELAY", "NAS_DELAY")

    logger.info("Mapping flights with the weather wban and timezone")
    data = data.join(mappingData, $"ORIGIN_AIRPORT_ID" === $"AirportId", "inner")
      .drop("AirportId")
    logger.info(s"flights data after the mapping: ${data.count()}")

    logger.info("computing FL_DATE, FL_DEP_TIME and FL_ARR_TIME")
    data = data.withColumn("FL_DEP_TIME", unix_timestamp(concat_ws("", $"FL_DATE", $"CRS_DEP_TIME"), "yyyy-MM-ddHHmm").minus($"TimeZone"))
      .withColumn("FL_ARR_TIME", ($"FL_DEP_TIME" + $"CRS_ELAPSED_TIME" * 60).cast(LongType))
      .select("FL_ID", "FL_DEP_TIME", "FL_ARR_TIME", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_ONTIME")

    if (config.mlBalanceDataset) {
      data = Utility.balanceDataset(data, "FL_ONTIME")
      logger.info(s"Data size after balancing: ${data.count()}")
    }
    if (config.flightsFrac > 0) {
      logger.info(s"sampling flights data with ratio ${config.flightsFrac}: ${data.count()}")
      data = data.sample(withReplacement = false, config.flightsFrac)
      logger.info(s"Flights data size used for the analysis: ${data.count()}")
    }

    logger.info("Flights cleaned data:")
    data.show(truncate = false)

    data
  }

  def loadWeatherData(mappingData: DataFrame): DataFrame = {
    val weatherCondColumns = Array("SkyCondition", "DryBulbCelsius", "WeatherType", "StationPressure", "WindDirection", "Visibility", "RelativeHumidity", "WindSpeed")

    val s = config.weatherPath.mkString(",")
    logger.info(s"Loading weather data from $s")
    var data = config.weatherPath.map(Utility.sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load(_)).reduce(_ union _)
      .select(Array(col("WBAN"), col("Date"), col("Time")) ++ weatherCondColumns.map(c => col(c)): _*)
    //logger.info(Data.schema.treeString)
    logger.info(s"weather initial data: ${data.count()}")
    logger.info(s"total wbans: ${data.select("WBAN").distinct().count()}")

    logger.info("getting timezones of each station and converting weather time to utc...")
    data = data.join(mappingData, $"MAPPING_WBAN" === $"WBAN", "inner")
      .withColumnRenamed("AirportId", "AIRPORTID")
      .withColumn("WEATHER_TIME", unix_timestamp(concat_ws("", $"Date", $"Time"), "yyyyMMddHHmm").minus($"TimeZone"))
      .drop("MAPPING_WBAN", "Time", "Date", "TimeZone", "WBAN")
    logger.info(s"weather data after the mapping: ${data.count()}")
    data.show(truncate = false)

    logger.info("transforming categorical variables...")
    data = data.withColumn("WeatherFeatures", parseWeatherVariablesUdf(weatherCondColumns.map(c => col(c)): _*))
      .drop(weatherCondColumns: _*)
    data.show(truncate = false)
/*
    logger.info("indexing categorical variables...")
    val featureIndexer = new VectorIndexer()
      .setInputCol("WeatherFeatures")
      .setOutputCol("WEATHER_COND")
      .setMaxCategories(9) // features with > 9 distinct values are treated as continuous
    val featureIndexerPipeline = new Pipeline().setStages(Array(featureIndexer))
    data = featureIndexerPipeline.fit(data).transform(data)
      .drop("WeatherFeatures")
    data.show(truncate = false)
*/
    val categoricalColumnsRange = 0 until 10
    categoricalColumnsRange.foreach(i => data = data.withColumn("WeatherFeature" + i, col("WeatherFeatures")(i)))
    data = data.drop("WeatherFeatures")
    data.show(truncate = false)

    logger.info("encoding categorical variables...")
    val oneHotEncoders = categoricalColumnsRange.map(i => new OneHotEncoder().setInputCol("WeatherFeature" + i).setOutputCol("WeatherFeatureVect" + i)).toArray
    val pipeline = new Pipeline().setStages(oneHotEncoders)
    data = pipeline.fit(data).transform(data)
      .drop(categoricalColumnsRange.map(i => "WeatherFeature" + i): _*)
    data.show(truncate = false)

    logger.info("assembling weather conditions")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(categoricalColumnsRange.map(i => "WeatherFeatureVect" + i).toArray)
      .setOutputCol("WEATHER_COND")
    val pipelineVectAss = new Pipeline().setStages(Array(vectorAssembler))
    data = pipelineVectAss.fit(data).transform(data)
      .drop(categoricalColumnsRange.map(i => "WeatherFeatureVect" + i): _*)
    println(data.schema.treeString)
    data.show(truncate = false)

    data
  }

  def combineData(flightData: DataFrame, weatherData: DataFrame): DataFrame = {
    val tf: Int = 3600 * config.weatherTimeFrame

    logger.info("joining flights data with weather data")
    var data = flightData.join(weatherData.as("dep"), $"ORIGIN_AIRPORT_ID" === $"dep.AIRPORTID", "inner")
      .join(weatherData.as("arr"), $"DEST_AIRPORT_ID" === $"arr.AIRPORTID", "inner")
      .where(s"dep.WEATHER_TIME >= FL_DEP_TIME - $tf and dep.WEATHER_TIME <= FL_DEP_TIME " +
        s"and arr.WEATHER_TIME >= FL_ARR_TIME - $tf and arr.WEATHER_TIME <= FL_ARR_TIME")
      .drop("dep.AIRPORTID", "arr.AIRPORTID", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID")
      .groupBy($"FL_ID", $"FL_DEP_TIME", $"FL_ARR_TIME", $"FL_ONTIME")
      .agg(collect_list($"dep.WEATHER_TIME").as("ORIGIN_WEATHER_TIME"),
        collect_list($"dep.WEATHER_COND").as("ORIGIN_WEATHER_COND"),
        collect_list($"arr.WEATHER_TIME").as("DEST_WEATHER_TIME"),
        collect_list($"arr.WEATHER_COND").as("DEST_WEATHER_COND"))
      .drop("FL_ID")
    logger.info(s"flights data with weather data: ${data.count()}")

    //    if (config.mlBalanceDataset) {
    //      data = Utility.balanceDataset(data, "FL_ONTIME")
    //      logger.info(s"data after balancing: ${data.count()}")
    //    }

    logger.info("filling weather data...")
    data = data.withColumn("WEATHER_COND",
      fillWeatherDataUdf(
        $"FL_DEP_TIME", $"ORIGIN_WEATHER_TIME", $"ORIGIN_WEATHER_COND",
        $"FL_ARR_TIME", $"DEST_WEATHER_TIME", $"DEST_WEATHER_COND",
        lit(config.weatherTimeFrame), lit(config.weatherTimeStep)
      ))
      .drop("FL_DEP_TIME", "ORIGIN_WEATHER_TIME", "ORIGIN_WEATHER_COND",
        "FL_ARR_TIME", "DEST_WEATHER_TIME", "DEST_WEATHER_COND")
      .filter("WEATHER_COND is not null")
    logger.info(s"data after filling: ${data.count()}")

    data.show(truncate = true)

    data
  }

}
