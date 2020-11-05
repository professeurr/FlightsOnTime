import UdfUtility._
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class DataLoader(config: Configuration) {

  import Utility.sparkSession.implicits._

  def loadStationsData(): DataFrame = {
    val schema: StructType = StructType(Array(
      StructField("AirportID", StringType, nullable = true),
      StructField("WBAN", StringType, nullable = true),
      StructField("TimeZone", LongType, nullable = true)))
    Utility.log(s"Loading weather stations ${config.wbanAirportsPath}")
    val data = Utility.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "false").schema(schema)
      .load(config.wbanAirportsPath)
      .withColumn("TimeZone", $"TimeZone" * 3600L)
      .withColumnRenamed("WBAN", "STATION_WBAN")
    Utility.show(data)
    data
  }

  def loadFlightData(path: Array[String], mappingData: DataFrame): DataFrame = {
    val s = path.mkString(",")
    Utility.log(s"Loading flights records from $s")
    var data = path.map(Utility.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "false")
      .load(_)).reduce(_ union _)
      .na.fill("0.0", Array("ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY", "CANCELLED", "DIVERTED"))

    // remove empty columns
    val cols = data.columns.toSet.toList.filter(c => !c.trim.startsWith("_c"))
    data = data.select(cols.map(c => col(c)): _*)
    Utility.log(s"number of flights: ${data.count()}; Flights columns: $cols")

    Utility.log("computing flights identifier (FL_ID)...")
    data = data.withColumn("FL_ID",
      concat_ws("_", $"OP_CARRIER_AIRLINE_ID", $"OP_CARRIER_FL_NUM", $"FL_DATE", $"ORIGIN_AIRPORT_ID", $"DEST_AIRPORT_ID"))
      .drop("OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM")
    //Utility.log(Data.schema.treeString)

    // remove cancelled and diverted data
    Utility.log("Removing cancelled and diverted flights (they are out of this analysis)...")
    data = data.filter("CANCELLED + DIVERTED = 0").drop("CANCELLED", "DIVERTED")
    Utility.log(s"flights dataset without cancelled and diverted flights: ${data.count()}")

    Utility.log("Removing non-weather related delayed records...")
    data = data.filter(s"ARR_DELAY_NEW <= ${config.flightsDelayThreshold} or  WEATHER_DELAY + NAS_DELAY >= ${config.flightsDelayThreshold} ")
    Utility.log(s"flights dataset without non-weather related delay records: ${data.count()}")

    Utility.log("computing FL_ONTIME flag (1=on-time; 0=delayed)")
    data = data.withColumn("FL_ONTIME", ($"ARR_DELAY_NEW" <= config.flightsDelayThreshold).cast(DoubleType))
      .drop("WEATHER_DELAY", "NAS_DELAY")

    Utility.log("mapping flights with the weather stations...")
    data = data.join(mappingData, $"ORIGIN_AIRPORT_ID" === $"AirportId", "inner")
      .drop("AirportId")
    Utility.log(s"flights data after the mapping: ${data.count()}")

    Utility.log("computing FL_DATE, FL_DEP_TIME and FL_ARR_TIME")
    data = data.withColumn("FL_DEP_TIME", unix_timestamp(concat_ws("", $"FL_DATE", $"CRS_DEP_TIME"), "yyyy-MM-ddHHmm").minus($"TimeZone"))
      .withColumn("FL_ARR_TIME", ($"FL_DEP_TIME" + $"CRS_ELAPSED_TIME" * 60).cast(LongType))
      .select("FL_ID", "FL_DEP_TIME", "FL_ARR_TIME", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_ONTIME")

    if (config.mlBalanceDataset)
      data = Utility.balanceDataset(data, "FL_ONTIME")
    if (config.flightsFrac > 0)
      data = data.sample(withReplacement = false, config.flightsFrac)

    Utility.log(s"number of flights used for the analysis: ${data.count()}")
    Utility.show(data)

    data
  }

  def loadWeatherData(path: Array[String], mappingData: DataFrame, testMode: Boolean): DataFrame = {
    val weatherCondColumns = Array("SkyCondition", "DryBulbCelsius", "WeatherType", "StationPressure", "WindDirection", "Visibility", "RelativeHumidity", "WindSpeed")

    Utility.log(s"Loading weather records from ${path.mkString(",")}")
    var data = path.map(Utility.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "false").load(_)).reduce(_ union _)
      .select(Array(col("WBAN"), col("Date"), col("Time")) ++ weatherCondColumns.map(c => col(c)): _*)
    //Utility.log(Data.schema.treeString)
    Utility.log(s"weather initial number of records: ${data.count()};")

    Utility.log("getting timezones of each station and converting weather record time to utc...")
    data = data.join(mappingData, $"STATION_WBAN" === $"WBAN", "inner")
      .withColumnRenamed("AirportId", "AIRPORTID")
      .withColumn("WEATHER_TIME", unix_timestamp(concat_ws("", $"Date", $"Time"), "yyyyMMddHHmm").minus($"TimeZone"))
      .drop("STATION_WBAN", "Time", "Date", "TimeZone", "WBAN")
    Utility.log(s"weather number of records after the mapping with the stations: ${data.count()}")
    Utility.show(data)

    val categoricalColumnsRange = 0 until 10
    val featureLabel = "WF"
    val featuresLabel = "WeatherFeatures"

    Utility.log("transforming categorical variables...")
    data = data.withColumn(featuresLabel, parseWeatherVariablesUdf(weatherCondColumns.map(c => col(c)): _*))
      .drop(weatherCondColumns: _*)
    Utility.show(data)

    categoricalColumnsRange.foreach(i => data = data.withColumn(featureLabel + i, col(featuresLabel)(i)))
    data = data.drop(featuresLabel)
    Utility.show(data)

    var pipelineModel: PipelineModel = null
    if (!testMode) {
      Utility.log("assembling weather conditions")
      val vectorAssembler = new VectorAssembler()
        .setInputCols(categoricalColumnsRange.map(i => featureLabel + i).toArray)
        .setOutputCol(featuresLabel)
      val featureIndexer = new VectorIndexer()
        .setInputCol(featuresLabel)
        .setOutputCol("WEATHER_COND")
        .setMaxCategories(9) // features with > 9 distinct values are treated as continuous

      val pipeline = new Pipeline().setStages(Array(vectorAssembler, featureIndexer))
      pipelineModel = pipeline.fit(data)
      Utility.log("saving weather pipeline to a file...")
      pipelineModel.write.overwrite.save(config.modelPath + "weather.pipeline")
    }
    else {
      Utility.log("loading weather pipeline from a file...")
      pipelineModel = PipelineModel.load(config.modelPath + "weather.pipeline")
    }
    data = pipelineModel.transform(data)
      .drop(categoricalColumnsRange.map(i => featureLabel + i): _*).drop(featuresLabel)
    Utility.show(data)

    data
  }

  def combineData(flightData: DataFrame, weatherData: DataFrame): DataFrame = {
    val tf: Int = 3600 * config.weatherTimeFrame

    Utility.log("joining flights data with weather data")
    var data = flightData.join(weatherData.as("dep"), $"ORIGIN_AIRPORT_ID" === $"dep.AIRPORTID", "inner")
      .join(weatherData.as("arr"), $"DEST_AIRPORT_ID" === $"arr.AIRPORTID", "inner")
      .where(s"dep.WEATHER_TIME >= FL_DEP_TIME - $tf and dep.WEATHER_TIME <= FL_DEP_TIME " +
        s"and arr.WEATHER_TIME >= FL_ARR_TIME - $tf and arr.WEATHER_TIME <= FL_ARR_TIME")
      .drop("dep.AIRPORTID", "arr.AIRPORTID", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID")

    Utility.log("grouping weather records for each flight")
    data = data.groupBy($"FL_ID", $"FL_DEP_TIME", $"FL_ARR_TIME", $"FL_ONTIME")
      .agg(collect_list($"dep.WEATHER_TIME").as("ORIGIN_WEATHER_TIME"),
        collect_list($"dep.WEATHER_COND").as("ORIGIN_WEATHER_COND"),
        collect_list($"arr.WEATHER_TIME").as("DEST_WEATHER_TIME"),
        collect_list($"arr.WEATHER_COND").as("DEST_WEATHER_COND"))
      .drop("FL_ID").cache()

    if (config.mlBalanceDataset)
      data = Utility.balanceDataset(data, "FL_ONTIME")

    Utility.log("filling missing weather records for each flight ...")
    data = data.withColumn("WEATHER_COND",
      fillWeatherDataUdf(
        $"FL_DEP_TIME", $"ORIGIN_WEATHER_TIME", $"ORIGIN_WEATHER_COND",
        $"FL_ARR_TIME", $"DEST_WEATHER_TIME", $"DEST_WEATHER_COND",
        lit(config.weatherTimeFrame), lit(config.weatherTimeStep)
      ))
      .drop("FL_DEP_TIME", "ORIGIN_WEATHER_TIME", "ORIGIN_WEATHER_COND",
        "FL_ARR_TIME", "DEST_WEATHER_TIME", "DEST_WEATHER_COND")
      .filter("WEATHER_COND is not null")
    Utility.log(s"data after filling: ${data.count()}")

    Utility.show(data, true)

    data
  }

}
