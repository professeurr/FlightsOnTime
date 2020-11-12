import UdfUtility.{computeLineUdf, parseNumericalVariableUdf, parseSkyConditionUdf, parseTemperatureUdf, parseVisibilityUdf, parseWeatherTypeUdf, parseWindDirectionUdf}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler}
import org.apache.spark.sql.functions.{col, concat_ws, desc, lit, substring, unix_timestamp}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SaveMode}

class DataFeaturing(config: Configuration) {

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

  def prepFlights(mappingData: DataFrame): DataFeaturing = {
    val s = config.flightsPath.mkString(",")
    Utility.log(s"Loading flights records from $s")
    val delayColumns = Array("ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY", "CANCELLED", "DIVERTED")
    var data = Utility.sparkSession.read
      .option("header", "true")
      .option("inferSchema", "false")
      .csv(config.flightsPath: _*)

    // remove empty columns
    val cols = data.columns.toSet.toList.filter(c => !c.trim.startsWith("_c"))
    data = data.select(cols.map(c => col(c)): _*)
    // convert delay related columns to numerical type
    delayColumns.foreach(c => data = data.withColumn(c, col(c).cast(DoubleType)))
    data = data.na.fill(0.0, delayColumns)
    Utility.show(data)

    Utility.log("computing flights identifier (FL_ID)...")
    data = data
      .withColumn("FL_LINE", computeLineUdf($"ORIGIN_AIRPORT_ID", $"DEST_AIRPORT_ID"))
      .withColumn("FL_ID", concat_ws("_", $"FL_LINE", $"FL_DATE", $"OP_CARRIER_AIRLINE_ID", $"OP_CARRIER_FL_NUM"))
    Utility.show(data.select("FL_ID", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM"))

    Utility.log("some delays flights...")
    Utility.show(data.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "ARR_DELAY_NEW")
      .filter("ARR_DELAY_NEW > 40")
      .groupBy("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID").count()
      .orderBy(desc("count")))

    Utility.log("lines which have the longest delays...")
    Utility.show(data.select("FL_DATE", "FL_ID", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "ARR_DELAY_NEW")
      .filter("ARR_DELAY_NEW > 40")
      .withColumn("Delay (Hour)", ($"ARR_DELAY_NEW" / 60).cast(IntegerType))
      .orderBy(desc("ARR_DELAY_NEW")))

    data = data.drop("OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM")

    Utility.log("some diverted or cancelled flights")
    Utility.show(data.select("FL_ID", "CANCELLED", "DIVERTED")
      .filter("CANCELLED = 1 or DIVERTED = 1"))

    Utility.log("number of cancelled or diverted flights...")
    Utility.show(data.select("FL_ID", "CANCELLED", "DIVERTED")
      .filter("CANCELLED = 1 or DIVERTED = 0")
      .groupBy("CANCELLED", "DIVERTED").count())

    Utility.log("number of regular flights...")
    Utility.show(data.select("FL_ID", "CANCELLED", "DIVERTED")
      .filter("CANCELLED = 0 and DIVERTED = 0")
      .groupBy("CANCELLED", "DIVERTED").count())

    // remove cancelled and diverted data
    Utility.log("Removing cancelled and diverted flights (they are out of this analysis)...")
    data = data.filter("CANCELLED = 0 and DIVERTED = 0").drop("CANCELLED", "DIVERTED")

    Utility.log("mapping flights with the weather stations...")
    data = data.join(mappingData, $"ORIGIN_AIRPORT_ID" === $"AirportId", "inner")
      .drop("AirportId")

    Utility.log("computing FL_DEP_TIME and FL_ARR_TIME")
    data = data.withColumn("FL_DEP_TIME", unix_timestamp(concat_ws("", $"FL_DATE", $"CRS_DEP_TIME"), "yyyy-MM-ddHHmm").minus($"TimeZone"))
      .withColumn("FL_ARR_TIME", ($"FL_DEP_TIME" + $"CRS_ELAPSED_TIME" * 60).cast(LongType))
      .withColumn("year", substring($"FL_DATE", 0, 4))
      .withColumn("month", substring($"FL_DATE", 6, 2))

    val outputPath = config.persistPath + "data.flights.parquet"
    Utility.log(s"saving flights dataset into $outputPath")
    data.repartition(config.partitions, col("year"), col("month"))
      .write.mode(SaveMode.Overwrite)
      .partitionBy("year", "month")
      .parquet(outputPath)

    this
  }

  def prepWeather(mappingData: DataFrame): DataFeaturing = {

    Utility.log(s"Loading weather records from ${config.weatherPath.mkString(",")}")
    val weatherCondColumns = Array("SkyCondition", "DryBulbCelsius", "WeatherType", "StationPressure", "WindDirection", "Visibility", "RelativeHumidity", "WindSpeed")
    var data = Utility.sparkSession.read
      .option("header", "true")
      .option("inferSchema", "false")
      .csv(config.weatherPath: _*)
      .select(Array(col("WBAN"), col("Date"), col("Time")) ++ weatherCondColumns.map(c => col(c)): _*)
    Utility.log(s"weather initial number of records: ${Utility.count(data)};")
    Utility.show(data)

    Utility.log("getting timezones of each station and converting weather record time to utc...")
    data = data.join(mappingData, $"STATION_WBAN" === $"WBAN", "inner")
      .withColumnRenamed("AirportId", "AIRPORTID")
      .withColumn("WEATHER_TIME", unix_timestamp(concat_ws("", $"Date", $"Time"), "yyyyMMddHHmm").minus($"TimeZone"))
      .withColumn("year", substring($"Date", 0, 4))
      .withColumn("month", substring($"Date", 5, 2))
      .drop("STATION_WBAN", "WBAN", "Time", "Date", "TimeZone")
    Utility.log(s"weather number of records after the mapping with the stations: ${Utility.count(data)}")
    Utility.show(data)

    Utility.log("transforming/cleaning weather variables...")

    Utility.log("cleaning SkyCondition...")
    data = data.withColumn("SkyConditionLowCategory", parseSkyConditionUdf($"SkyCondition", lit(0)))
      .withColumn("SkyConditionMediumCategory", parseSkyConditionUdf($"SkyCondition", lit(1)))
      .withColumn("SkyConditionHighCategory", parseSkyConditionUdf($"SkyCondition", lit(2)))

    Utility.log("cleaning WeatherType...")
    data = data.withColumn("WeatherTypeCategory", parseWeatherTypeUdf($"WeatherType"))

    Utility.log("cleaning WindDirection...")
    data = data.withColumn("WindDirectionCategory", parseWindDirectionUdf($"WindDirection"))

    Utility.log("cleaning Visibility...")
    data = data.withColumn("Visibility", parseVisibilityUdf($"Visibility"))

    Utility.log("cleaning Temperature...")
    data = data.withColumn("DryBulbCelsius", parseTemperatureUdf($"DryBulbCelsius"))

    Utility.log("cleaning numerical variables...")
    var continuousVariables = Array("StationPressure", "RelativeHumidity", "WindSpeed")
    continuousVariables.foreach(c => {
      data = data.withColumn(c, parseNumericalVariableUdf(col(c)))
    })
    continuousVariables ++= Array("Visibility", "DryBulbCelsius")

    var pipelineModel: PipelineModel = null
    Utility.log("assembling weather conditions")
    val pipelinePath = s"${config.persistPath}pipeline.weather.pickle"
    // apply one-hot encoding of columns of the features
    val categoricalVariables = Array("SkyConditionLow", "SkyConditionMedium", "SkyConditionHigh", "WeatherType", "WindDirection")
    val oneHotEncoder = new OneHotEncoder()
      .setInputCols(categoricalVariables.map(c => c + "Category"))
      .setOutputCols(categoricalVariables.map(c => c + "Vect"))
    // creating vector assembler transformer to group the weather conditions features to one column
    val vectorAssembler = new VectorAssembler()
      .setInputCols(categoricalVariables.map(c => c + "Vect") ++ continuousVariables)
      .setOutputCol("WEATHER_COND")
    // the transformation pipeline that transforms categorial variables and assembles all variable into feature column
    val pipeline = new Pipeline().setStages(Array(oneHotEncoder, vectorAssembler))
    Utility.log("fitting features (one-hot encoding + vector assembler)...")
    pipelineModel = pipeline.fit(data)
    Utility.log(s"saving weather trained pipeline to a file '$pipelinePath'...")
    pipelineModel.write.overwrite.save(pipelinePath)

    Utility.log("transforming features (one-hot encoding + vector assembler)...")
    data = pipelineModel.transform(data)

    data = data.select("year", "month", "AIRPORTID", "WEATHER_TIME", "WEATHER_COND")
    Utility.show(data)

    val outputPath = config.persistPath + "data.weather.parquet"
    Utility.log(s"saving weather dataset into $outputPath")
    data.repartition(config.partitions, $"year", $"month")
      .write.mode(SaveMode.Overwrite)
      .partitionBy("year", "month")
      .parquet(outputPath)

    this
  }
}
