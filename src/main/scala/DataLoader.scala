import UdfUtility._
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler}
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
    val delayColumns = Array("ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY", "CANCELLED", "DIVERTED")
    var data = path.map(Utility.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "false")
      .load(_)).reduce(_ union _)

    // remove empty columns
    val cols = data.columns.toSet.toList.filter(c => !c.trim.startsWith("_c"))
    data = data.select(cols.map(c => col(c)): _*)
    // convert delay related columns to numerical type
    delayColumns.foreach(c => data = data.withColumn(c, col(c).cast(DoubleType)))
    data = data.na.fill(0.0, delayColumns)
    //Utility.log(s"number of flights: ${Utility.count(data)}; Flights columns: $cols")
    Utility.show(data)

    Utility.log("computing flights identifier (FL_ID)...")
    data = data.withColumn("FL_ID",
      concat_ws("_", computeLineUdf($"ORIGIN_AIRPORT_ID", $"DEST_AIRPORT_ID"), $"FL_DATE", $"OP_CARRIER_AIRLINE_ID", $"OP_CARRIER_FL_NUM"))
      .drop("OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM")
    //Utility.log(Data.schema.treeString)
    Utility.show(data.select("FL_ID", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM"))

    // remove cancelled and diverted data
    Utility.log("Removing cancelled and diverted flights (they are out of this analysis)...")
    data = data.filter("CANCELLED + DIVERTED = 0").drop("CANCELLED", "DIVERTED")
    //Utility.log(s"flights dataset without cancelled and diverted flights: ${Utility.count(data)}")

    Utility.log("Removing non-weather related delayed records...")

    data = data.filter(s"ARR_DELAY_NEW <= ${config.flightsDelayThreshold} " +
      s"or  WEATHER_DELAY + NAS_DELAY >= ${config.flightsDelayThreshold} ")
    //Utility.log(s"flights dataset without non-weather related delay records: ${Utility.count(data)}")

    Utility.log("computing FL_ONTIME flag (1=on-time; 0=delayed)")
    data = data.withColumn("FL_ONTIME", ($"ARR_DELAY_NEW" <= config.flightsDelayThreshold).cast(DoubleType))
      .drop("WEATHER_DELAY", "NAS_DELAY")
    Utility.show(data.select("FL_ID", "FL_DATE", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_ONTIME"))

    Utility.log("mapping flights with the weather stations...")
    data = data.join(mappingData, $"ORIGIN_AIRPORT_ID" === $"AirportId", "inner")
      .drop("AirportId")
    Utility.show(data.select("FL_ID", "FL_ONTIME", "FL_DATE", "CRS_DEP_TIME", "TimeZone"))
    //Utility.log(s"flights data after the mapping: ${Utility.count(data)}")

    Utility.log("computing FL_DATE, FL_DEP_TIME and FL_ARR_TIME")
    data = data.withColumn("FL_DEP_TIME", unix_timestamp(concat_ws("", $"FL_DATE", $"CRS_DEP_TIME"), "yyyy-MM-ddHHmm").minus($"TimeZone"))
      .withColumn("FL_ARR_TIME", ($"FL_DEP_TIME" + $"CRS_ELAPSED_TIME" * 60).cast(LongType))
      .select("FL_ID", "FL_DEP_TIME", "FL_ARR_TIME", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_ONTIME")

    Utility.show(data.filter("FL_ONTIME = 0").limit(12).union(data.filter("FL_ONTIME = 1").limit(10)).sample(withReplacement = false, 0.99))

    if (config.flightsFrac > 0 && config.flightsFrac != 1)
      data = data.sample(withReplacement = false, config.flightsFrac)

    //Utility.log(s"number of flights used for the analysis: ${Utility.count(data)}")
    //Utility.show(data)
    data

  }

  def loadWeatherData(path: Array[String], mappingData: DataFrame, testMode: Boolean): DataFrame = {

    Utility.log(s"Loading weather records from ${path.mkString(",")}")
    val weatherCondColumns = Array("SkyCondition", "DryBulbCelsius", "WeatherType", "StationPressure", "WindDirection", "Visibility", "RelativeHumidity", "WindSpeed")
    var data = path.map(Utility.sparkSession.read.format("csv")
      .option("header", "true").option("inferSchema", "false").load(_)).reduce(_ union _)
      .select(Array(col("WBAN"), col("Date"), col("Time")) ++ weatherCondColumns.map(c => col(c)): _*)
    Utility.log(s"weather initial number of records: ${Utility.count(data)};")
    Utility.show(data)

    Utility.log("getting timezones of each station and converting weather record time to utc...")
    data = data.join(mappingData, $"STATION_WBAN" === $"WBAN", "inner")
      .withColumnRenamed("AirportId", "AIRPORTID")
      .withColumn("WEATHER_TIME", unix_timestamp(concat_ws("", $"Date", $"Time"), "yyyyMMddHHmm").minus($"TimeZone"))
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
    val pipelinePath = s"${config.modelPath}weather.pipeline"
    val categoricalVariables = Array("SkyConditionLow", "SkyConditionMedium", "SkyConditionHigh", "WeatherType", "WindDirection")
    if (!testMode) {
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
    }
    else {
      Utility.log("loading weather pipeline from a file...")
      pipelineModel = PipelineModel.load(pipelinePath)
    }
    Utility.log("transforming features (one-hot encoding + vector assembler)...")
    data = pipelineModel.transform(data)

    data = data.select("AIRPORTID", "WEATHER_COND", "WEATHER_TIME")
    Utility.show(data)
    data
  }

  def combineData(flightData: DataFrame, weatherData: DataFrame): DataFrame = {
    val tf: Int = 3600 * config.weatherTimeFrame
    Utility.log("partitions weatherData: " + weatherData.rdd.getNumPartitions)
    Utility.log("partitions flightData: " + flightData.rdd.getNumPartitions)
    Utility.log("joining flights data with weather data")

    Utility.log("repartitioning dep...")
    val balancedWeatherData = weatherData.repartition($"AIRPORTID")
    Utility.log("joining dep...")

    val depData = flightData.select("FL_ID", "FL_DEP_TIME", "ORIGIN_AIRPORT_ID", "FL_ONTIME")
      .join(balancedWeatherData.as("dep"), $"ORIGIN_AIRPORT_ID" === $"dep.AIRPORTID", "inner")
      .where(s"dep.WEATHER_TIME >= FL_DEP_TIME - $tf and dep.WEATHER_TIME <= FL_DEP_TIME ")
      .drop("dep.AIRPORTID", "ORIGIN_AIRPORT_ID")
    Utility.log("partitions dep:" + depData.rdd.getNumPartitions)

    Utility.log("repartitioning arr...")
    val arrData = flightData.select("FL_ID", "FL_ARR_TIME", "DEST_AIRPORT_ID", "FL_ONTIME")
      .join(balancedWeatherData.as("arr"), $"DEST_AIRPORT_ID" === $"arr.AIRPORTID", "inner")
      .where(s"arr.WEATHER_TIME >= FL_ARR_TIME - $tf and arr.WEATHER_TIME <= FL_ARR_TIME ")
      .drop("arr.AIRPORTID", "ORIGIN_AIRPORT_ID", "FL_ONTIME")
    Utility.log("partitions arr:" + arrData.rdd.getNumPartitions)

    var data = depData.join(arrData, Array("FL_ID"))
    Utility.log("partitions :" + data.rdd.getNumPartitions)

    Utility.log("grouping weather records for each flight...")
    data = data.groupBy($"FL_ID")
      .agg(min($"FL_DEP_TIME").as("FL_DEP_TIME"),
        min($"FL_ARR_TIME").as("FL_ARR_TIME"),
        min($"FL_ONTIME").as("FL_ONTIME"),
        collect_list($"dep.WEATHER_TIME").as("ORIGIN_WEATHER_TIME"),
        collect_list($"dep.WEATHER_COND").as("ORIGIN_WEATHER_COND"),
        collect_list($"arr.WEATHER_TIME").as("DEST_WEATHER_TIME"),
        collect_list($"arr.WEATHER_COND").as("DEST_WEATHER_COND"))

    //Utility.log(data.schema.treeString)
    //Utility.show(data, truncate = true)

    Utility.log("partitions :" + data.rdd.getNumPartitions)
    if (config.mlBalanceDataset) {
      data = balanceDataset(data)
    }

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
    //Utility.log(s"data after filling: ${Utility.count(data)}")
    //Utility.show(data, truncate = true)

    data
  }

  def balanceDataset(data: DataFrame): DataFrame = {
    Utility.log(s"balancing the dataset...")

    Utility.log(s"repartitioning the dataset...")
    var balancedData = data.select("Fl_ID", "FL_ONTIME").repartition($"FL_ONTIME")
    Utility.log(s"counting delayed and on-time flights...")
    // number of delayed flights
    val nbDelayed = balancedData.filter(s"FL_ONTIME = 0").count()
    // number of on-time flights
    val nbOnTime = balancedData.filter(s"FL_ONTIME = 1").count()
    //balancing map (we take the number of delayed flights as on-time ones)
    val fractions = Map(0.0 -> 1.0, 1.0 -> nbDelayed.toDouble / nbOnTime)
    balancedData = balancedData.stat.sampleBy("FL_ONTIME", fractions, 42L)
    Utility.log("joining...")
    balancedData = data.join(balancedData.as("b"), Array("FL_ID", "FL_ONTIME"), "inner")
      .drop("b.FL_ID", "b.FL_ONTIME")
    balancedData
  }
}
