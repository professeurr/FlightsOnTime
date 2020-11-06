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
      .na.fill(0.0, delayColumns)
    delayColumns.foreach(c => data = data.withColumn(c, col(c).cast(DoubleType)))
    Utility.show(data)

    // remove empty columns
    val cols = data.columns.toSet.toList.filter(c => !c.trim.startsWith("_c"))
    data = data.select(cols.map(c => col(c)): _*)
    Utility.log(s"number of flights: ${Utility.count(data)}; Flights columns: $cols")

    Utility.log("computing flights identifier (FL_ID)...")
    data = data.withColumn("FL_ID",
      concat_ws("_", $"OP_CARRIER_AIRLINE_ID", $"OP_CARRIER_FL_NUM", $"FL_DATE", $"ORIGIN_AIRPORT_ID", $"DEST_AIRPORT_ID"))
      .drop("OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM")
    //Utility.log(Data.schema.treeString)

    // remove cancelled and diverted data
    Utility.log("Removing cancelled and diverted flights (they are out of this analysis)...")
    data = data.filter("CANCELLED + DIVERTED = 0").drop("CANCELLED", "DIVERTED")
    Utility.log(s"flights dataset without cancelled and diverted flights: ${Utility.count(data)}")

    Utility.log("Removing non-weather related delayed records...")
    data = data.filter(s"ARR_DELAY_NEW <= ${config.flightsDelayThreshold} or  WEATHER_DELAY + NAS_DELAY >= ${config.flightsDelayThreshold} ")
    Utility.log(s"flights dataset without non-weather related delay records: ${Utility.count(data)}")

    Utility.log("computing FL_ONTIME flag (1=on-time; 0=delayed)")
    data = data.withColumn("FL_ONTIME", ($"ARR_DELAY_NEW" <= config.flightsDelayThreshold).cast(DoubleType))
      .drop("WEATHER_DELAY", "NAS_DELAY")

    Utility.log("mapping flights with the weather stations...")
    data = data.join(mappingData, $"ORIGIN_AIRPORT_ID" === $"AirportId", "inner")
      .drop("AirportId")
    Utility.log(s"flights data after the mapping: ${Utility.count(data)}")

    Utility.log("computing FL_DATE, FL_DEP_TIME and FL_ARR_TIME")
    data = data.withColumn("FL_DEP_TIME", unix_timestamp(concat_ws("", $"FL_DATE", $"CRS_DEP_TIME"), "yyyy-MM-ddHHmm").minus($"TimeZone"))
      .withColumn("FL_ARR_TIME", ($"FL_DEP_TIME" + $"CRS_ELAPSED_TIME" * 60).cast(LongType))
      .select("FL_ID", "FL_DEP_TIME", "FL_ARR_TIME", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_ONTIME")

    if (config.mlBalanceDataset)
      data = balanceDataset(data, "FL_ONTIME")
    if (config.flightsFrac > 0)
      data = data.sample(withReplacement = false, config.flightsFrac)

    Utility.log(s"number of flights used for the analysis: ${Utility.count(data)}")
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
    Utility.log(s"weather initial number of records: ${Utility.count(data)};")
    Utility.show(data)

    Utility.log("getting timezones of each station and converting weather record time to utc...")
    data = data.join(mappingData, $"STATION_WBAN" === $"WBAN", "inner")
      .withColumnRenamed("AirportId", "AIRPORTID")
      .withColumn("WEATHER_TIME", unix_timestamp(concat_ws("", $"Date", $"Time"), "yyyyMMddHHmm").minus($"TimeZone"))
      .drop("STATION_WBAN", "WBAN", "Time", "Date", "TimeZone")
    Utility.log(s"weather number of records after the mapping with the stations: ${Utility.count(data)}")
    Utility.show(data)

    val featureLabel = "WF" // temporary weather features columns names.

    Utility.log("transforming categorical variables...")
    data = data.withColumn(featureLabel + "0", parseSkyConditionUdf($"SkyCondition", lit(0)))
      .withColumn(featureLabel + "1", parseSkyConditionUdf($"SkyCondition", lit(1)))
      .withColumn(featureLabel + "2", parseSkyConditionUdf($"SkyCondition", lit(2)))
      .withColumn(featureLabel + "3", array_min(array(lit($"Visibility".cast(IntegerType) / 4), lit(3))) + 1)
      .withColumn(featureLabel + "4", array_min(array(lit($"RelativeHumidity".cast(IntegerType) / 25), lit(4))) + 1)
      .withColumn(featureLabel + "5", array_min(array(lit($"WindSpeed".cast(IntegerType) / 25), lit(4))) + 1)
      .withColumn(featureLabel + "6", array_min(array(lit(($"StationPressure".cast(IntegerType) - 10) / 10), lit(4))) + 1)
      .withColumn(featureLabel + "7", array_min(array(lit(($"DryBulbCelsius".cast(IntegerType) + 50) / 25), lit(4))) + 1)
      .withColumn(featureLabel + "8", size(split($"WeatherType", " ")))
      .withColumn(featureLabel + "9", when(col("WindDirection") === "VR", 11).when(col("WindDirection") === "0", 10)
        .otherwise($"WindDirection".cast(IntegerType) / 45) + 1)
      .drop(weatherCondColumns: _*)
      .na.fill(0)
    val categoricalColumnsRange = 0 until data.columns.count(c => c.contains(featureLabel))
    categoricalColumnsRange.foreach(i => data = data.withColumn(featureLabel + i, col(featureLabel + i).cast(IntegerType)))
    //    data = data.withColumn(featuresLabel, parseWeatherVariablesUdf(weatherCondColumns.map(c => col(c)): _*))
    //    categoricalColumnsRange.foreach(i => data = data.withColumn(featureLabel + i, col(featuresLabel)(i)))
    //    data = data.drop(featuresLabel).drop(weatherCondColumns: _*).cache()
    Utility.show(data)
    //data.explain(false)

    var pipelineModel: PipelineModel = null
    Utility.log("assembling weather conditions")
    val pipelinePath = s"${config.modelPath}weather.pipeline"
    if (!testMode) {
      // apply one-hot encoding of columns of the features
      val oneHotEncoder = new OneHotEncoder()
        .setInputCols(categoricalColumnsRange.map(i => featureLabel + i).toArray)
        .setOutputCols(categoricalColumnsRange.map(i => featureLabel + "Vect" + i).toArray)
        .setDropLast(true)

      // creating vector assembler transformer to group the weather conditions features to one column
      val vectorAssembler = new VectorAssembler()
        .setInputCols(categoricalColumnsRange.map(i => featureLabel + "Vect" + i).toArray)
        .setOutputCol("WEATHER_COND")

      // the transformation pipeline that assemble and index the features
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
      .drop(categoricalColumnsRange.map(i => featureLabel + "Vect" + i): _*)
      .drop(categoricalColumnsRange.map(i => featureLabel + i): _*)
    Utility.show(data)

    data
  }

  def combineData(flightData: DataFrame, weatherData: DataFrame): DataFrame = {
    val tf: Int = 3600 * config.weatherTimeFrame
    Utility.log("partitions weatherData: " + weatherData.rdd.getNumPartitions)
    Utility.log("partitions flightData: " + flightData.rdd.getNumPartitions)
    Utility.log("joining flights data with weather data")
    var data = flightData.join(weatherData.as("dep"), $"ORIGIN_AIRPORT_ID" === $"dep.AIRPORTID", "inner")
    Utility.log("partitions dep:" + data.rdd.getNumPartitions)
    data = data.join(weatherData.as("arr"), $"DEST_AIRPORT_ID" === $"arr.AIRPORTID", "inner")
    Utility.log("partitions arr:" + data.rdd.getNumPartitions)

    data = data.where(s"dep.WEATHER_TIME >= FL_DEP_TIME - $tf and dep.WEATHER_TIME <= FL_DEP_TIME " +
      s"and arr.WEATHER_TIME >= FL_ARR_TIME - $tf and arr.WEATHER_TIME <= FL_ARR_TIME")
      .drop("dep.AIRPORTID", "arr.AIRPORTID", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID")

    Utility.log("partitions :" + data.rdd.getNumPartitions)

    Utility.log("grouping weather records for each flight")
    data = data.groupBy($"FL_ID", $"FL_DEP_TIME", $"FL_ARR_TIME", $"FL_ONTIME")
      .agg(collect_list($"dep.WEATHER_TIME").as("ORIGIN_WEATHER_TIME"),
        collect_list($"dep.WEATHER_COND").as("ORIGIN_WEATHER_COND"),
        collect_list($"arr.WEATHER_TIME").as("DEST_WEATHER_TIME"),
        collect_list($"arr.WEATHER_COND").as("DEST_WEATHER_COND"))


    Utility.log("partitions :" + data.rdd.getNumPartitions)

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
    Utility.log(s"data after filling: ${Utility.count(data)}")

    if (config.mlBalanceDataset) {
      data = balanceDataset(data, "FL_ONTIME")
    }
    data = data.drop("FL_ID")
    Utility.show(data, true)

    data
  }

  def balanceDataset(data: DataFrame, label: String): DataFrame = {
    Utility.log(s"balancing the dataset...")
    Utility.log(s"number of records: ${Utility.count(data)}")
    val Array(count0, count1) = Array(0, 1).map(f => data.filter(s"FL_ONTIME = $f").count())
    val Array(index0, index1, r) = if (count1 > count0) Array(0, 1, 1.0 * count0 / count1) else Array(1, 0, 1.0 * count1 / count0)
    Utility.log(s"number of on-time flights=$count1, number of delayed flights=$count0")
    val balancedData = data.filter(s"FL_ONTIME = $index1").sample(withReplacement = false, r)
      .union(data.filter(s"FL_ONTIME = $index0"))
    //balancedData = balancedData.withColumn("WEIGHT", when(col(label) === 0, r).otherwise(1 - r))
    balancedData
  }

}
