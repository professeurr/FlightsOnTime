import UdfUtility._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{when, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SaveMode}

class DataFeaturing(config: Configuration) {

  import Utility.sparkSession.implicits._

  def extractStationsFeatures(): DataFrame = {
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

    val outputPath = config.persistPath + config.mlMode + "/stations"
    Utility.log(s"saving stations dataset into $outputPath")
    data.repartition(config.partitions)
      .write.mode(SaveMode.Overwrite)
      .parquet(outputPath)

    Utility.sparkSession.read.parquet(outputPath)
  }

  def extractFlightsFeatures(mappingData: DataFrame): DataFeaturing = {
    val s = config.flightsPath.mkString(",")
    Utility.log(s"Loading flights records from $s")
    var data = Utility.sparkSession.read
      .option("header", "true")
      .option("inferSchema", "false")
      .csv(config.flightsPath: _*)
      .drop("OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM")
      .withColumn("FL_ID", monotonically_increasing_id())

    // remove empty columns
    val cols = data.columns.toSet.toList.filter(c => !c.trim.startsWith("_c"))
    data = data.select(cols.map(c => col(c)): _*)
    // convert delay related columns to numerical type
    val numericalColumns = Array("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY", "CANCELLED", "DIVERTED")
    numericalColumns.foreach(c => data = data.withColumn(c, col(c).cast(DoubleType)))
    data = data.na.fill(0.0, numericalColumns)

    // remove cancelled and diverted data
    Utility.log("Removing cancelled and diverted flights (they are out of this analysis)...")
    data = data.filter("CANCELLED = 0 and DIVERTED = 0").drop("CANCELLED", "DIVERTED")

    Utility.log("mapping flights with the weather stations...")
    data = data.join(mappingData, $"ORIGIN_AIRPORT_ID" === $"AirportId", "inner")
      .drop("AirportId")

    Utility.log("computing FL_DEP_TIME and FL_ARR_TIME")
    data = data.withColumn("FL_DEP_TIME", unix_timestamp(concat_ws("", $"FL_DATE", $"CRS_DEP_TIME"), "yyyy-MM-ddHHmm").minus($"TimeZone"))
      .withColumn("FL_ARR_TIME", ($"FL_DEP_TIME" + $"CRS_ELAPSED_TIME" * 60).cast(LongType))
      .withColumn("DELAY", ($"WEATHER_DELAY" + $"NAS_DELAY").cast(LongType))
      .withColumn("year", substring($"FL_DATE", 0, 4))
      .withColumn("month", substring($"FL_DATE", 6, 2))
      .drop("CRS_ELAPSED_TIME", "CRS_DEP_TIME", "FL_DATE", "STATION_WBAN", "TimeZone", "WEATHER_DELAY", "NAS_DELAY")

    val outputPath = config.persistPath + config.mlMode + "/flights"
    Utility.log(s"saving flights dataset into $outputPath")
    data.repartition(config.partitions, col("year"), col("month"))
      .write.mode(SaveMode.Overwrite)
      .partitionBy("year", "month")
      .parquet(outputPath)

    this
  }

  def extractWeatherFeatures(mappingData: DataFrame): DataFeaturing = {

    Utility.log(s"Loading weather records from ${config.weatherPath.mkString(",")}")

    var data = Utility.sparkSession.read
      .option("header", "true")
      .option("inferSchema", "false")
      .csv(config.weatherPath: _*)
      .select("WBAN", "Date", "Time", "SkyCondition", "DryBulbCelsius", "WeatherType", "StationPressure", "WindDirection", "Visibility", "RelativeHumidity", "WindSpeed")

    Utility.log(s"weather initial number of records: ${Utility.count(data)}")

    Utility.log("getting timezones of each station and converting weather record time to utc...")
    data = data.join(mappingData, $"STATION_WBAN" === $"WBAN", "inner")
      //.withColumn("AirportId", $"AIRPORTID".cast(DoubleType))
      .withColumn("WEATHER_TIME", unix_timestamp(concat_ws("", $"Date", $"Time"), "yyyyMMddHHmm").minus($"TimeZone"))
      .withColumn("year", substring($"Date", 0, 4))
      .withColumn("month", substring($"Date", 5, 2))
      .drop("STATION_WBAN", "WBAN", "TimeZone")
    Utility.log(s"weather number of records after the mapping with the stations: ${Utility.count(data)}")

    Utility.log("cleaning numerical variables...")
    val continuousVariables = Array("StationPressure", "RelativeHumidity", "WindSpeed", "Visibility", "DryBulbCelsius")
    continuousVariables.foreach(c => data = data.withColumn(c, col(c).cast(DoubleType)))
    data = data.withColumn("Visibility", when($"Visibility" > 10.0, 10.0).otherwise($"Visibility"))

    Utility.log("cleaning SkyCondition...")
    data = data.withColumn("SkyCondition", parseSkyConditionUdf($"SkyCondition"))
    val skyConditionColumns = Array("SkyConditionLow", "SkyConditionMedium", "SkyConditionHigh")
    skyConditionColumns.zipWithIndex.foreach { case (c, i) => data = data.withColumn(c, col("SkyCondition")(i)) }

    Utility.log("cleaning WeatherType...")
    data = data.withColumn("WeatherTypeIndex", when(length($"WeatherType") < 2, 0.0)
      .otherwise(parseWeatherTypeUdf($"WeatherType")))
      .drop("WeatherType", "SkyCondition")

    Utility.log("cleaning WindDirection...")
    data = data.withColumn("WindDirection", when($"WindDirection" === "VR", -1.0)
      .otherwise($"WindDirection".cast(DoubleType)))

    Utility.log("applying forward/backward-fill on weather conditions data...")
    val columns = Array("WeatherTypeIndex", "WindDirection", "StationPressure", "RelativeHumidity", "WindSpeed", "Visibility", "DryBulbCelsius")
    val w0 = Window.partitionBy($"AIRPORTID", $"Date").orderBy($"Time".asc).rowsBetween(Window.unboundedPreceding, 0)
    //    val w1 = Window.partitionBy($"AIRPORTID", $"Date").orderBy($"Time".asc).rowsBetween(0, Window.unboundedFollowing)
    //    columns.foreach(c => data = data.withColumn(c, last(last(c, ignoreNulls = true).over(w0)).over(w1)))
    columns.foreach(c => data = data.withColumn(c, last(c, ignoreNulls = true).over(w0)))

    Utility.log("removing na...")
    data = data.drop("Date", "Time")
      .na.drop().cache()

    var pipelineModel: PipelineModel = null
    val pipelinePath = s"${config.persistPath}/extract/pipeline"

    if (config.mlMode.equalsIgnoreCase("extract")) {
      // string indexer sky conditions features
      var stages: Array[PipelineStage] = skyConditionColumns.map(c =>
        new StringIndexer().setInputCol(c).setOutputCol(c + "Index").setHandleInvalid("skip")
      )
      //scaling continuous variables
      stages ++= continuousVariables.flatMap(c => Array[PipelineStage](
        new VectorAssembler().setInputCols(Array(c)).setOutputCol(c + "Index"),
        new MinMaxScaler().setInputCol(c + "Index").setOutputCol(c + "Vect").setMin(0.0).setMax(1.0)
      ))

      // wind directions handling
      val splits = Array(Double.NegativeInfinity, 0, 0.01, 45, 90, 135, 180, 225, 270, 315, 360, Double.PositiveInfinity)
      stages :+= new Bucketizer().setInputCol("WindDirection").setOutputCol("WindDirectionIndex").setSplits(splits)

      // apply one-hot encoding of columns of the features
      val otherColumns = Array("WeatherType", "WindDirection")
      stages :+= new OneHotEncoder()
        .setInputCols((skyConditionColumns ++ otherColumns).map(c => c + "Index"))
        .setOutputCols((skyConditionColumns ++ otherColumns).map(c => c + "Vect")).setHandleInvalid("keep")

      // creating vector assembler transformer to group the weather conditions features to one column
      val output = if (config.features > 0) "WEATHER_COND_" else "WEATHER_COND"
      stages :+= new VectorAssembler()
        .setInputCols((skyConditionColumns ++ otherColumns ++ continuousVariables).map(c => c + "Vect"))
        .setOutputCol(output)

      // dimensionality reduction (if requested)
      if (config.features > 0)
        stages :+= new PCA().setInputCol(output).setOutputCol("WEATHER_COND").setK(config.features)

      // the transformation pipeline that transforms categorical variables and assembles all variable into feature column
      val pipeline = new Pipeline().setStages(stages)

      Utility.log("fitting features (one-hot encoding + vector assembler)...")
      pipelineModel = pipeline.fit(data)

      Utility.log(s"saving weather trained pipeline to a file '$pipelinePath'...")
      pipelineModel.write.overwrite.save(pipelinePath)
    }
    else {
      pipelineModel = PipelineModel.load(pipelinePath)
    }

    Utility.log("transforming features (one-hot encoding + vector assembler)...")
    val data2 = pipelineModel.transform(data)
      .select("year", "month", "AIRPORTID", "WEATHER_TIME", "WEATHER_COND")

    val outputPath = config.persistPath + config.mlMode + "/weather"
    Utility.log(s"saving weather dataset into $outputPath")
    data2.repartition(config.partitions, $"year", $"month", $"AIRPORTID")
      .write.mode(SaveMode.Overwrite)
      .partitionBy("year", "month", "AIRPORTID")
      .parquet(outputPath)
    data.unpersist(blocking = false)
    this
  }
}
