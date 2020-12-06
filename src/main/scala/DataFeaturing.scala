import UdfUtility._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
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
    Utility.show(data)

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

    //    Utility.log(s"flights: ${data.count()}")
    //    Utility.log(s"on-time: ${data.filter("ARR_DELAY_NEW < 15").count()}")
    //    Utility.log(s"delayed: ${data.filter("ARR_DELAY_NEW >= 15").count()}")
    //    Utility.log(s"cancelled: ${data.filter("CANCELLED = 1").count()}")
    //    Utility.log(s"diverted: ${data.filter("DIVERTED = 1").count()}")
    //    Utility.log(s"delayed by nas+weather: ${data.filter("NAS_DELAY + WEATHER_DELAY >= 15").count()}")
    //
    //    Utility.exit()

    // remove empty columns
    val cols = data.columns.toSet.toList.filter(c => !c.trim.startsWith("_c"))
    data = data.select(cols.map(c => col(c)): _*)
    // convert delay related columns to numerical type
    val numericalColumns = Array("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY", "CANCELLED", "DIVERTED")
    numericalColumns.foreach(c => data = data.withColumn(c, col(c).cast(DoubleType)))
    data = data.na.fill(0.0, numericalColumns)
    Utility.show(data)

    Utility.log("computing flights identifier (FL_ID)...")
    data = data.withColumn("FL_ID", monotonically_increasing_id())
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

    // remove unnecessary columns
    data = data.drop("CRS_ELAPSED_TIME", "CRS_DEP_TIME", "FL_DATE", "STATION_WBAN", "TimeZone")

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
    Utility.show(data)

    Utility.log("getting timezones of each station and converting weather record time to utc...")
    data = data.join(mappingData, $"STATION_WBAN" === $"WBAN", "inner")
      .withColumn("AirportId", $"AIRPORTID".cast(DoubleType))
      .withColumn("WEATHER_TIME", unix_timestamp(concat_ws("", $"Date", $"Time"), "yyyyMMddHHmm").minus($"TimeZone"))
      .withColumn("year", substring($"Date", 0, 4))
      .withColumn("month", substring($"Date", 5, 2))
      .drop("STATION_WBAN", "WBAN", "TimeZone", "Date", "Time")
    Utility.log(s"weather number of records after the mapping with the stations: ${Utility.count(data)}")
    Utility.show(data.filter("StationPressure <> 'M'"))

    data = data.withColumn("W_ID", monotonically_increasing_id())

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
    Utility.show(data)

    //    Utility.log("applying forward/backward-fill on weather conditions data...")
    //    val columns = Array("WeatherTypeIndex", "WindDirection") ++ continuousVariables
    //    val w0 = Window.partitionBy($"AIRPORTID").orderBy($"WEATHER_TIME".asc).rowsBetween(Window.unboundedPreceding, 0)
    //    val w1 = Window.partitionBy($"AIRPORTID").orderBy($"WEATHER_TIME".asc).rowsBetween(0, Window.unboundedFollowing)
    //    columns.foreach(c => data = data.withColumn(c, last(last(c, ignoreNulls = true).over(w0)).over(w1)))

    Utility.log("removing na...")
    data = data.na.drop().cache()

    //    val tmpOutputPath = config.persistPath + config.mlMode + "/weather.tmp"
    //    Utility.log(s"saving weather dataset into $tmpOutputPath")
    //    data.repartition(config.partitions, $"year", $"month")
    //      .write.mode(SaveMode.Overwrite)
    //      .partitionBy("year", "month")
    //      .parquet(tmpOutputPath)
    //    data = Utility.sparkSession.read.parquet(tmpOutputPath)
    //      .withColumn("month", lpad($"month", 2, "0"))

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
    data = pipelineModel.transform(data)

    //    val corrMatrix = Correlation.corr(data.select("WEATHER_COND"), "WEATHER_COND", "pearson")
    //    data.select("WEATHER_COND").show(truncate = false)
    //    val Row(coeff1: Matrix) = corrMatrix.head
    //    println("Pearson correlation matrix:\n" + coeff1.toString(35, Int.MaxValue))
    //    Utility.exit()

    data = data.select("year", "month", "W_ID", "AIRPORTID", "WEATHER_TIME", "WEATHER_COND")

    Utility.show(data)

    val outputPath = config.persistPath + config.mlMode + "/weather"
    Utility.log(s"saving weather dataset into $outputPath")
    data.repartition(config.partitions, $"year", $"month", $"AIRPORTID")
      .write.mode(SaveMode.Overwrite)
      .partitionBy("year", "month", "AIRPORTID")
      .parquet(outputPath)

    this
  }
}
