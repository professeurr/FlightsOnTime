import UdfUtility._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.functions.{when, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SaveMode}

class DataFeaturing(config: Configuration) {

  import Utility.sparkSession.implicits._

  def extractFlightsFeatures(flightData: DataFrame, mappingData: DataFrame): DataFeaturing = {
    var data = flightData.drop("OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM")
      .withColumn("FL_ID", monotonically_increasing_id())

    // remove empty columns
    val cols = data.columns.toSet.toList.filter(c => !c.trim.startsWith("_c"))
    data = data.select(cols.map(c => col(c)): _*)
    // convert delay related columns to numerical type
    val numericalColumns = Array("ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY", "CANCELLED", "DIVERTED")
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
      .withColumn("delay", ($"WEATHER_DELAY" + $"NAS_DELAY").cast(LongType))

    var outputPath = config.persistPath + config.mlMode + "/flights.dep"
    Utility.log(s"saving flights departure dataset into $outputPath")
    data.select($"FL_ID", $"ORIGIN_AIRPORT_ID".as("AIRPORTID"), $"FL_DEP_TIME", $"delay")
      .repartition(config.partitions, col("AIRPORTID"))
      .write.mode(SaveMode.Overwrite)
      .partitionBy("AIRPORTID")
      .parquet(outputPath)

    outputPath = config.persistPath + config.mlMode + "/flights.arr"
    Utility.log(s"saving flights arrival dataset into $outputPath")
    data.select($"FL_ID", $"DEST_AIRPORT_ID".as("AIRPORTID"), $"FL_ARR_TIME")
      .repartition(config.partitions, col("AIRPORTID"))
      .write.mode(SaveMode.Overwrite)
      .partitionBy("AIRPORTID")
      .parquet(outputPath)

    this
  }

  def extractWeatherFeatures(weatherData: DataFrame, mappingData: DataFrame): DataFeaturing = {
    var data = weatherData.select("WBAN", "Date", "Time", "SkyCondition", "DryBulbCelsius",
      "WeatherType", "StationPressure", "WindDirection", "Visibility", "RelativeHumidity", "WindSpeed")

    Utility.log("getting timezones of each station and converting weather record time to utc...")
    data = data.join(mappingData, Array("WBAN"), "inner")
      .withColumn("WEATHER_TIME", unix_timestamp(concat_ws("", $"Date", $"Time"), "yyyyMMddHHmm").minus($"TimeZone"))
      .drop("WBAN", "TimeZone", "Time", "Date")

    Utility.log(s"weather number of records: ${Utility.count(data)}")

    Utility.log("cleaning numerical variables...")
    val continuousVariables = Array("StationPressure", "RelativeHumidity", "WindSpeed", "Visibility", "DryBulbCelsius")
    continuousVariables.foreach(c => data = data.withColumn(c, col(c).cast(DoubleType)))

    Utility.log("cleaning categorical variables...")
    val categoricalColumns = Array("WeatherType", "WindDirection", "SkyCondition")
    categoricalColumns.foreach(c => data = data.withColumn(c, when(col(c) === "M", null).otherwise(col(c))))
    data = data.na.drop()
      .withColumn("SkyCondition", parseSkyConditionUdf($"SkyCondition"))
      .withColumn("WeatherTypeIndex", when(length($"WeatherType") < 2, 0.0).otherwise(1.0))
      .withColumn("WindDirection", when($"WindDirection" === "VR", -1.0).otherwise($"WindDirection".cast(DoubleType)))
      .drop("WeatherType")

    Utility.log("removing na...")
    data = data.na.drop().cache()

    var pipelineModel: PipelineModel = null
    val pipelinePath = s"${config.persistPath}/extract/pipeline"

    if (config.mlMode.equalsIgnoreCase("extract")) {
      // string indexer sky conditions features
      var stages: Array[PipelineStage] = Array()

      // encoding SkyCondition with StringIndexer
      stages +:= new StringIndexer()
        .setInputCol("SkyCondition")
        .setOutputCol("SkyConditionIndex")
        .setHandleInvalid("skip")

      // wind directions handling
      val splits = Array(Double.NegativeInfinity, 0, 0.01, 45, 90, 135, 180, 225, 270, 315, 360, Double.PositiveInfinity)
      stages :+= new Bucketizer()
        .setInputCol("WindDirection")
        .setOutputCol("WindDirectionIndex")
        .setSplits(splits)

      // apply one-hot encoding of categorical variables
      stages :+= new OneHotEncoder()
        .setInputCols(categoricalColumns.map(c => c + "Index"))
        .setOutputCols(categoricalColumns.map(c => c + "Vect"))
        .setHandleInvalid("keep")

      // creating vector assembler transformer to group the weather conditions features to one column
      val output = if (config.features > 0) "WEATHER_COND_" else "WEATHER_COND"
      stages :+= new VectorAssembler()
        .setInputCols(categoricalColumns.map(c => c + "Vect") ++ continuousVariables)
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

    Utility.log("transforming features with the pipeline model...")
    data = pipelineModel.transform(data)
      .select("AIRPORTID", "WEATHER_TIME", "WEATHER_COND")

    val outputPath = config.persistPath + config.mlMode + "/weather"
    Utility.log(s"saving weather dataset into $outputPath")
    data.repartition(config.partitions, $"AIRPORTID")
      .write.mode(SaveMode.Overwrite)
      .partitionBy("AIRPORTID")
      .parquet(outputPath)

    data.unpersist(blocking = false)

    this
  }
}
