import UdfUtility._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{when, _}
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

  def preloadFlights(mappingData: DataFrame): DataFeaturing = {
    val s = config.flightsDataPath.mkString(",")
    Utility.log(s"Loading flights records from $s")
    val numericalColumns = Array("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "ARR_DELAY_NEW", "WEATHER_DELAY", "NAS_DELAY", "CANCELLED", "DIVERTED")
    var data = Utility.sparkSession.read
      .option("header", "true")
      .option("inferSchema", "false")
      .csv(config.flightsDataPath: _*)

    // remove empty columns
    val cols = data.columns.toSet.toList.filter(c => !c.trim.startsWith("_c"))
    data = data.select(cols.map(c => col(c)): _*)
    // convert delay related columns to numerical type
    numericalColumns.foreach(c => data = data.withColumn(c, col(c).cast(DoubleType)))
    data = data.na.fill(0.0, numericalColumns)
    Utility.show(data)

    Utility.log("computing flights identifier (FL_ID)...")
    data = data.withColumn("FL_ID", concat_ws("_", $"ORIGIN_AIRPORT_ID", $"DEST_AIRPORT_ID",
      $"FL_DATE", $"OP_CARRIER_AIRLINE_ID", $"OP_CARRIER_FL_NUM"))
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

    val outputPath = config.persistPath + "data.flights"
    Utility.log(s"saving flights dataset into $outputPath")
    data.repartition(config.partitions, col("year"), col("month"))
      .write.mode(SaveMode.Overwrite)
      .partitionBy("year", "month")
      .parquet(outputPath)

    this
  }

  def preloadWeather(mappingData: DataFrame): DataFeaturing = {

    Utility.log(s"Loading weather records from ${config.weatherDataPath.mkString(",")}")
    val weatherCondColumns = Array("SkyCondition", "DryBulbCelsius", "WeatherType", "StationPressure", "WindDirection", "Visibility", "RelativeHumidity", "WindSpeed")
    var data = Utility.sparkSession.read
      .option("header", "true")
      .option("inferSchema", "false")
      .csv(config.weatherDataPath: _*)
      .select(Array(col("WBAN"), col("Date"), col("Time")) ++ weatherCondColumns.map(c => col(c)): _*)
    Utility.log(s"weather initial number of records: ${Utility.count(data)};")
    Utility.show(data)

    Utility.log("getting timezones of each station and converting weather record time to utc...")
    data = data.join(mappingData, $"STATION_WBAN" === $"WBAN", "inner")
      .withColumn("AirportId", $"AIRPORTID".cast(DoubleType))
      .withColumn("WEATHER_TIME", unix_timestamp(concat_ws("", $"Date", $"Time"), "yyyyMMddHHmm").minus($"TimeZone"))
      .withColumn("year", substring($"Date", 0, 4))
      .withColumn("month", substring($"Date", 5, 2))
      .drop("STATION_WBAN", "WBAN", "TimeZone")
    Utility.log(s"weather number of records after the mapping with the stations: ${Utility.count(data)}")
    Utility.show(data)

    Utility.log("cleaning numerical variables...")
    var continuousVariables = Array("StationPressure", "RelativeHumidity", "WindSpeed", "Visibility", "DryBulbCelsius")
    continuousVariables.foreach(c => {
      data = data.withColumn(c, when(col(c).cast(DoubleType).isNull, null).otherwise(col(c).cast(DoubleType)))
    })

    Utility.log("transforming/cleaning weather variables...")
    Utility.log("cleaning SkyCondition...")
    Array("SkyConditionLowCategory", "SkyConditionMediumCategory", "SkyConditionHighCategory")
      .zipWithIndex.foreach { case (c, i) => data = data.withColumn(c, parseSkyConditionUdf($"SkyCondition", lit(i)))
      .withColumn(c, when(col(c) === -1, null).otherwise(col(c)))
    }

    Utility.log("cleaning WeatherType...")
    data = data.withColumn("WeatherTypeCategory", when(length($"WeatherType") < 2, null)
      .otherwise(parseWeatherTypeUdf($"WeatherType")))

    Utility.log("cleaning WindDirection...")
    data = data.withColumn("WindDirectionCategory",
      when($"WindDirection" === "VR", 9)
        .when($"WindDirection" === "0", 8)
        .when($"WindDirection".cast(IntegerType).isNull, null)
        .otherwise(($"WindDirection".cast(IntegerType) / 45).cast(IntegerType)))
    Utility.show(data)

    Utility.log("applying forward/backward-fill on weather conditions data...")
    val columns = Array("SkyConditionLowCategory", "SkyConditionMediumCategory", "SkyConditionHighCategory",
      "WindDirectionCategory", "DryBulbCelsius", "WeatherTypeCategory",
      "StationPressure", "Visibility", "RelativeHumidity", "WindSpeed")
    val w0 = Window.partitionBy($"AIRPORTID", $"Date").orderBy($"Time".asc)
    val w1 = Window.partitionBy($"AIRPORTID", $"Date").orderBy($"Time".desc)
    columns.foreach(c => data =
      data.withColumn(c, last(col(c), ignoreNulls = true).over(w0))
        .withColumn(c, last(col(c), ignoreNulls = true).over(w1))
    )

    Utility.log("removing na...")
    data = data.na.drop().cache()

    Utility.log("scaling continuous variables...")
    data = data.withColumn("DryBulbCelsius", when($"DryBulbCelsius" <= -8, 0)
      .when($"DryBulbCelsius" <= -3, 1).when($"DryBulbCelsius" <= 0, 2)
      .when($"DryBulbCelsius" <= 35, 3).otherwise(4))
      .withColumn("StationPressure", ((greatest(least($"StationPressure", lit(36)), lit(20)) - 20) / 4.1).cast(IntegerType))
      .withColumn("Visibility", (greatest(least($"Visibility", lit(10)), lit(0)) / 2.6).cast(IntegerType)) // TODO: enhance the intervals
      .withColumn("RelativeHumidity", ($"RelativeHumidity" / 26).cast(IntegerType))
      .withColumn("WindSeep", when($"WindSpeed" <= 9, 0).when($"WindSpeed" <= 15, 1).when($"WindSpeed" <= 24, 2).otherwise(3))
    Utility.show(data)
    //Utility.exit()

    Utility.log("assembling weather conditions")
    val pipelinePath = s"${config.persistPath}pipeline.weather"
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
    val pipelineModel = pipeline.fit(data)
    Utility.log(s"saving weather trained pipeline to a file '$pipelinePath'...")
    pipelineModel.write.overwrite.save(pipelinePath)

    Utility.log("transforming features (one-hot encoding + vector assembler)...")
    data = pipelineModel.transform(data)

    data = data.select("year", "month", "AIRPORTID", "WEATHER_TIME", "WEATHER_COND").cache()
    Utility.show(data)

    val outputPath = config.persistPath + "data.weather"
    Utility.log(s"saving weather dataset into $outputPath")
    data.repartition(config.partitions, $"year", $"month")
      .write.mode(SaveMode.Overwrite)
      .partitionBy("year", "month")
      .parquet(outputPath)

    this
  }
}
