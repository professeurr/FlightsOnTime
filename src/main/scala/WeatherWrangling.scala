import org.apache.log4j.Logger
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Bucketizer, StringIndexer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataType, DoubleType, StringType}

class WeatherWrangling(val path: String, val mappingData: DataFrame, config: Configuration) {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  import Utils.sparkSession.implicits._

  var Data: DataFrame = _
  var WeatherCondColumns: Array[String] = Array("RelativeHumidity", "DryBulbCelsius", "WindSpeed", "StationPressure", "Visibility", "WindDirection", "SkyCondition", "WeatherType")
  val WeatherFeatures2: Seq[(String, DataType, String)] = Array(
    ("RelativeHumidity", DoubleType, null),
    ("DryBulbCelsius", DoubleType, null),
    ("WindSpeed", DoubleType, null),
    ("StationPressure", DoubleType, null),
    ("Visibility", DoubleType, null),
    ("WindDirection", DoubleType, null),
    ("SkyCondition", StringType, null),
    ("WeatherType", StringType, null))

  def loadData(): DataFrame = {

    logger.info("Loading weather data")
    Data = Utils.sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load(path)
      .select(Array(col("WBAN"), col("Date"), col("Time")) ++ WeatherCondColumns.map(c => col(c)): _*)
    //logger.info(Data.schema.treeString)
    logger.info(s"weather initial data: ${Data.count()}")
    logger.info(s"total wbans: ${Data.select("WBAN").distinct().count()}")

    logger.info("getting timezones of each station and converting weather time to utc...")
    Data = Data.join(mappingData, $"MAPPING_WBAN" === $"WBAN", "inner")
      .withColumnRenamed("AirportId", "AIRPORTID")
      .withColumn("WEATHER_TIME", unix_timestamp(concat_ws("", $"Date", $"Time"), "yyyyMMddHHmm").minus($"TimeZone"))
      .drop("MAPPING_WBAN", "Time", "Date", "TimeZone")
    logger.info(s"weather data after the mapping: ${Data.count()}")
    Data.show(truncate = false)

    logger.info("applying forward-fill on weather conditions data")
    val w0 = Window.partitionBy($"WBAN").orderBy($"WEATHER_TIME".asc)
    WeatherCondColumns.foreach(c => {
      val colValue = trim(col(c))
      Data = Data.withColumn(c, when(colValue === "" || colValue === "M", null).otherwise(colValue))
        .withColumn(c, last(colValue, ignoreNulls = true).over(w0))
    })
    Data = Data.withColumn("SkyCondition", when($"SkyCondition" === "VR", -1).otherwise($"SkyCondition"))
      .drop("WBAN")
    Data.show(false)

    logger.info(s"weather data before na.drop(): ${Data.count()}")
    Data = Data.na.drop() // remove null data
    logger.info(s"weather data after na.drop(): ${Data.count()}")
    Data = Data.cache() // put the data into the cache before the transformation steps
    Data.show(false)

    if (config.weatherBucketizeWindDirection) {
      logger.info("transforming WindDirection to categorical")
      val splits = Array(-1, 0, 0.1, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360)
      val bucketizer = new Bucketizer().setInputCol("WindDirection").setOutputCol("WindDirectionCategory").setSplits(splits)
      Data = Data.withColumn("WindDirection", $"WindDirection".cast(DoubleType))
      Data = bucketizer.transform(Data)
        .withColumn("WindDirectionCategory", when(col("WindDirectionCategory") === 10, 2)
          .otherwise(col("WindDirectionCategory")))
        .drop("WindDirection")
      Data.show(truncate = false)
    } else {
      Data = Data.withColumnRenamed("WindDirection", "WindDirectionCategory")
    }

    logger.info(s"splitting SkyCondition into ${config.weatherSkyconditionLayers} column(s)")
    val scRange = 0 until config.weatherSkyconditionLayers
    Data = Data.withColumn("SkyCondition", UtilUdfs.padSkyConditionValueUdf($"SkyCondition", lit(scRange.length), lit("ZZZ"))) // pad Z
      .filter("SkyCondition is not null")
    Data = Data.select(Data.columns.map(c => col(c)) ++ scRange.map(i => col("SkyCondition")(i).as(s"SkyConditionCategory_$i")): _*)
    val sc = Data.select("SkyConditionCategory_0").distinct()
    logger.info(s"nb of sky conditions:${sc.count()}")

    logger.info(s"splitting WeatherType into ${config.weatherWeatherTypeLayers} column(s)")
    val wtRange = 0 until config.weatherWeatherTypeLayers
    Data = Data.withColumn("WeatherType", UtilUdfs.padWeatherTypeValueUdf($"WeatherType", lit(wtRange.length), lit("ZZ"))) // pad Z
      .filter("WeatherType is not null")
    Data = Data.select(Data.columns.map(c => col(c)) ++ wtRange.map(i => col("WeatherType")(i).as(s"WeatherTypeCategory_$i")): _*) // split into 5 columns
    val wt = Data.select("WeatherTypeCategory_0").distinct()
    logger.info(s"nb of weather types:${wt.count()}")

    Data = Data.drop("SkyCondition", "WeatherType")
    //Data.where("WeatherTypeCategory_0 <> 'ZZ'").show(false)

    logger.info("building stringIndex for the SkyCondition and WeatherType variables")
    var indexers = scRange.map(i => new StringIndexer().setInputCol(s"SkyConditionCategory_$i").setOutputCol(s"SkyCondition_$i")).toArray
    indexers ++:= wtRange.map(i => new StringIndexer().setInputCol(s"WeatherTypeCategory_$i").setOutputCol(s"WeatherType_$i")).toArray
    val pipeline = new Pipeline().setStages(indexers)
    val model = pipeline.fit(Data)
    Data = model.transform(Data)
      .drop(scRange.map(i => s"SkyConditionCategory_$i") ++ wtRange.map(i => s"WeatherTypeCategory_$i"): _*)

    logger.info("assembling weather conditions")
    var columns = Array("RelativeHumidity", "DryBulbCelsius", "WindSpeed", "StationPressure", "Visibility", "WindDirectionCategory")
    columns ++= scRange.map(i => s"SkyCondition_$i") ++ wtRange.map(i => s"WeatherType_$i")
    Data = Data.withColumn("WEATHER_COND", array(columns.map(c => col(c).cast(DoubleType)): _*))
      .drop(columns: _*)

    //Data.show(truncate = false)

    Data = Data.cache()
    Data
  }
}
