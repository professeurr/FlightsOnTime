import org.apache.log4j.Logger
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Bucketizer, StringIndexer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

class WeatherWrangling(val path: String, val airportWbanWrangling: AirportWbanWrangling) {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  import Utils.sparkSession.implicits._

  var Data: DataFrame = _
  var WeatherCondColumns: Array[String] = Array("RelativeHumidity", "DryBulbCelsius", "WindSpeed", "StationPressure", "Visibility", "WindDirection", "SkyCondition", "WeatherType")

  def loadData(): DataFrame = {

    logger.info("Loading weather data")
    Data = Utils.sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load(path)
      .select(col("WBAN") +: col("Date") +: col("Time") +: WeatherCondColumns.map(c => col(c)): _*)
    logger.info(Data.schema.treeString)
    logger.info(s"weather initial data: ${Data.count()}")

    logger.info("getting timezones of each station and normalizing weather time...")
    Data = Data.join(airportWbanWrangling.Data, $"MAPPING_WBAN" === $"WBAN", "inner")
      .withColumn("WEATHER_TIME", unix_timestamp(concat_ws("", $"Date", $"Time"), "yyyyMMddHHmm").minus($"TimeZone"))
      .drop("Time", "MAPPING_WBAN", "Date")
    logger.info(Data.schema.treeString)
    logger.info(s"weather data after the mapping: ${Data.count()}")
    Data.show(truncate = false)

    logger.info("cast variables")
    WeatherCondColumns.foreach(c => {
      if (c.equalsIgnoreCase("WindDirection"))
        Data = Data.withColumn(c, when(trim(col(c)) === "" || col(c) === "M", null).when(trim(col(c)) === "VR", -1).otherwise(col(c).cast(DoubleType)))
      else
        Data = Data.withColumn(c, when(trim(col(c)) === "" || col(c) === "M", null).otherwise(col(c)))
    })

    logger.info(s"data partitions: ${Data.rdd.getNumPartitions}")
    logger.info("splitting SkyCondition into 5 columns")
    val scRange = 0 until 5
    Data = Data.withColumn("skyCondition", UtilUdfs.skyConditionPadValueUdf(split(trim($"SkyCondition"), " "))) // pad Z
      .filter("skyCondition is not null")
    Data = Data.select(Data.columns.map(c => col(c)) ++ scRange.map(i => col("SkyCondition")(i).as(s"SkyConditionCategory_$i")): _*) // split into 5 columns
      .drop("SkyCondition")

    logger.info("applying forward-fill on weather conditions data")
    val w0 = Window.partitionBy($"WBAN").orderBy($"WEATHER_TIME".asc)
    WeatherCondColumns = WeatherCondColumns.filter(s => !s.equalsIgnoreCase("SkyCondition")) ++ scRange.map(i => s"SkyConditionCategory_$i")
    WeatherCondColumns.foreach(c => Data = Data.withColumn(c, last(col(c), ignoreNulls = true).over(w0)))

    logger.info(s"weather data before na.drop(): ${Data.count()}")
    Data = Data.na.drop().cache() // put the data into the cache before the transformation steps
    logger.info(s"weather data after na.drop(): ${Data.count()}")

    logger.info("transforming WindDirection to categorical")
    val splits = Array(-1, 0, 0.1, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360)
    val bucketizer = new Bucketizer().setInputCol("WindDirection").setOutputCol("WindDirectionCategory").setSplits(splits)
    Data = bucketizer.transform(Data)
      .withColumn("WindDirectionCategory", when(col("WindDirectionCategory") === 10, 2)
        .otherwise(col("WindDirectionCategory")))
      .drop("WindDirection")

    logger.info("building stringIndex for the SkyCondition variables")
    logger.info(s"data partitions: ${Data.rdd.getNumPartitions}")
    var indexers = scRange.map(i => new StringIndexer().setInputCol(s"SkyConditionCategory_$i").setOutputCol(s"SkyCondition_$i")).toArray
    indexers :+= new StringIndexer().setInputCol("WeatherType").setOutputCol("WeatherTypeCategory")
    val pipeline = new Pipeline().setStages(indexers)
    val model = pipeline.fit(Data)
    Data = model.transform(Data)
    logger.info(Data.schema.treeString)

    logger.info("assembling weather conditions")
    var columns = Array("RelativeHumidity", "DryBulbCelsius", "WindSpeed", "StationPressure", "Visibility", "WindDirectionCategory", "WeatherTypeCategory")
    columns ++= scRange.map(i => s"SkyCondition_$i")
    Data = Data.withColumn("WEATHER_COND", array(columns.map(c => col(c).cast(DoubleType)): _*))
      .drop(columns ++ scRange.map(i => s"SkyConditionCategory_$i") :+ "WeatherType": _*)
    logger.info(Data.schema.treeString)

    logger.info("selecting useful columns")
    Data = Data.select("AirportID", "WEATHER_TIME", "WEATHER_COND")
    logger.info(Data.schema.treeString)

    Data = Data.cache()
    Data
  }
}
