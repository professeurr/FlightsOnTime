import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Bucketizer, StringIndexer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}

class WeatherWrangling(val path: String, val airportWbanWrangling: AirportWbanWrangling) {

  import Utils.sparkSession.implicits._

  var Data: DataFrame = _
  var WeatherCondColumns: Array[String] = Array("RelativeHumidity", "DryBulbCelsius", "WindSpeed", "StationPressure", "Visibility", "WindDirection", "SkyCondition")

  def loadData(): DataFrame = {

    Utils.log("Loading weather data")
    Data = Utils.sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load(path)
      .select(col("WBAN") +: col("Date") +: col("Time") +: WeatherCondColumns.map(c => col(c)): _*)
    Utils.log(Data)

    Utils.log("cast variables")
    WeatherCondColumns.foreach(c => {
      if (c.equalsIgnoreCase("WindDirection"))
        Data = Data.withColumn(c, when(trim(col(c)) === "" || col(c) === "M", null).when(trim(col(c)) === "VR", -1).otherwise(col(c).cast(DoubleType)))
      else
        Data = Data.withColumn(c, when(trim(col(c)) === "" || col(c) === "M", null).otherwise(col(c)))
    })

    Utils.log("split SkyCondition into 5 columns")
    val scRange = 0 until 5
    Data = Data.withColumn("skyCondition", Utils.skyConditionPadValueUdf(split(trim($"SkyCondition"), " "))) // pad Z
      .filter("skyCondition is not null")
    Data = Data.select(Data.columns.map(c => col(c)) ++ scRange.map(i => col("SkyCondition")(i).as(s"SkyConditionCategory_$i")): _*) // split into 5 columns
      .drop("SkyCondition")

    Utils.log("applying forward-fill on weather conditions data")
    val w0 = Window.partitionBy($"WBAN", $"Date").orderBy($"Time".asc)
    WeatherCondColumns = WeatherCondColumns.filter(s => !s.equalsIgnoreCase("SkyCondition")) ++ scRange.map(i => s"SkyConditionCategory_$i")
    WeatherCondColumns.foreach(c => Data = Data.withColumn(c, last(col(c), ignoreNulls = true).over(w0)))
    Data = Data.na.drop().cache() // put the data into the cache before the transformation steps

    Utils.log("transforming WindDirection to categorical")
    val splits = Array(-1, 0, 0.1, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360)
    val bucketizer = new Bucketizer().setInputCol("WindDirection").setOutputCol("WindDirectionCategory").setSplits(splits)
    Data = bucketizer.transform(Data)
      .withColumn("WindDirectionCategory", when(col("WindDirectionCategory") === 10, "W2")
        .otherwise(concat(lit("W"), col("WindDirectionCategory").cast(IntegerType))))
      .drop("WindDirection")

    Utils.log("building stringIndex for the categorical variables")
    var indexers = scRange.map(i => new StringIndexer().setInputCol(s"SkyConditionCategory_$i").setOutputCol(s"SkyCondition_$i")).toArray
    indexers = indexers :+ new StringIndexer().setInputCol("WindDirectionCategory").setOutputCol("WindDirection")
    val pipeline = new Pipeline().setStages(indexers)
    val model = pipeline.fit(Data)
    Data = model.transform(Data)
    Utils.log(Data)

    Utils.log("assembling weather conditions")
    val columns = Array("RelativeHumidity", "DryBulbCelsius", "WindSpeed", "StationPressure", "Visibility", "WindDirection") ++ scRange.map(i => s"SkyCondition_$i")
    Data = Data.withColumn("WEATHER_COND", array(columns.map(c => col(c).cast(DoubleType)): _*))
      .drop(columns ++ scRange.map(i => s"SkyConditionCategory_$i") :+ "WindDirectionCategory": _*)
    Utils.log(Data)

    Utils.log("getting timezones of each station and normalizing weather time")
    Data = Data.withColumn("Date", unix_timestamp(concat_ws("", $"Date", $"Time"), "yyyyMMddHHmm"))
      .join(airportWbanWrangling.Data, $"JOIN_WBAN" === $"WBAN", "inner")
      .withColumn("WEATHER_TIME", $"Date".minus($"TimeZone"))
      .drop("Time", "JOIN_WBAN")
    Utils.log(Data)

    Utils.log("selecting useful columns")
    Data = Data.select("AirportID", "WEATHER_TIME", "WEATHER_COND")
    Utils.log(Data)

    Data = Data.cache()
    Data
  }
}
