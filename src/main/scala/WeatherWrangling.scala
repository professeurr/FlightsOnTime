import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Bucketizer, StringIndexer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}

class WeatherWrangling(val path: String, val airportWbanWrangling: AirportWbanWrangling) {

  import Utils.sparkSession.implicits._

  var Data: DataFrame = _
  val WeatherCondColumns: Array[String] = Array("RelativeHumidity", "DryBulbCelsius", "WindSpeed", "StationPressure", "Visibility", "WindDirection")
  val WeatherCategoricalColumns: Array[String] = Array("WindDirection")

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

    Utils.log("applying forward-fill on weather conditions data")
    val w0 = Window.partitionBy($"WBAN", $"Date").orderBy($"Time".asc)
    WeatherCondColumns.foreach(c => Data = Data.withColumn(c, last(col(c), ignoreNulls = true).over(w0)))
    Data = Data.na.drop()

    Data = Data.cache() // put the data into cache before the transformation steps

    Utils.log("transform WindDirection to categorical")
    val splits = Array(-1, 0, 0.1, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360)
    val bucketizer = new Bucketizer().setInputCol("WindDirection").setOutputCol("WindDirectionCategory").setSplits(splits)
    Data = bucketizer.transform(Data)
      .withColumn("WindDirection", when(col("WindDirectionCategory") === 10, "W2")
        .otherwise(concat(lit("W"), col("WindDirectionCategory").cast(IntegerType))))
      .drop("WindDirectionCategory")

    val dummyWD = (0 until 10).map(i => "W" + i).toDF("WindDirection") // use this dummy dataset to accelerate the fitting/conversion
    val indexer = new StringIndexer().setInputCol("WindDirection").setOutputCol("WindDirectionIndex");
    val pipeline = new Pipeline().setStages(Array(indexer))
    val model = pipeline.fit(dummyWD)
    Data = model.transform(Data)
      .withColumn("WindDirection", $"WindDirectionIndex")
      .drop("WindDirectionIndex")
    Utils.log(Data)

    Utils.log("assembling weather conditions")
    Data = Data.withColumn("WEATHER_COND", array(WeatherCondColumns.map(c => col(c).cast(DoubleType)): _*))
      .drop(WeatherCondColumns: _*)
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
