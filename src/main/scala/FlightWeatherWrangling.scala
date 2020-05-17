import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

class FlightWeatherWrangling(flightWrangling: FlightWrangling, weatherWrangling: WeatherWrangling, weatherTimeFrame: Int) {

  import Utils.sparkSession.implicits._

  var OriginData: DataFrame = _
  var DestinationData: DataFrame = _
  var Data: DataFrame = _

  def loadData(): DataFrame = {
    val tf: Int = 3600 * weatherTimeFrame

    Utils.log("loading origin weather data")
    OriginData = flightWrangling.Data.join(weatherWrangling.Data, $"ORIGIN_AIRPORT_ID" === $"AirportId", "inner")
      .drop("DEST_AIRPORT_ID", "ORIGIN_AIRPORT_ID", "AirportID", "FL_CRS_ARR_TIME")
      .filter(s"WEATHER_TIME >= FL_CRS_DEP_TIME - $tf and WEATHER_TIME <= FL_CRS_DEP_TIME")
    Utils.log(OriginData)

    Utils.log("building origin weather data")
    OriginData = OriginData.groupBy($"FL_ID", $"FL_CRS_DEP_TIME", $"FL_ONTIME")
      .agg(Utils.fillMissingDataUdf($"FL_CRS_DEP_TIME",
        collect_list($"WEATHER_TIME"), collect_list($"WEATHER_COND"), lit(weatherTimeFrame)).as("WEATHER_COND"))
      .filter("WEATHER_COND is not null")
      .drop("FL_CRS_DEP_TIME")
    Utils.log(OriginData)

    Utils.log("loading destination weather data")
    DestinationData = flightWrangling.Data.join(weatherWrangling.Data, $"DEST_AIRPORT_ID" === $"AirportId", "inner")
      .drop("DEST_AIRPORT_ID", "ORIGIN_AIRPORT_ID", "AirportID", "FL_CRS_DEP_TIME")
      .filter(s"WEATHER_TIME >= FL_CRS_ARR_TIME - $tf and WEATHER_TIME <= FL_CRS_ARR_TIME")
    Utils.log(DestinationData)

    Utils.log("building destination weather data")
    DestinationData = DestinationData.groupBy($"FL_ID", $"FL_CRS_ARR_TIME", $"FL_ONTIME")
      .agg(Utils.fillMissingDataUdf($"FL_CRS_ARR_TIME",
        collect_list($"WEATHER_TIME"), collect_list($"WEATHER_COND"), lit(weatherTimeFrame)).as("WEATHER_COND"))
      .filter("WEATHER_COND is not null")
      .drop("FL_CRS_ARR_TIME")
    Utils.log(DestinationData)

    Utils.log("Building final dataset with origin and destion weather conditions + on-time flag")
    Data = OriginData.as("origin").join(DestinationData.as("dest"), $"origin.FL_ID" === $"dest.FL_ID")
      .select($"origin.FL_ID".as("FL_ID"), $"origin.FL_ONTIME".as("FL_ONTIME"),
        $"origin.WEATHER_COND".as("ORIGIN_WEATHER_COND"),
        $"dest.WEATHER_COND".as("DEST_WEATHER_COND"))
    Utils.log(Data)

    Utils.log("exploding features column into individual columns")
    val columns = weatherWrangling.WeatherCondColumns
    val catColumns = weatherWrangling.WeatherCategoricalColumns
    Data = Data.select(col("FL_ID") +: col("FL_ONTIME") +: (
      (0 until weatherTimeFrame).flatMap(i => columns.indices.map(c => {
        if (!catColumns.contains(columns(c)))
          col("ORIGIN_WEATHER_COND")(i)(c).cast(DoubleType).alias(s"ORIGIN_${columns(c)}_${i}h")
        else
          col("ORIGIN_WEATHER_COND")(i)(c).alias(s"ORIGIN_${columns(c)}_${i}h")
      })) ++
        (0 until weatherTimeFrame).flatMap(i => columns.indices.map(c => {
          if (!catColumns.contains(columns(c)))
            col("DEST_WEATHER_COND")(i)(c).cast(DoubleType).alias(s"DEST_${columns(c)}_${i}h")
          else
            col("DEST_WEATHER_COND")(i)(c).alias(s"DEST_${columns(c)}_${i}h")
        }))
      ): _*
    )

    Data = Data.cache() // Very important to cache the data here to speed up the data propagation during the pipeline execution

    val features = Data.columns.filter(c => !c.startsWith("FL_") && !catColumns.exists(x => c.contains(x)))
    val assembler = new VectorAssembler()
      .setInputCols(features).setOutputCol("WEATHER_COND")

    val pipeline = new Pipeline().setStages(Array(assembler))

    Utils.log(s"Fitting features transformation model")
    val model = pipeline.fit(Data)

    Utils.log(s"Transforming data")
    Data = model.transform(Data)
      .select("FL_ID", "FL_ONTIME", "WEATHER_COND")

    Utils.log(Data)

    Data = Data.cache() // cache the resulting data for being used during ML analysis

    Data
  }
}
