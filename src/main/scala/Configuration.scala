

case class Configuration(
                          verbose: Boolean,
                          features: Int = 8,
                          rootPath: String = "./",
                          var wbanAirportsPath: String,
                          var flightsDataPath: Array[String],
                          var weatherDataPath: Array[String],
                          var flightsExtractPath: Array[String],
                          var weatherExtractPath: Array[String],
                          var flightsTestPath: Array[String],
                          var weatherTestPath: Array[String],
                          var persistPath: String,
                          var mlMode: String = "extract",
                          flightsDelayThreshold: Int = 15,
                          weatherTimeFrame: Int = 12,
                          weatherTimeStep: Int = 1,
                        ) {

  var flightsPath:Array[String] = _
  var weatherPath:Array[String] = _
  var partitions: Int = -1
  var executionModes: Array[String] = Array()

  def init(): Configuration = {
    executionModes = mlMode.trim.split(",").map(x => x.trim.toLowerCase())
    partitions = Utility.sparkSession.conf.get("spark.sql.shuffle.partitions").toInt

    persistPath = rootPath + persistPath
    wbanAirportsPath = rootPath + wbanAirportsPath

    flightsDataPath = flightsDataPath.map(x => rootPath + x)
    weatherDataPath = weatherDataPath.map(x => rootPath + x)

    flightsExtractPath = flightsExtractPath.map(x => persistPath + "extract/" + x)
    weatherExtractPath = weatherExtractPath.map(x => persistPath + "extract/" + x)

    flightsTestPath = flightsTestPath.map(x => rootPath + x)
    weatherTestPath = weatherTestPath.map(x => rootPath + x)

    this
  }

}
