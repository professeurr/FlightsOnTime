

object FlightOnTimeMain {

  def main(args: Array[String]): Unit = {

    val t0 = System.nanoTime()
    Utils.log("START")

    val flightsPath: String = getClass.getResource("data/flights/").getPath + "sample/*"
    val weatherPath = getClass.getResource("data/weather/").getPath + "sample/*"
    val wbanAirportsPath = getClass.getResource("data/wban_airport_timezone.csv").getPath

    val delayThreshold = 90 // the threshold of the flight delay is set to 15 minutes by default
    val partitions = Math.max(Utils.numberOfCores - 1, 1) // number of cores available on the cluster
    val weatherTimeFrame = 12 // 12h
    val weatherTimeStep = 1
    Utils.sparkSession.conf.set("spark.sql.shuffle.partitions", partitions)
    Utils.sparkSession.conf.set("spark.executor.cores", partitions)
    Utils.sparkSession.conf.set("spark.executor.instances", partitions)

    val airportWbanWrangling = new AirportWbanWrangling(wbanAirportsPath)
    airportWbanWrangling.loadData()

    val flightWrangling = new FlightWrangling(flightsPath, airportWbanWrangling, delayThreshold)
    flightWrangling.loadData()

    val weatherWrangling = new WeatherWrangling(weatherPath, airportWbanWrangling)
    weatherWrangling.loadData()

    val flightWeatherWrangling = new FlightWeatherWrangling(flightWrangling, weatherWrangling, weatherTimeFrame, weatherTimeStep)
    flightWeatherWrangling.loadData()

    val flightWeatherDecisionTree = new FlightWeatherDecisionTree(flightWeatherWrangling)
    flightWeatherDecisionTree.evaluate()

    Utils.log(s"END: ${(System.nanoTime() - t0)/1000000000} s")
  }
}
