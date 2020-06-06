

object FlightOnTimeMain {

  def main(args: Array[String]): Unit = {

    val t0 = System.nanoTime()
    Utils.log("START")

    try {

      val config = Utils.config

      val airportWbanWrangling = new AirportWbanWrangling(config.wbanAirportsPath)
      airportWbanWrangling.loadData()

      val flightWrangling = new FlightWrangling(config.flightsPath, airportWbanWrangling, config.flightsDelayThreshold)
      flightWrangling.loadData()

      val weatherWrangling = new WeatherWrangling(config.weatherPath, airportWbanWrangling)
      weatherWrangling.loadData()

      val flightWeatherWrangling = new FlightWeatherWrangling(flightWrangling, weatherWrangling, config.weatherTimeFrame, config.weatherTimeStep)
      flightWeatherWrangling.loadData()

      val flightWeatherDecisionTree = new FlightWeatherDecisionTree(flightWeatherWrangling)
      flightWeatherDecisionTree.evaluate()

    } catch {
      case e: Exception =>
        Utils.log(e.toString)
        Utils.destroy()
    }

    Utils.log(s"END: ${(System.nanoTime() - t0) / 1000000000} s")
  }
}
