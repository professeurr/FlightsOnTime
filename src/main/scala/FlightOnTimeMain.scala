import org.apache.log4j.Logger

object FlightOnTimeMain {

  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {

    val t0 = System.nanoTime()
    logger.info("[START]")

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

      val flightWeatherDecisionTree = new FlightWeatherDecisionTree(flightWeatherWrangling, config.modelPath)
      flightWeatherDecisionTree.evaluate()

    } catch {
      case e: Exception =>
        logger.info(e.toString)
        Utils.destroy()
    }

    logger.info(s"[END: ${(System.nanoTime() - t0) / 1000000000} s]")
  }
}
