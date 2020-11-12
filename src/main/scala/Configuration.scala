case class Configuration(
                          verbose: Boolean,
                          wbanAirportsPath: String,
                          flightsPath: Array[String],
                          weatherPath: Array[String],
                          persistPath: String,
                          mlMode: String,
                          flightsDelayThreshold: Int,
                          weatherTimeFrame: Int,
                          weatherTimeStep: Int,
                          partitions: Int
                        )
