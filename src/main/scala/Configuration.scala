case class Configuration(
                          verbose: Boolean,
                          wbanAirportsPath: String,
                          flightsDataPath: Array[String],
                          weatherDataPath: Array[String],
                          flightsTrainPath: Array[String],
                          weatherTrainPath: Array[String],
                          persistPath: String,
                          mlMode: String,
                          flightsDelayThreshold: Int,
                          weatherTimeFrame: Int,
                          weatherTimeStep: Int,
                          partitions: Int
                        )
