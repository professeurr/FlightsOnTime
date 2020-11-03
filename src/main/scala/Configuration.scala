case class Configuration(wbanAirportsPath: String,
                         flightsPath: Array[String],
                         weatherPath: Array[String],
                         modelPath: String,
                         flightsDelayThreshold: Int,
                         flightsFrac: Double,
                         weatherTimeFrame: Int,
                         weatherTimeStep: Int,
                         mlBalanceDataset: Boolean
                        )
