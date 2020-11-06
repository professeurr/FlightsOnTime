case class Configuration(wbanAirportsPath: String,
                         flightsPath: Array[String],
                         weatherPath: Array[String],
                         flightsTestPath: Array[String],
                         weatherTestPath: Array[String],
                         modelPath: String,
                         flightsDelayThreshold: Int,
                         flightsFrac: Double,
                         weatherTimeFrame: Int,
                         weatherTimeStep: Int,
                         mlBalanceDataset: Boolean,
                         trainModel: Boolean,
                         testModel: Boolean,
                         clusterMode : Boolean,
                         partitions: Int
                        )
