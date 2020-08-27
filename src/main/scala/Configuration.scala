case class Configuration(cluster: String,
                         numberOfCores: Int,
                         wbanAirportsPath: String,
                         flightsPath: String,
                         flightsDelayThreshold: Int,
                         weatherPath: String,
                         weatherTimeFrame: Int,
                         weatherTimeStep: Int,
                         weatherSkyconditionLayers: Int,
                         modelPath: String
                        )
