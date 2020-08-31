case class Configuration(cluster: String,
                         numberOfCores: Int,
                         wbanAirportsPath: String,
                         flightsPath: String,
                         flightsDelayThreshold: Int,
                         weatherPath: String,
                         weatherTimeFrame: Int,
                         weatherTimeStep: Int,
                         weatherSkyconditionLayers: Int,
                         weatherWeatherTypeLayers: Int,
                         weatherBucketizeWindDirection: Boolean,
                         mlBalanceDataset: Boolean,
                         modelPath: String
                        )
