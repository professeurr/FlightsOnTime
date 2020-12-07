# Prédiction des retards de vols avec Spark ML

## Context technique
Le projet est réalisé avec `scala 2.012.10` et `Spark 3.0.0`

## Exécuter l'application
Le package de l'application (.jar)  se trouve dans le répertoire `./bin` situé à la racine du projet.
Il s'exécute avec la commande spark-submit comme suit:

```shell
spark-submit \
  --master spark://127.0.0.1:7077 \
  --deploy-mode client \
  --executor-cores 4 \
  --num-executors 2 \
  --executor-memory 4G \
  --files ./config.json \
  --conf spark.sql.autoBroadcastJoinThreshold=-1 \
  --conf spark.sql.execution.useObjectHashAggregateExec=true \
  --conf spark.sql.objectHashAggregate.sortBased.fallbackThreshold=12800000 \
  --conf spark.sql.join.preferSortMergeJoin=false \
  --conf spark.sql.shuffle.partitions=64 \
  --conf spark.executor.memoryOverhead=4G \
  --conf spark.driver.memoryOverhead=4G \
  --conf spark.driver.memory=6G \
  --class Main \
  target/scala-2.12/flightsontime_klouvi_riva_2.12-1.0.jar
```
Vous avez la possibilité d'adapter les paramètres en fonction de la configuration de votre cluster. 
Vous devez fournir un fichier de configuration nommé `config.json` à l'application. Ci-dessous un exemple de ce fichier:

```json
{
  "ml_mode": "extract, transform, train, evaluate",
  "root_path": "file:///home/masterai/dev/master_iasd/bigdata/project/flightsontime/workspace/",
  "persist_path": "persist/",
  "wban_airports_path": "data/wban_airport_timezone.csv",
  "flights_data_path": [
    "data/flights/2013*"
  ],
  "weather_data_path": [
    "data/weather/2013*"
  ],
  "flights_extract_path": [
    "flights/year=2013"
  ],
  "weather_extract_path": [
    "weather/year=2013"
  ],
  "flights_test_path": [
    "data/flights/201301*"
  ],
  "weather_test_path": [
    "data/weather/201301*"
  ],
  "features": -1,
  "flights_delay_threshold": 15,
  "weather_time_frame": 12,
  "weather_time_step": 1,
  "verbose": false
}

```
- `ml_mode`: permet de spécifier quelle(s) action(s) effectuée(s). Les actions son réalisées séquentiellement.
             -- `extract`: features extraction sur les fichiers placés `flights_data_path` et `weather_data_path`
             -- `transform`: la tranformation et la jointure des données extraites lors de l'étape `extract`
             -- `train`: l'étape d'entraînement du modèle ML sur les données obtenues après la phase `transform`. Le modèle est sauvegardé pour réaliser la phase d'évaluation
             `evaluate`: le modèle est évalué en utilisant les données placées dans `flights_test_path` et `weather_test_path`.
- `root_path`: le répertoire racine où sont stockées les données météo et vols en format CSV. 
- `persist_path`: les données intermediaires seront sauvegardées dans ce répertoire au format `parquet`
- `flights_delay_threshold`: le temps en minutes au-delà duquel vol sera considéré en retard
- `weather_time_frame`: la plage horaire (en heure) sur laquelle les observations météo son prises en compte avvant le départ et l'arrivée d'un vol.

## Le code source
Les fichiers du code source sont placés dans le répertoire `src/main/scala`.
- `Configuration`: cette classe permet de lire le fichier de configuration.
- `DataFeaturing`: cette classe lit les données brutes de vols et météo en format CSV, réalise des sélections et sauvegarde le résultat en format parquet.
- `DataTranformer`: réalise principalement la jointure entre les données de vols et de météo.
- `FlightModel`: contient l'implémentation des modèles (Cross Validation, Random Forest, Decision Tree). Il permet également d'évaluer les modèles.
