spark-submit \
  --master $1 \
  --deploy-mode cluster \
  --executor-cores 7 \
  --executor-memory 14G \
  --files ./config.json \
  --class FlightOnTimeMain \
  target/scala-2.12/flightsontime_klouvi_riva_2.12-1.0.jar
