spark-submit \
  --master $1 \
  --deploy-mode client \
  --executor-cores 7 \
  --executor-memory 2G \
  --files ./config.json \
  --class FlightOnTimeMain \
  target/scala-2.12/flightsontime_klouvi_riva_2.12-1.0.jar
