#!/bin/bash

spark-submit \
  --master $1 \
  --deploy-mode $2 \
  --executor-cores 8 \
  --num-executors 4 \
  --executor-memory 8G \
  --files ./config.json \
  --class Main \
  flightsontime_klouvi_riva_2.12-1.0.jar


#./run-remote.sh spark://vmhadoopmaster.cluster.lamsade.dauphine.fr:7077 client