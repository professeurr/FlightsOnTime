spark-submit \
  --master $1 \
  --deploy-mode $2 \
  --executor-cores 3 \
  --num-executors 3 \
  --executor-memory 5G \
  --files ./config.json \
  --class FlightOnTimeMain \
  target/scala-2.12/flightsontime_klouvi_riva_2.12-1.0.jar

  #--executor-cores 5
  # --num-executors 2 \
  # --executor-memory 6G \

  # cores per executor 5 (max for hdfs throughput)
  # total number of executors = (total available cores - number of nodes)/cores per executor
  # number of executors available for tasks = total number of executors - 1 // Yarn AM costs 1 executor
  # number of executors per node = (number of executors available for tasks)/number of nodes //assume that the nodes have the same capacity
  # --executor-memory = 93%(memory per node/number of executors per node) // 7% of total memory is used for Yarn memoryOverhead

  # e.g total cores = 16, total memory = 21G, total nodes = 2
  # --executor-cores 3
  # --num-executors = (16-2)/3 - 1 = 3
  # --executor-memory = 93%*21/(3+1) = 5G

  # e.g total cores = 8, total memory = 16G, total nodes = 1
  # --executor-cores 3
  # --num-executors = (8-1)/3 - 1 = 2
  # --executor-memory = 93%*16/3 = 5G

  # e.g total cores = 56, total memory = 250G, total nodes = 1
  # --executor-cores 5
  # --num-executors = (56-1)/5 - 1 = 10
  # --executor-memory = 93%*250/11= 22G

  # full command line ./run.sh spark://192.168.0.38:7077
  # hdfs web ui :98070
  # file :