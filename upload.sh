#!/bin/bash

id=159


#scp -i ~/.ssh/id_rsa_user159 -P 993  ./data/flights/full/2014*.csv user$id@www.lamsade.dauphine.fr:~/data/

scp -i ~/.ssh/id_rsa_user$id -P 993  ./target/scala-2.12/flightsontime_klouvi_riva_2.12-1.0.jar user$id@www.lamsade.dauphine.fr:~/projects/flights

cp ./target/scala-2.12/flightsontime_klouvi_riva_2.12-1.0.jar ./bin/

#scp -i ~/.ssh/id_rsa_user$id -P 993  ./config-remote.json user$id@www.lamsade.dauphine.fr:~/projects/flights/config.json
#scp -i ~/.ssh/id_rsa_user159 -P 993  ./run-remote.sh user$id@www.lamsade.dauphine.fr:~/projects/flights

#ssh -i ~/.ssh/id_rsa_user$id -p 993  user$id@www.lamsade.dauphine.fr -t "cd ~/projects/flights/ ; ~/projects/flights/run-remote.sh spark://127.0.0.1:7077 client; bash"

#ssh -i ~/.ssh/id_rsa_user159 -p 993  $user_id@www.lamsade.dauphine.fr # ./run.sh ./kmeans_scala_klouvi_riva_2.11-1.0.jar hdfs:///user/$user_id/iris.data.txt
#exit

# ssh -i ~/.ssh/id_rsa_user159 -p 993  user159@www.lamsade.dauphine.fr
