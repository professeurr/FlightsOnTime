name := "FlightsOnTime_Klouvi_Riva"

version := "1.0"

scalaVersion := "2.12"

val sparkVersion = "2.4.5"


resolvers ++= Seq(
  "apache-snapshots" at "https://repository.apache.org/snapshots/"
)


libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "compile"
  ,"org.apache.spark" %% "spark-sql" % sparkVersion % "compile"
  //,"org.apache.spark" %% "spark-streaming" % sparkVersion % "compile"
  ,"org.apache.spark" %% "spark-mllib" % sparkVersion % "compile"
  //,"org.apache.spark" %% "spark-hive" % sparkVersion % "compile"
  //,"org.apache.spark" %% "spark-yarn" % sparkVersion % "compile"
  //,"mysql" % "mysql-connector-java" % "5.1.6"
)

/*
lazy val pushPackageTask = TaskKey[Unit]("pushPackage", "Push compiled package to cluster")

pushPackageTask := {
  import sys.process._
  Seq("scp", "-i ~/.ssh/id_rsa_user159 -P 993  run.sh user159@www.lamsade.dauphine.fr:~/")
  Seq("scp", "-i ~/.ssh/id_rsa_user159 -P 993  target/scala-2.11/scala_2.11-0.1.jar user159@www.lamsade.dauphine.fr:~/")!
}

`package` := (pushPackageTask dependsOn `package`).value
*/
