name := "FlightsOnTime_Klouvi_Riva"
version := "1.0"
scalaVersion := "2.12.8"
val sparkVersion = "2.4.5"

resolvers ++= Seq(
  "apache-snapshots" at "https://repository.apache.org/snapshots/"
)


libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "compile"
  ,"org.apache.spark" %% "spark-sql" % sparkVersion % "compile"
  ,"org.apache.spark" %% "spark-mllib" % sparkVersion % "compile"
)
