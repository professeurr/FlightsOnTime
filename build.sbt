name := "FlightsOnTime_Klouvi_Riva"
version := "1.0"
scalaVersion := "2.12.10"
val sparkVersion = "3.0.0-preview2"

resolvers ++= Seq(
  "apache-snapshots" at "https://repository.apache.org/snapshots/"
)


libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "compile"
  ,"org.apache.spark" %% "spark-sql" % sparkVersion % "compile"
  ,"org.apache.spark" %% "spark-mllib" % sparkVersion % "compile"
  ,"org.apache.spark" %% "spark-ml" % sparkVersion % "compile"
)
