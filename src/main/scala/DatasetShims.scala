import org.apache.spark.sql.Dataset

trait DatasetShims {

  implicit class DatasetHelper[T](df: Dataset[T]) {

    def toShowString(numRows: Int = 20, truncate: Int = 20, vertical: Boolean = false): String =
      df.toShowString(numRows, truncate, vertical)

    def printSchema(): Unit = {
      df.printSchema()
    }

    def toShowString: String = {
      "\n" + df.schema.treeString
    }
  }
}
