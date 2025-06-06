from pyspark.sql import SparkSession
DELTA_PACKAGE = "io.delta:delta-spark_2.12:3.3.0"

def get_spark_session() -> SparkSession:
    """
    Initializes and returns a SparkSession configured for Delta Lake.
    """
    print("Initializing Spark Session...")
    try:
        spark = SparkSession.builder \
            .appName("LloguerFormattedZone") \
            .master("local[*]") \
            .config("spark.jars.packages", DELTA_PACKAGE) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.sql.parquet.int96AsTimestamp", "true") \
            .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR") # Reduce verbosity
        print("Spark Session Initialized. Log level set to ERROR.")
        return spark
    except Exception as e:
        print(f"FATAL: Error initializing Spark Session: {e}")
        raise