# formatted_zone_idescat_simple_final_fix_v2.py

from pathlib import Path
from typing import Optional

# --- PySpark Imports ---
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, TimestampType, LongType
)
# ---------------------

# --- Delta Lake Package Configuration ---
DELTA_PACKAGE = "io.delta:delta-spark_2.12:3.3.0"


class IdescatFormattedZone:
    """
    Reads Idescat landing zone Parquet, applies schema and types (handling timestamp issue),
    and writes to a Delta Lake table in the Formatted Zone. (Simplified & Fixed v2)
    """

    def __init__(self, spark: SparkSession, input_path: str or Path, output_path: str or Path):
        self.spark = spark
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        self.target_schema = self._define_target_schema()
        print(f"Formatted Zone Initialized. Input: {self.input_path}, Output: {self.output_path}")

    def _define_target_schema(self) -> StructType:
        """Defines the FINAL desired schema for the formatted table."""
        return StructType([
            StructField("municipality_id", StringType(), True),
            StructField("municipality_name", StringType(), True),
            StructField("comarca_name", StringType(), True),
            StructField("indicator_id", StringType(), True),
            StructField("indicator_name", StringType(), True),
            StructField("reference_year", IntegerType(), True), # Final Type: Integer
            StructField("municipality_value", DoubleType(), True),
            StructField("comarca_value", DoubleType(), True),
            StructField("catalunya_value", DoubleType(), True),
            StructField("source_update_timestamp", TimestampType(), True)
        ])

    def _define_read_schema(self) -> StructType:
         """Defines schema to READ Parquet, matching physical types written by Pandas."""
         # Based on landing zone script output columns & types
         return StructType([
            StructField("municipality_id", StringType(), True),
            StructField("municipality_name", StringType(), True),
            StructField("comarca_name", StringType(), True),
            StructField("indicator_id", StringType(), True),
            StructField("indicator_name", StringType(), True),
            StructField("reference_year", LongType(), True),     # !!! READ AS LONG (matches Int64 from Pandas) !!!
            StructField("municipality_value", DoubleType(), True), # Pandas already coerced to numeric (float64 -> double)
            StructField("comarca_value", DoubleType(), True),    # Pandas already coerced to numeric (float64 -> double)
            StructField("catalunya_value", DoubleType(), True),  # Pandas already coerced to numeric (float64 -> double)
            StructField("source_update_timestamp", LongType(), True) # Read timestamp as Long (nanos)
        ])

    def run(self):
        """Executes the Formatted Zone processing for the Idescat data."""
        print(f"--- Running Idescat Formatted Zone Task ---")
        try:
            # 1. Define Read Schema (Matches physical types in Parquet)
            read_schema = self._define_read_schema()

            # 2. Read Landing Data using the specific read_schema
            print(f"Reading landing data from: {self.input_path} using explicit read schema")
            df_landing = self.spark.read.schema(read_schema).parquet(self.input_path)
            print("Landing data read schema:")
            df_landing.printSchema()

            # 3. Select columns and Cast to TARGET Schema
            print("Applying final schema and casting types...")

            select_expressions = []
            for field in self.target_schema.fields:
                 col_name = field.name
                 target_type = field.dataType

                 if col_name not in df_landing.columns:
                      print(f"Warning: Column '{col_name}' expected but not found. Adding as null.")
                      select_expressions.append(F.lit(None).cast(target_type).alias(col_name))
                      continue

                 current_type = df_landing.schema[col_name].dataType
                 # --- Special handling for timestamp ---
                 if col_name == "source_update_timestamp" and isinstance(current_type, LongType):
                      print(f"  - Converting '{col_name}' from Long (nanos) to Timestamp")
                      select_expressions.append(
                          (F.col(col_name) / 1e9).cast("timestamp").alias(col_name)
                      )
                 # --- Cast other columns ONLY if types differ ---
                 elif current_type != target_type:
                      print(f"  - Casting '{col_name}' from {current_type} to {target_type}")
                      select_expressions.append(F.col(col_name).cast(target_type).alias(col_name))
                 # --- Keep column as is if type already matches ---
                 else:
                      # print(f"  - Keeping '{col_name}' as {current_type}") # Optional print
                      select_expressions.append(F.col(col_name)) # No cast needed

            df_formatted = df_landing.select(select_expressions)

            print("Schema applied. Final schema:")
            df_formatted.printSchema()

            # 4. Write to Delta Lake
            print(f"Writing formatted data to Delta table: {self.output_path}")
            df_formatted.write.format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(self.output_path)

            print(f"Formatted data successfully written to {self.output_path}")
            print("--- Idescat Formatted Zone Task Successfully Completed ---")

        except Exception as e:
            print(f"!!! Error during Formatted Zone execution: {e}")
            import traceback
            traceback.print_exc()
            print("--- Idescat Formatted Zone Task Failed ---")


# --- Spark Session Creation Helper (No changes needed here from previous fix) ---
def get_spark_session() -> SparkSession:
    """
    Initializes and returns a SparkSession configured for Delta Lake
    and updated Parquet compatibility settings.
    """
    print("Initializing Spark Session...")
    try:
        # Ensure consistent indentation for all chained methods
        spark = SparkSession.builder \
            .appName("IdescatFormattedZoneSimple") \
            .master("local[*]") \
            .config("spark.jars.packages", DELTA_PACKAGE) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.sql.parquet.int96AsTimestamp", "true") \
            .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")
        print("Spark Session Initialized. Log level set to ERROR.")
        return spark
    except Exception as e:
        print(f"FATAL: Error initializing Spark Session: {e}")
        raise

# --- Main Execution Block ---
if __name__ == "__main__":
    INPUT_PARQUET = "./data/landing/idescat.parquet" # Make sure this file exists
    OUTPUT_DELTA = "./data/formatted/idescat"

    spark = None
    try:
        spark = get_spark_session()
        formatter = IdescatFormattedZone(spark=spark, input_path=INPUT_PARQUET, output_path=OUTPUT_DELTA)
        formatter.run()
    except Exception as main_error:
        print(f"An error occurred outside the run method: {main_error}")
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            print("Stopping Spark Session.")
            spark.stop()