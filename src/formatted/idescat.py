# --- PySpark Imports ---
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, TimestampType, LongType
)

class IdescatFormattedZone:
    """
    Reads Idescat landing zone Parquet, normalizes municipality IDs,
    applies schema and types, and writes to a Delta Lake table in the Formatted Zone.
    """

    def __init__(self, spark: SparkSession, input_path: str = "./data/landing/idescat.parquet", output_path: str = "./data/formatted/idescat"):
        self.spark = spark
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        self.target_schema = self._define_target_schema()
        print(f"Formatted Zone Initialized. Input: {self.input_path}, Output: {self.output_path}")

    def _define_target_schema(self) -> StructType:
        """Defines the FINAL desired schema for the formatted table."""
        return StructType([
            StructField("municipality_id", StringType(), True), # Standard 5-digit ID
            StructField("municipality_name", StringType(), True),
            StructField("comarca_name", StringType(), True),
            StructField("indicator_id", StringType(), True),
            StructField("indicator_name", StringType(), True),
            StructField("reference_year", IntegerType(), True),
            StructField("municipality_value", DoubleType(), True),
            StructField("comarca_value", DoubleType(), True),
            StructField("catalunya_value", DoubleType(), True),
            StructField("source_update_timestamp", TimestampType(), True)
        ])

    def _define_read_schema(self) -> StructType:
         """Defines schema to READ Parquet, matching physical types written by Pandas."""
         return StructType([
            StructField("municipality_id", StringType(), True), # Read original ID as String
            StructField("municipality_name", StringType(), True),
            StructField("comarca_name", StringType(), True),
            StructField("indicator_id", StringType(), True),
            StructField("indicator_name", StringType(), True),
            StructField("reference_year", LongType(), True),
            StructField("municipality_value", DoubleType(), True),
            StructField("comarca_value", DoubleType(), True),
            StructField("catalunya_value", DoubleType(), True),
            StructField("source_update_timestamp", LongType(), True)
        ])

    def run(self):
        """Executes the Formatted Zone processing for the Idescat data, including municipality_id normalization."""
        print(f"--- Running Idescat Formatted Zone Task ---")
        try:
            # 1. Define Read Schema (Matches physical types in Parquet)
            read_schema = self._define_read_schema()

            # 2. Read Landing Data using the specific read_schema
            print(f"Reading landing data from: {self.input_path} using explicit read schema")
            df_landing = self.spark.read.schema(read_schema).parquet(self.input_path)
            print("Landing data read schema:")
            df_landing.printSchema()

            # 3. Normalize Municipality ID
            print("Normalizing municipality_id...")
            df_normalized = df_landing.withColumn(
                "municipality_id_standard",
                # Ensure municipality_id is treated as string for length/substring checks
                F.col("municipality_id").cast(StringType())
            ).withColumn(
                "municipality_id_standard",
                 # If length is 5, prepend '0'
                 F.when(F.length(F.col("municipality_id_standard")) == 5, F.lpad(F.col("municipality_id_standard"), 6, '0'))
                  # Otherwise, keep the original (assuming it's already 6 digits or some other edge case)
                  .otherwise(F.col("municipality_id_standard"))
            ).withColumn(
                "municipality_id_standard",
                # Now, remove the last digit from the (potentially padded) 6-digit code
                F.substring(F.col("municipality_id_standard"), 1, 5) # Substring(column, startPosition, length) - Positions are 1-based
            )
            # Overwrite the original column with the standardized one
            df_normalized = df_normalized.drop("municipality_id").withColumnRenamed("municipality_id_standard", "municipality_id")
            print("Normalization complete. Sample IDs after normalization:")
            df_normalized.select("municipality_id").distinct().show(5, truncate=False)

            # 4. Select columns and Cast to TARGET Schema (using df_normalized now)
            print("Applying final schema and casting types...")

            select_expressions = []
            # Use the normalized dataframe for further processing
            df_processed = df_normalized

            for field in self.target_schema.fields:
                 col_name = field.name
                 target_type = field.dataType

                 if col_name not in df_processed.columns: # Check against df_processed
                      print(f"Warning: Column '{col_name}' expected but not found. Adding as null.")
                      select_expressions.append(F.lit(None).cast(target_type).alias(col_name))
                      continue

                 current_type = df_processed.schema[col_name].dataType # Get type from df_processed
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
                      select_expressions.append(F.col(col_name)) # No cast needed

            df_formatted = df_processed.select(select_expressions) # Select from df_processed

            print("Schema applied. Final schema:")
            df_formatted.printSchema()

            # 5. Write to Delta Lake
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

# --- Main Execution Block ---
if __name__ == "__main__":
    from src.spark_session import get_spark_session

    spark = None
    try:
        spark = get_spark_session()
        formatter = IdescatFormattedZone(spark=spark)
        formatter.run()
    except Exception as main_error:
        print(f"An error occurred outside the run method: {main_error}")
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            print("Stopping Spark Session.")
            spark.stop()