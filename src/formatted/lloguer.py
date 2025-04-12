from pathlib import Path

# --- PySpark Imports ---
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)

class LloguerFormattedZone:
    """
    Reads Lloguer landing zone Parquet (based on actual observed schema),
    applies schema and types, and writes to a Delta Lake table in the Formatted Zone.
    Includes an option for verbose output inspection.
    """

    def __init__(self, spark: SparkSession, input_path: str = "./data/landing/lloguer.parquet", output_path: str = "./data/formatted/lloguer"):
        """
        Initializes the formatter.

        Args:
            spark: Active SparkSession.
            input_path: Path to the landing zone Parquet file (lloguer.parquet).
            output_path: Path to the output Delta table directory.
        """
        self.spark = spark
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        # Define the target schema based on the actual landing data
        self.target_schema = self._define_target_schema()
        print(f"Lloguer Formatted Zone Initialized.")
        print(f"  Input Path: {self.input_path}")
        print(f"  Output Delta Path: {self.output_path}")

    def _define_target_schema(self) -> StructType:
        """
        Defines the FINAL desired schema for the formatted Lloguer table,
        matching the columns observed in the landing parquet file.
        """
        # Updated schema based on the provided pandas.info() output
        return StructType([
            StructField("ambit_territorial", StringType(), True),
            StructField("codi_territorial", StringType(), True),
            StructField("nom_territori", StringType(), True),
            StructField("any", IntegerType(), True),           # Target: Integer
            StructField("periode", StringType(), True),
            StructField("habitatges", IntegerType(), True),      # Target: Integer (assuming this is count)
            StructField("renda", DoubleType(), True),          # Target: Double (average rent, allows nulls)
            StructField("tram_preus", StringType(), True)
        ])

    def _inspect_output_delta(self):
        """Reads the output Delta table and prints schema and sample data."""
        print(f"--- Inspecting Output Delta Table: {self.output_path} ---")
        try:
            # Check if the Delta path exists before attempting to read
            # Note: For Delta, checking for _delta_log directory is more reliable than the base path
            delta_log_path = Path(self.output_path) / "_delta_log"
            if not delta_log_path.is_dir():
                 print(f"Inspection failed: Delta log not found at {delta_log_path}. Table might not exist or is corrupted.")
                 return

            df_read_back = self.spark.read.format("delta").load(self.output_path)
            count = df_read_back.count()
            print(f"Successfully read back {count} records from Delta table.")

            print("\nFinal Schema in Delta Table:")
            df_read_back.printSchema()

            print(f"\nSample Data (First 5 rows) from Delta Table:")
            # Use truncate=False to see full column content
            df_read_back.show(5, truncate=False)

            # Optional: Add more checks here, e.g., count nulls in key columns
            # print("\nNull counts per column:")
            # df_read_back.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in df_read_back.columns]).show()

        except Exception as e:
             print(f"!!! Inspection failed: Error reading or processing Delta table: {e}")
             import traceback
             traceback.print_exc()
        print(f"--- Finished Inspecting Output Delta Table ---")


    def run(self, verbose: bool = False):
        """
        Executes the Formatted Zone processing for the Lloguer data.

        Args:
            verbose (bool): If True, inspect the output Delta table after writing.
                           Defaults to False.
        """
        print(f"--- Running Lloguer Formatted Zone Task (Verbose: {verbose}) ---")
        write_successful = False
        try:
            # 1. Read Landing Data
            print(f"Reading landing data from: {self.input_path}")
            df_landing = self.spark.read.parquet(self.input_path)
            print("Landing data schema (inferred by Spark):")
            df_landing.printSchema()

            # 2. Select columns and Cast to TARGET Schema
            print("Applying final schema and casting types...")
            select_expressions = []
            landing_cols = df_landing.columns
            for field in self.target_schema.fields:
                 col_name = field.name
                 target_type = field.dataType
                 if col_name not in landing_cols:
                      print(f"Warning: Target column '{col_name}' not found in landing data. Adding as null column.")
                      select_expressions.append(F.lit(None).cast(target_type).alias(col_name))
                 else:
                      current_type = df_landing.schema[col_name].dataType
                      if current_type != target_type:
                           print(f"  - Casting '{col_name}' from {current_type} to {target_type}")
                           select_expressions.append(F.col(col_name).cast(target_type).alias(col_name))
                      else:
                           select_expressions.append(F.col(col_name))

            df_formatted = df_landing.select(select_expressions)
            print("Schema after casting. Final desired schema:")
            df_formatted.printSchema()

            # 3. Write to Delta Lake
            print(f"Writing formatted data to Delta table: {self.output_path}")
            df_formatted.write.format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(self.output_path)
            write_successful = True # Mark write as successful
            print(f"Formatted Lloguer data successfully written to {self.output_path}")

            # 4. Inspect if requested and write was successful
            if verbose and write_successful:
                self._inspect_output_delta()

            print("--- Lloguer Formatted Zone Task Successfully Completed ---")

        except Exception as e:
            print(f"!!! Error during Lloguer Formatted Zone execution: {e}")
            import traceback
            traceback.print_exc()
            print("--- Lloguer Formatted Zone Task Failed ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    from src.spark_session import get_spark_session

    spark = None
    try:
        spark = get_spark_session()
        formatter = LloguerFormattedZone(spark=spark)
        # Set verbose=True to inspect the output Delta table after writing
        formatter.run(verbose=True)

    except Exception as main_error:
        print(f"An unexpected error occurred in the main execution block: {main_error}")
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            print("Stopping Spark Session.")
            spark.stop()