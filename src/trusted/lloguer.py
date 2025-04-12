# trusted_zone_lloguer.py

from pathlib import Path
from typing import Optional, Union
import datetime # To get current year

# --- PySpark Imports ---
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window # Import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)
# ---------------------

# --- Delta Lake Package Configuration (Keep consistent) ---
DELTA_PACKAGE = "io.delta:delta-spark_2.12:3.3.0" # Ensure this version is compatible

class LloguerTrustedZone:
    """
    Cleans and validates the formatted Lloguer Catalunya data,
    applying quality rules (Denial Constraints) and deduplication
    before saving to the Trusted Zone as a Delta table. Includes checks
    for duplicate keys with differing values.
    """

    def __init__(self, spark: SparkSession, input_path: Union[str, Path] = None, output_path: Union[str, Path] = None):
        """
        Initializes the processor.

        Args:
            spark: An active SparkSession configured for Delta Lake.
            input_path: Path to the input Delta table (e.g., './data/formatted/lloguer_catalunya').
            output_path: Path for the output Delta Lake table (e.g., './data/trusted/lloguer_catalunya').
        """
        self.spark = spark
        self.input_path = str(input_path) if input_path else "./data/formatted/lloguer_catalunya"
        self.output_path = str(output_path) if output_path else "./data/trusted/lloguer_catalunya"
        # Schema is expected to be the same as the formatted output
        print(f"Lloguer Trusted Zone Initialized.")
        print(f"  Input Formatted Path: {self.input_path}")
        print(f"  Output Trusted Path: {self.output_path}")

    def _inspect_output_delta(self):
        """Reads the output Trusted Delta table and prints schema and sample data."""
        print(f"--- Inspecting Output Trusted Delta Table: {self.output_path} ---")
        try:
            delta_log_path = Path(self.output_path) / "_delta_log"
            if not delta_log_path.is_dir():
                 print(f"Inspection failed: Delta log not found at {delta_log_path}. Table might not exist or is corrupted.")
                 return

            df_read_back = self.spark.read.format("delta").load(self.output_path)
            count = df_read_back.count()
            print(f"Successfully read back {count} records from Trusted Delta table.")

            print("\nSchema in Trusted Delta Table:")
            df_read_back.printSchema()

            print(f"\nSample Data (First 5 rows) from Trusted Delta Table:")
            df_read_back.show(5, truncate=False)

        except Exception as e:
             print(f"!!! Inspection failed: Error reading or processing Trusted Delta table: {e}")
             import traceback
             traceback.print_exc()
        print(f"--- Finished Inspecting Output Trusted Delta Table ---")

    def run(self, verbose: bool = False):
        """
        Executes the Trusted Zone processing for the Lloguer data.

        Args:
            verbose (bool): If True, inspect the output Delta table after writing.
                           Defaults to False.
        """
        print(f"--- Running Lloguer Trusted Zone Task (Verbose: {verbose}) ---")
        write_successful = False
        try:
            # 1. Read Formatted Data
            print(f"Reading formatted data from Delta table: {self.input_path}")
            df_formatted = self.spark.read.format("delta").load(self.input_path)
            initial_count = df_formatted.count()
            if initial_count == 0:
                print("Input data is empty. Skipping processing.")
                return # Exit early if no data
            print(f"Initial row count from Formatted Zone: {initial_count}")
            print("Schema read from Formatted Zone:")
            df_formatted.printSchema()

            # 2. Apply Data Quality Rules (Filtering based on Denial Constraints)
            df_filtered = df_formatted # Start with the original formatted data
            current_count = initial_count

            # Rule 1: Filter Nulls in Key/Essential Columns
            key_cols_not_null = [
                "ambit_territorial", "codi_territorial", "nom_territori",
                "any", "periode", "tram_preus"
            ]
            filter_condition_nulls = F.lit(True)
            for col_name in key_cols_not_null:
                 filter_condition_nulls = filter_condition_nulls & F.col(col_name).isNotNull()
            df_filtered = df_filtered.filter(filter_condition_nulls)
            count_after_null_keys = df_filtered.count()
            print(f"Rows after filtering NULLs in essential columns ({key_cols_not_null}): {count_after_null_keys} (Removed: {current_count - count_after_null_keys})")
            current_count = count_after_null_keys

            # Rule 2: Validate Year Range
            min_year = 2005
            max_year = datetime.datetime.now().year + 1
            df_filtered = df_filtered.filter(F.col("any").between(min_year, max_year))
            count_after_year_check = df_filtered.count()
            print(f"Rows after year range check ({min_year}-{max_year}): {count_after_year_check} (Removed: {current_count - count_after_year_check})")
            current_count = count_after_year_check

            # Rule 3: Validate 'habitatges' (count) >= 0
            df_filtered = df_filtered.filter(F.col("habitatges") >= 0)
            count_after_habitatges_check = df_filtered.count()
            print(f"Rows after 'habitatges' >= 0 check: {count_after_habitatges_check} (Removed: {current_count - count_after_habitatges_check})")
            current_count = count_after_habitatges_check

            # Rule 4: Validate 'renda' > 0 (if not NULL)
            df_filtered = df_filtered.filter(F.col("renda").isNull() | (F.col("renda") > 0))
            count_after_renda_check = df_filtered.count()
            print(f"Rows after 'renda' > 0 (or NULL) check: {count_after_renda_check} (Removed: {current_count - count_after_renda_check})")
            current_count = count_after_renda_check # This is the count before deduplication

            # --- Check for Duplicates with Value Differences (Step 3) ---
            key_columns_dedup = ["ambit_territorial", "nom_territori", "codi_territorial", "any", "periode", "tram_preus"]
            print("\n--- Checking for duplicate keys with differing values ---")
            print(f"Using key columns: {key_columns_dedup}")

            # Group by the keys and aggregate counts and distinct counts of values
            duplicate_check_df = df_filtered.groupBy(key_columns_dedup).agg(
                F.count("*").alias("num_rows_per_key"),
                F.countDistinct("habitatges").alias("distinct_habitatges"),
                F.countDistinct("renda").alias("distinct_renda")
            )

            # Filter for keys that appear more than once AND have different values for habitatges OR renda
            conflicting_keys_df = duplicate_check_df.filter(
                (F.col("num_rows_per_key") > 1) &
                ((F.col("distinct_habitatges") > 1) | (F.col("distinct_renda") > 1))
            )

            conflicting_keys_count = conflicting_keys_df.count()

            if conflicting_keys_count > 0:
                print(f"\nWARNING: Found {conflicting_keys_count} key combinations with duplicate entries but differing 'habitatges' or 'renda' values.")
                print("This indicates that the current dropDuplicates step will discard potentially meaningful variations.")
                print("Showing up to 10 examples of conflicting keys and their value counts:")
                conflicting_keys_df.show(10, truncate=False)

                # Optional: Show the actual conflicting rows for more detail
                if verbose: # Only show full rows if verbose is on
                    print("\nShowing some full rows associated with these conflicting keys (up to 20 rows):")
                    # Join back to the filtered data to get all columns for the conflicting keys
                    conflicting_rows_to_show = df_filtered.join(
                        F.broadcast(conflicting_keys_df.select(*key_columns_dedup)), # Broadcast small df
                        on=key_columns_dedup,
                        how="inner"
                    ).orderBy(*key_columns_dedup, "habitatges", "renda") # Order to see differences easily
                    conflicting_rows_to_show.show(20, truncate=False)
            else:
                print("No duplicate keys found with differing 'habitatges' or 'renda' values. Proceeding with standard deduplication.")

            # 4. Deduplication (Still applying the original strategy for now)
            print(f"\nApplying deduplication based on columns: {key_columns_dedup}")
            # Note: This step WILL arbitrarily remove rows if conflicting keys were found above.
            # Consider aggregation or window functions if those variations need to be handled differently.
            df_deduplicated = df_filtered.dropDuplicates(key_columns_dedup)
            count_after_dedup = df_deduplicated.count()
            print(f"Rows after deduplication: {count_after_dedup} (Removed: {current_count - count_after_dedup})")

            # 5. Select columns in the correct order
            df_final = df_deduplicated.select(df_formatted.columns)

            # 6. Write Cleaned Data to Trusted Zone
            print(f"\nWriting cleaned data ({count_after_dedup} rows) to Trusted Delta table: {self.output_path}")
            df_final.write.format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(self.output_path)
            write_successful = True
            print(f"Trusted Lloguer data successfully written to {self.output_path}")

            # 7. Inspect if requested and write was successful
            if verbose and write_successful:
                self._inspect_output_delta()

            print("--- Lloguer Trusted Zone Task Successfully Completed ---")

        except Exception as e:
            print(f"!!! Error during Lloguer Trusted Zone execution: {e}")
            import traceback
            traceback.print_exc()
            print("--- Lloguer Trusted Zone Task Failed ---")


# --- Spark Session Creation Helper (Identical) ---
def get_spark_session() -> SparkSession:
    """Initializes and returns a SparkSession configured for Delta Lake."""
    print("Initializing Spark Session...")
    # ... (rest of the function is identical, no changes needed)
    try:
        spark = SparkSession.builder \
            .appName("LloguerTrustedZone") \
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

# --- Main Execution Block (Identical) ---
if __name__ == "__main__":
    # ... (main block is identical, no changes needed)
    INPUT_DELTA_FORMATTED = "./data/formatted/lloguer_catalunya"
    OUTPUT_DELTA_TRUSTED = "./data/trusted/lloguer_catalunya"
    Path(OUTPUT_DELTA_TRUSTED).parent.mkdir(parents=True, exist_ok=True)
    spark = None
    try:
        spark = get_spark_session()
        formatted_log_path = Path(INPUT_DELTA_FORMATTED) / "_delta_log"
        if not formatted_log_path.is_dir():
            print(f"Error: Input Formatted Delta table log not found at {formatted_log_path}")
            print("Please run the Lloguer formatted zone script first.")
        else:
            truster = LloguerTrustedZone(spark=spark, input_path=INPUT_DELTA_FORMATTED, output_path=OUTPUT_DELTA_TRUSTED)
            truster.run(verbose=True) # Keep verbose=True to see the check results
    except Exception as main_error:
        print(f"An unexpected error occurred in the main execution block: {main_error}")
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            print("Stopping Spark Session.")
            spark.stop()