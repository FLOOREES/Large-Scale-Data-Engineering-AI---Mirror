# --- PySpark Imports ---
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ---------------------

# --- Delta Lake Package Configuration ---
DELTA_PACKAGE = "io.delta:delta-spark_2.12:3.3.0" # Example: Use 2.4.0 for Spark 3.3

class IdescatTrustedZone:
    """
    Cleans and validates the formatted Idescat indicators data,
    applying quality rules and deduplication before saving to the Trusted Zone.
    """

    def __init__(self, spark: SparkSession, input_path: str, output_path: str):
        """
        Initializes the processor.

        Args:
            spark: An active SparkSession configured for Delta Lake.
            input_path: Path to the input Delta table (e.g., './data/formatted/idescat_indicators').
            output_path: Path for the output Delta Lake table (e.g., './data/trusted/idescat_indicators').
        """
        self.spark = spark
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        # We expect the schema to be the same as the formatted output,
        # but we might read it dynamically or redefine if needed for clarity.
        print(f"Trusted Zone Initialized. Input: {self.input_path}, Output: {self.output_path}")

    def run(self):
        """Executes the Trusted Zone processing for the Idescat data."""
        print(f"--- Running Idescat Trusted Zone Task ---")
        try:
            # 1. Read Formatted Data
            print(f"Reading formatted data from Delta table: {self.input_path}")
            df_formatted = self.spark.read.format("delta").load(self.input_path)
            initial_count = df_formatted.count()
            print(f"Initial row count: {initial_count}")
            df_formatted.printSchema() # Verify schema

            # 2. Apply Data Quality Rules (Filtering)

            # Rule 1: Enforce non-null primary keys (Denial Constraint: Key columns cannot be NULL)
            df_filtered = df_formatted.filter(
                F.col("municipality_id").isNotNull() &
                F.col("indicator_id").isNotNull() &
                F.col("reference_year").isNotNull()
            )
            count_after_null_keys = df_filtered.count()
            print(f"Rows after filtering null keys: {count_after_null_keys} (Removed: {initial_count - count_after_null_keys})")

            # Rule 2: Validate reference_year range (Denial Constraint: Year must be realistic)
            min_year = 2000
            max_year = 2025 # Adjust as needed
            df_filtered = df_filtered.filter(
                (F.col("reference_year") >= min_year) & (F.col("reference_year") <= max_year)
            )
            count_after_year_check = df_filtered.count()
            print(f"Rows after year range check ({min_year}-{max_year}): {count_after_year_check} (Removed: {count_after_null_keys - count_after_year_check})")

            # Rule 3: Validate specific indicator values (Example Denial Constraints)
            # Define non-negative indicators from your CHOSEN_INDICATORS list in landing zone
            non_negative_indicators = [
                'f171', 'f36', 'f42', 'f187', 'f183', # Population counts/births
                'f261', 'f262',                       # Surface, Density
                'f308',                               # Unemployment (assuming count >= 0)
                'f191', 'f270', 'f293', 'f294', 'f301' # Housing, Libraries, Facilities counts
            ]
            # Define Lat/Lon bounds for Catalonia (approximate)
            min_lon, max_lon = 0.0, 3.5
            min_lat, max_lat = 40.5, 43.0

            # Apply conditions using 'when'. We filter out rows where the condition is FALSE.
            # Keep rows that don't match the indicator ID OR that match and satisfy the condition.
            df_filtered = df_filtered.filter(
                # Check non-negative indicators
                (~F.col("indicator_id").isin(non_negative_indicators)) |
                (F.col("indicator_id").isin(non_negative_indicators) & (F.col("municipality_value") >= 0))
            ).filter(
                # Check Longitude bounds
                (F.col("indicator_id") != 'f328') | # Keep if not Longitude
                ((F.col("indicator_id") == 'f328') & F.col("municipality_value").between(min_lon, max_lon)) # Or if Longitude and within bounds
            ).filter(
                # Check Latitude bounds
                (F.col("indicator_id") != 'f329') | # Keep if not Latitude
                ((F.col("indicator_id") == 'f329') & F.col("municipality_value").between(min_lat, max_lat)) # Or if Latitude and within bounds
            )
            # Note: This doesn't explicitly handle NULL municipality_value for non-negative checks.
            # If >= 0 comparison evaluates to null for null input, the row might be kept or dropped depending on Spark version/config.
            # Explicitly checking F.col("municipality_value").isNull() might be needed if specific null handling is required.
            # For simplicity here, we assume filtering based on the condition is sufficient.

            count_after_value_checks = df_filtered.count()
            print(f"Rows after value range checks: {count_after_value_checks} (Removed: {count_after_year_check - count_after_value_checks})")

            # 3. Deduplication (Based on logical primary key for this long table)
            key_columns = ["municipality_id", "indicator_id", "reference_year"]
            df_deduplicated = df_filtered.dropDuplicates(key_columns)
            count_after_dedup = df_deduplicated.count()
            print(f"Rows after deduplication on {key_columns}: {count_after_dedup} (Removed: {count_after_value_checks - count_after_dedup})")

            # 4. Write Cleaned Data to Trusted Zone (Delta Lake)
            print(f"Writing cleaned data to Delta table: {self.output_path}")
            # Ensure the schema hasn't changed unexpectedly
            df_final = df_deduplicated.select(df_formatted.columns) # Select in original order

            df_final.write.format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(self.output_path)

            print(f"Trusted data successfully written to {self.output_path}")
            print("--- Idescat Trusted Zone Task Successfully Completed ---")

        except Exception as e:
            print(f"!!! Error during Trusted Zone execution: {e}")
            import traceback
            traceback.print_exc()
            print("--- Idescat Trusted Zone Task Failed ---")


# --- Spark Session Creation Helper (Identical to Formatted Zone) ---
def get_spark_session() -> SparkSession:
    """
    Initializes and returns a SparkSession configured for Delta Lake
    and updated Parquet compatibility settings.
    """
    print("Initializing Spark Session...")
    try:
        spark = SparkSession.builder \
            .appName("IdescatTrustedZone") \
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
    # Define paths for direct execution
    INPUT_DELTA_FORMATTED = "./data/formatted/idescat" # Read from Formatted
    OUTPUT_DELTA_TRUSTED = "./data/trusted/idescat"   # Write to Trusted

    spark = None
    try:
        spark = get_spark_session()
        # Instantiate and run the processor
        truster = IdescatTrustedZone(spark=spark, input_path=INPUT_DELTA_FORMATTED, output_path=OUTPUT_DELTA_TRUSTED)
        truster.run()
    except Exception as main_error:
        print(f"An error occurred outside the run method: {main_error}")
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            print("Stopping Spark Session.")
            spark.stop()