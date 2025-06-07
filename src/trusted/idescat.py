from pyspark.sql import SparkSession
from pyspark.sql import functions as F

class IdescatTrustedZone:
    """
    Cleans and validates the formatted Idescat indicators data,
    applying quality rules and deduplication before saving to the Trusted Zone.
    """

    def __init__(self, spark: SparkSession, input_path: str = "./data/formatted/idescat", output_path: str = "./data/trusted/idescat"):
        """
        Initializes the processor.

        Args:
            spark: An active SparkSession configured for Delta Lake.
            input_path: Path to the input Delta table (e.g., './data/formatted/idescat').
            output_path: Path for the output Delta Lake table (e.g., './data/trusted/idescat').
        """
        self.spark = spark
        self.input_path = str(input_path)
        self.output_path = str(output_path)
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

            # Rule 3: Apply comprehensive validation rules for all relevant indicators
            print("\nApplying comprehensive validation rules for indicator values...")
            count_before_value_checks = df_filtered.count()

            # Define plausible value ranges for each indicator.
            # This single 'when' statement replaces the multiple filters.
            # If an indicator is not listed here, it is not checked and automatically kept.
            df_validated = df_filtered.filter(
                F.when(F.col("indicator_id") == 'f171', F.col("municipality_value").between(0, 2_000_000))   # Population
                .when(F.col("indicator_id") == 'f36',  F.col("municipality_value").between(0, 1_000_000))   # Men
                .when(F.col("indicator_id") == 'f42',  F.col("municipality_value").between(0, 1_000_000))   # Women
                .when(F.col("indicator_id") == 'f187', F.col("municipality_value").between(0, 50_000))      # Births
                .when(F.col("indicator_id") == 'f183', F.col("municipality_value").between(0, 2_000_000))   # Spanish Nationality Pop
                .when(F.col("indicator_id") == 'f261', F.col("municipality_value").between(0, 500))         # Surface area (km²)
                .when(F.col("indicator_id") == 'f262', F.col("municipality_value").between(0, 50_000))      # Density (Pop/km²)
                .when(F.col("indicator_id") == 'f328', F.col("municipality_value").between(0.0, 3.5))       # Longitude
                .when(F.col("indicator_id") == 'f329', F.col("municipality_value").between(40.5, 43.0))     # Latitude
                .when(F.col("indicator_id") == 'f308', F.col("municipality_value").between(0, 500_000))     # Total Unemployment
                .when(F.col("indicator_id") == 'f191', F.col("municipality_value").between(0, 1_000_000))   # Dwellings
                .when(F.col("indicator_id") == 'f270', F.col("municipality_value").between(0, 1_000))       # Libraries
                .when(F.col("indicator_id") == 'f293', F.col("municipality_value").between(0, 1_000))       # Pavilions
                .when(F.col("indicator_id") == 'f294', F.col("municipality_value").between(0, 1_000))       # Sports Courts
                .when(F.col("indicator_id") == 'f301', F.col("municipality_value").between(0, 1_000))       # Indoor Pools
                .otherwise(True) # IMPORTANT: Keep all rows for indicators that don't have a specific rule
            )

            count_after_value_checks = df_validated.count()
            rows_removed = count_before_value_checks - count_after_value_checks
            print(f"Rows after all value range checks: {count_after_value_checks} (Removed: {rows_removed})")


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

# --- Main Execution Block ---
if __name__ == "__main__":
    from spark_session import get_spark_session

    spark = None
    try:
        spark = get_spark_session()
        truster = IdescatTrustedZone(spark=spark)
        truster.run()
    except Exception as main_error:
        print(f"An error occurred outside the run method: {main_error}")
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            print("Stopping Spark Session.")
            spark.stop()