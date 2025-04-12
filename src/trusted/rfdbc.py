from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# --- Delta Lake Package Configuration ---
DELTA_PACKAGE = "io.delta:delta-spark_2.12:3.3.0" # Or your desired Delta version

class RFDBCTrustedZone:
    """
    Applies cleaning and transformation rules to data from the Formatted Zone
    and saves the result to a Trusted Zone Delta table.

    Processing steps separated into methods:
    - remove_columns: Removes specific columns.
    - modify_columns: Renames columns and applies modifications (e.g., municipality code).
    - remove_rows: Handles duplicates and missing value strategies.
    """

    def __init__(self, spark: SparkSession, input_path: str, output_path: str):
        self.spark = spark
        self.input_path = input_path
        self.output_path = output_path
        self.initial_df = self._load_data()
        print(f"RFDBCTrustedZone Initialized. Input: {input_path}, Output: {output_path}")

    def _load_data(self) -> DataFrame:
        """Loads the input Delta table."""
        print(f"Loading data from Delta table: {self.input_path}")
        try:
            df = self.spark.read.format("delta").load(self.input_path)
            print(f"Successfully loaded data with {df.count()} rows.")
            print("Initial Schema:")
            df.printSchema()
            return df
        except Exception as e:
            print(f"ERROR: Failed to load data from {self.input_path}. Error: {e}")
            raise RuntimeError(f"Could not load input data from {self.input_path}") from e

    def remove_columns(self, df: DataFrame) -> DataFrame:
        """Removes specified columns."""
        print("\n--- Step: Removing Columns ---")
        cols_to_drop = ["concept_code", "concept_label", "year_code"]

        # Check if columns exist before attempting to drop
        actual_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if len(actual_cols_to_drop) < len(cols_to_drop):
            missing = set(cols_to_drop) - set(actual_cols_to_drop)
            print(f"Warning: Columns to drop not found: {missing}")

        if not actual_cols_to_drop:
             print("No columns to drop found or specified.")
             return df # Return original df if nothing to drop

        df_dropped = df.drop(*actual_cols_to_drop)
        print(f"Columns dropped: {actual_cols_to_drop}")
        print("Schema after column removal:")
        df_dropped.printSchema()
        return df_dropped

    def modify_columns(self, df: DataFrame) -> DataFrame:
        """Renames specified columns and applies modifications."""
        print("\n--- Step: Modifying Columns (Rename & Adjustments) ---")
        df_modified = df

        # --- 1. Rename Columns ---
        col_to_rename = "year_label"
        new_name = "year"

        if col_to_rename in df_modified.columns:
            df_modified = df_modified.withColumnRenamed(col_to_rename, new_name)
            print(f"Renamed column '{col_to_rename}' to '{new_name}'.")
        else:
            print(f"Warning: Column to rename '{col_to_rename}' not found.")

        # --- 2. Modify Municipality Code/ID ---
        # Assuming the column name is 'municipality_id' from the previous steps
        mun_col = "municipality_id"
        if mun_col in df_modified.columns:
            print(f"Modifying column '{mun_col}': Removing last digit.")
            # Use substring: start at position 1, length is original length - 1
            # Ensure the column is treated as string for length check
            df_modified = df_modified.withColumn(
                mun_col,
                F.expr(f"substring({mun_col}, 1, length(cast({mun_col} as string)) - 1)")
            )
            # Optional: You might want to cast back to original type if needed,
            # but removing a digit likely keeps it as a string effectively.
            print(f"Applied modification to '{mun_col}'.")
        else:
            print(f"Warning: Column '{mun_col}' not found for modification.")

        print("Schema after column modifications:")
        df_modified.printSchema()
        # Optional: Show sample data to verify modification
        # print("Sample data after modification:")
        # df_modified.select(mun_col).distinct().show(5)
        return df_modified

    def remove_rows(self, df: DataFrame) -> DataFrame:
        """
        Removes duplicate rows and handles missing values according to the defined strategy.
        (Logic remains the same as the previous version)
        1. Removes exact duplicates.
        2. Finds municipalities associated with any missing data and removes all their rows.
        3. Removes any remaining rows with missing data.
        """
        print("\n--- Step: Removing Rows (Duplicates & Missing Value Strategy) ---")

        # --- 1. Remove Exact Duplicates ---
        initial_count = df.count()
        print(f"Starting row removal with {initial_count} rows.")
        df_deduplicated = df.dropDuplicates()
        deduplicated_count = df_deduplicated.count()
        duplicates_removed = initial_count - deduplicated_count
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows.")
        else:
            print("No duplicate rows found.")
        print(f"Rows after deduplication: {deduplicated_count}")

        # Cache potentially for missing value checks
        df_deduplicated.cache()

        # --- 2. Identify and Remove Rows for Municipalities with Missing Data ---
        print("\nIdentifying municipalities associated with missing data...")
        municipality_id_col = "municipality_id" # This should match the name *after* modification step

        if municipality_id_col not in df_deduplicated.columns:
            print(f"Warning: Column '{municipality_id_col}' not found. Skipping municipality-based removal.")
            df_filtered_mun = df_deduplicated # Proceed without this step
        else:
            # Build filter expression for rows with *any* missing value
            missing_filter_expr = None
            for c in df_deduplicated.columns:
                col_check = F.col(c).isNull()
                if dict(df_deduplicated.dtypes)[c] in ('double', 'float'):
                     col_check = col_check | F.isnan(F.col(c))
                if missing_filter_expr is None:
                    missing_filter_expr = col_check
                else:
                    missing_filter_expr = missing_filter_expr | col_check

            missing_rows_df = df_deduplicated.filter(missing_filter_expr)
            missing_rows_count = missing_rows_df.count()

            if missing_rows_count > 0:
                print(f"Found {missing_rows_count} rows containing at least one missing value.")
                # Get distinct municipality IDs from these rows
                # Note: these IDs will be the *modified* ones (last digit removed)
                mun_ids_with_missing = missing_rows_df.select(municipality_id_col).distinct()
                mun_ids_list = [row[municipality_id_col] for row in mun_ids_with_missing.collect()]

                if mun_ids_list:
                    print(f"Identified {len(mun_ids_list)} modified municipality IDs associated with missing data: {mun_ids_list}")
                    print(f"Removing ALL rows for these modified IDs...")
                    # Filter out rows where the *modified* municipality_id is in the collected list
                    df_filtered_mun = df_deduplicated.filter(~F.col(municipality_id_col).isin(mun_ids_list))
                    filtered_mun_count = df_filtered_mun.count()
                    mun_rows_removed = deduplicated_count - filtered_mun_count
                    print(f"Removed {mun_rows_removed} rows belonging to identified modified IDs.")
                    print(f"Rows after municipality-based removal: {filtered_mun_count}")
                else:
                    print("No specific municipality IDs found associated with the missing rows.")
                    df_filtered_mun = df_deduplicated
            else:
                print("No rows with missing values found. Skipping municipality-based removal.")
                df_filtered_mun = df_deduplicated

        # --- 3. Remove Any Other Rows with Missing Values ---
        print("\nRemoving any remaining rows with missing values...")
        count_before_final_na_drop = df_filtered_mun.count()
        df_cleaned = df_filtered_mun.dropna(how='any')
        final_count = df_cleaned.count()
        final_na_removed = count_before_final_na_drop - final_count
        if final_na_removed > 0:
             print(f"Removed {final_na_removed} additional rows containing missing values.")
        else:
             print("No further rows with missing values found to remove.")

        print(f"Final row count after all cleaning: {final_count}")
        df_deduplicated.unpersist()
        return df_cleaned

    def run(self):
        """Executes the full Trusted Zone processing pipeline."""
        print(f"\n--- Running RFDBC Trusted Zone Processing ---")
        if self.initial_df is None:
            print("ERROR: Initial DataFrame not loaded. Aborting.")
            return

        try:
            # Step 1: Remove Columns
            df_cols_removed = self.remove_columns(self.initial_df)

            # Step 2: Modify Columns (Rename, Adjustments)
            df_cols_modified = self.modify_columns(df_cols_removed)

            # Step 3: Remove Rows (Duplicates & Missing Handling)
            df_cleaned = self.remove_rows(df_cols_modified)

            # Step 4: Final Inspection (Optional but recommended)
            print("\n--- Final Inspection Before Save ---")
            print("Final Schema:")
            df_cleaned.printSchema()
            print(f"Final Row Count: {df_cleaned.count()}")
            print("Sample of final data (first 10 rows):")
            df_cleaned.show(10, truncate=False)

            # Step 5: Save to Trusted Zone Delta Lake
            print(f"\nSaving cleaned data to Delta table: {self.output_path}")
            df_cleaned.write.format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(self.output_path)
            print(f"Successfully saved cleaned data to {self.output_path}")
            print("--- RFDBC Trusted Zone Processing Successfully Completed ---")

        except Exception as e:
            print(f"!!! ERROR during Trusted Zone processing: {e}")
            import traceback
            traceback.print_exc()
            print("--- RFDBC Trusted Zone Processing Failed ---")


# --- Spark Session Creation Helper ---
def get_spark_session() -> SparkSession:
    """Initializes and returns a SparkSession configured for Delta Lake."""
    print("Initializing Spark Session...")
    try:
        spark = SparkSession.builder \
            .appName("RFDBCTrustedZone") \
            .master("local[*]") \
            .config("spark.jars.packages", DELTA_PACKAGE) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.databricks.delta.schema.autoMerge.enabled", "true") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        print("Spark Session Initialized.")
        return spark
    except Exception as e:
        print(f"FATAL: Error initializing Spark Session: {e}")
        raise

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define paths
    INPUT_DELTA_PATH = "./data/formatted/rfdbc_data" # Assumes output from RFDBCFormattedZone
    OUTPUT_DELTA_PATH = "./data/trusted/rfdbc_data" # Use different path to avoid conflicts

    spark = None
    try:
        spark = get_spark_session()

        # Instantiate and run the Trusted Zone processor
        processor = RFDBCTrustedZone(
            spark=spark,
            input_path=INPUT_DELTA_PATH,
            output_path=OUTPUT_DELTA_PATH
        )
        processor.run()

        # --- Optional: Verification Step ---
        print("\n--- Verifying Output Delta Table Contents ---")
        try:
            df_read = spark.read.format("delta").load(OUTPUT_DELTA_PATH)
            print(f"Successfully read final Delta table from {OUTPUT_DELTA_PATH}")
            print("Final Schema:")
            df_read.printSchema()
            print("Final Data:")
            df_read.show(truncate=False)
        except Exception as read_e:
            print(f"Warning: Error reading back final Delta table: {read_e}")
        # ------------------------------------

    except Exception as main_error:
        print(f"An error occurred in the main execution block: {main_error}")
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            print("\nStopping Spark Session.")
            spark.stop()
            print("Spark Session stopped.")