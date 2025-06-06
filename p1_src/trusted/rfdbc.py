from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType # Import IntegerType for casting

class RFDBCTrustedZone:
    """
    Applies cleaning and transformation rules to data from the Formatted Zone
    and saves the result to a Trusted Zone Delta table.

    Processing steps separated into methods:
    - remove_columns: Removes specific columns.
    - modify_columns: Renames columns and applies modifications (e.g., municipality code).
    - remove_rows: Removes duplicates and enforces denial constraints (data quality rules).
    """

    def __init__(self, spark: SparkSession, input_path: str = "./data/formatted/rfdbc", output_path: str = "./data/trusted/rfdbc"):
        self.spark = spark
        self.input_path = input_path
        self.output_path = output_path
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
        actual_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if len(actual_cols_to_drop) < len(cols_to_drop):
            missing = set(cols_to_drop) - set(actual_cols_to_drop)
            print(f"Warning: Columns to drop not found: {missing}")
        if not actual_cols_to_drop:
             print("No columns to drop found or specified.")
             return df
        df_dropped = df.drop(*actual_cols_to_drop)
        print(f"Columns dropped: {actual_cols_to_drop}")
        print("Schema after column removal:")
        df_dropped.printSchema()
        return df_dropped

    def modify_columns(self, df: DataFrame) -> DataFrame:
        """Renames specified columns and applies modifications."""
        print("\n--- Step: Modifying Columns (Rename & Adjustments) ---")
        df_modified = df

        # Rename 'year_label' to 'year'
        col_to_rename = "year_label"
        new_name = "year"
        if col_to_rename in df_modified.columns:
            df_modified = df_modified.withColumnRenamed(col_to_rename, new_name)
            print(f"Renamed column '{col_to_rename}' to '{new_name}'.")
        else:
            print(f"Warning: Column to rename '{col_to_rename}' not found.")

        # Modify 'municipality_id' - remove last digit to get 5-digit ID
        mun_col = "municipality_id"
        if mun_col in df_modified.columns:
            print(f"Modifying column '{mun_col}': Removing last digit.")
            df_modified = df_modified.withColumn(
                mun_col,
                F.expr(f"substring(cast({mun_col} as string), 1, length(cast({mun_col} as string)) - 1)")
            )
            print(f"Applied modification to '{mun_col}'.")
        else:
            print(f"Warning: Column '{mun_col}' not found for modification.")

        print("Schema after column modifications:")
        df_modified.printSchema()
        return df_modified

    def remove_rows(self, df: DataFrame) -> DataFrame:
        """
        Removes duplicate rows and filters data based on denial constraints.
        1. Removes exact duplicates.
        2. Filters based on year range [2000, 2025].
        3. Filters based on modified municipality_id length (must be 5).
        4. Filters based on value being positive (> 0).
        """
        print("\n--- Step: Removing Rows (Duplicates & Denial Constraints) ---")

        # --- 1. Remove Exact Duplicates ---
        initial_count = df.count()
        print(f"Starting row removal with {initial_count} rows.")
        df_deduplicated = df.dropDuplicates()
        deduplicated_count = df_deduplicated.count()
        duplicates_removed = initial_count - deduplicated_count
        if duplicates_removed > 0:
            print(f"- Removed {duplicates_removed} duplicate rows.")
        else:
            print("- No duplicate rows found.")
        print(f"  Rows remaining: {deduplicated_count}")

        # Cache after deduplication as multiple filters will be applied
        df_deduplicated.cache()
        current_df = df_deduplicated
        rows_before_constraints = deduplicated_count

        # --- 2. Apply Denial Constraint: Year Range [2000, 2025] ---
        year_col = "year"
        if year_col in current_df.columns:
            print(f"\nApplying constraint: '{year_col}' between 2000 and 2025 (inclusive).")
            # Ensure year is treated as integer for comparison
            year_condition = (F.col(year_col).cast(IntegerType()) >= 2000) & \
                             (F.col(year_col).cast(IntegerType()) <= 2025)
            df_filtered = current_df.filter(year_condition)
            count_after_filter = df_filtered.count()
            rows_removed = rows_before_constraints - count_after_filter
            print(f"- Removed {rows_removed} rows violating year constraint.")
            print(f"  Rows remaining: {count_after_filter}")
            current_df = df_filtered
            rows_before_constraints = count_after_filter
        else:
             print(f"Warning: Column '{year_col}' not found. Skipping year constraint.")

        # --- 3. Apply Denial Constraint: Municipality ID Length == 5 ---
        mun_col = "municipality_id"
        if mun_col in current_df.columns:
            print(f"\nApplying constraint: length of '{mun_col}' must be 5.")
            # Length function works on strings
            mun_len_condition = F.length(F.col(mun_col)) == 5
            df_filtered = current_df.filter(mun_len_condition)
            count_after_filter = df_filtered.count()
            rows_removed = rows_before_constraints - count_after_filter
            print(f"- Removed {rows_removed} rows violating municipality ID length constraint.")
            print(f"  Rows remaining: {count_after_filter}")
            current_df = df_filtered
            rows_before_constraints = count_after_filter
        else:
             print(f"Warning: Column '{mun_col}' not found. Skipping municipality ID length constraint.")

        # --- 4. Apply Denial Constraint: Value > 0 ---
        value_col = "value"
        if value_col in current_df.columns:
             print(f"\nApplying constraint: '{value_col}' must be positive (> 0).")
             # Direct comparison works for numeric types. Nulls will be filtered out.
             value_condition = F.col(value_col) > 0
             df_filtered = current_df.filter(value_condition)
             count_after_filter = df_filtered.count()
             rows_removed = rows_before_constraints - count_after_filter
             print(f"- Removed {rows_removed} rows violating positive value constraint.")
             print(f"  Rows remaining: {count_after_filter}")
             current_df = df_filtered
             rows_before_constraints = count_after_filter
        else:
             print(f"Warning: Column '{value_col}' not found. Skipping positive value constraint.")


        print(f"\nFinal row count after all constraints: {rows_before_constraints}")

        # Unpersist the cached DataFrame
        df_deduplicated.unpersist()

        return current_df # Return the final filtered DataFrame

    def run(self):
        """Executes the full Trusted Zone processing pipeline."""
        print(f"\n--- Running RFDBC Trusted Zone Processing ---")
        self.initial_df = self._load_data()

        try:
            # Step 1: Remove Columns
            df_cols_removed = self.remove_columns(self.initial_df)

            # Step 2: Modify Columns (Rename, Adjustments)
            df_cols_modified = self.modify_columns(df_cols_removed)

            # Step 3: Remove Rows (Duplicates & Denial Constraints)
            df_cleaned = self.remove_rows(df_cols_modified)

            # Step 4: Final Inspection
            print("\n--- Final Inspection Before Save ---")
            final_count = df_cleaned.count() # Recalculate count for final confirmation
            print("Final Schema:")
            df_cleaned.printSchema()
            print(f"Final Row Count: {final_count}")
            if final_count > 0:
                print("Sample of final data (first 10 rows):")
                df_cleaned.show(10, truncate=False)
            else:
                print("Final DataFrame is empty.")

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


# --- Main Execution Block ---
if __name__ == "__main__":
    from spark_session import get_spark_session

    spark = None
    try:
        spark = get_spark_session()

        processor = RFDBCTrustedZone(
            spark=spark
        )
        processor.run()

    except Exception as main_error:
        print(f"An error occurred in the main execution block: {main_error}")
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            print("\nStopping Spark Session.")
            spark.stop()
            print("Spark Session stopped.")