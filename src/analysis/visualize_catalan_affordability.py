# visualize_catalan_affordability_multiyear.py

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import traceback
from typing import Optional, List

# --- PySpark Imports ---
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
# ---------------------

# --- Configuration ---

# 1. Path to your Exploitation Zone Delta table
EXPLOITATION_DATA_PATH = "./data/exploitation/consolidated_municipal_annual"

# 2. Path to your downloaded Shapefile (specifically the .shp file)
SHAPEFILE_PATH = Path("./geospatial/MUC_TM.shp")

# 3. Column name in Shapefile containing the 5-digit municipality code
MUN_CODE_COLUMN_IN_SHAPEFILE = "CODI_INE" # Confirmed from your previous steps

# 4. Affordability Calculation Columns
RENT_COLUMN = "avg_monthly_rent_eur"
INCOME_COLUMN = "salary_per_capita_eur" # Using salary as the best available income proxy

# 5. Continuous Color Mapping Configuration
COLOR_MAP = 'RdYlGn_r'  # Red-Yellow-Green (Reversed: Low=Green, High=Red)
COLORBAR_LABEL = "Affordability (% Monthly Rent * 12 / Annual Salary)"
VMIN = 10  # Minimum percentage for color scale (e.g., 10%)
VMAX = 50  # Maximum percentage for color scale (e.g., 50%)
MISSING_DATA_COLOR = 'lightgrey'
MISSING_DATA_LABEL = 'Missing Data'

# 7. Output directory for the map image
OUTPUT_DIR = Path("./data/analysis_output/maps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 8. Delta Lake Package Configuration (for SparkSession)
DELTA_PACKAGE = "io.delta:delta-spark_2.12:3.3.0" # Use version compatible with Spark 3.3

# --- Spark Session Creation Helper ---
def get_spark_session() -> SparkSession:
    """Initializes and returns a SparkSession configured for Delta Lake."""
    print("Initializing Spark Session...")
    try:
        spark = SparkSession.builder \
            .appName("AffordabilityMapMultiYear") \
            .master("local[*]") \
            .config("spark.jars.packages", DELTA_PACKAGE) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.sql.parquet.int96AsTimestamp", "true") \
            .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
            .getOrCreate()
        # Set log level after creation
        spark.sparkContext.setLogLevel("ERROR")
        print("Spark Session Initialized. Log level set to ERROR.")
        return spark
    except Exception as e:
        print(f"FATAL: Error initializing Spark Session: {e}")
        raise

# --- Data Processing and Plotting ---

def load_and_prepare_stats_data_spark(spark: SparkSession, exploitation_path: str) -> Optional[DataFrame]:
    """Loads exploitation data and selects necessary columns, returns Spark DF."""
    print(f"\n--- Loading and Preparing Statistical Data from Spark ---")
    try:
        print(f"Reading Delta table from: {exploitation_path}")
        exploitation_df = spark.read.format("delta").load(exploitation_path)
        print("Exploitation data loaded. Schema:")
        exploitation_df.printSchema()

        # Check if required columns exist
        required_cols = ["municipality_id", "any", RENT_COLUMN, INCOME_COLUMN, "municipality_name"]
        if not all(col in exploitation_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in exploitation_df.columns]
            print(f"ERROR: Exploitation data missing required columns: {missing}.")
            return None

        # Select necessary columns for all years
        stats_df_spark = exploitation_df.select(*required_cols) # Use list unpacking
        print("Required columns selected.")
        return stats_df_spark

    except Exception as e:
        print(f"ERROR loading or preparing statistical data from Spark: {e}")
        traceback.print_exc()
        return None

def identify_valid_years(df_spark: DataFrame) -> List[int]:
    """Identifies years with at least one municipality having non-null rent AND income."""
    print("\n--- Identifying valid years for plotting ---")
    try:
        valid_years_df = df_spark.filter(
            F.col(RENT_COLUMN).isNotNull() & F.col(INCOME_COLUMN).isNotNull()
        ).select("any").distinct()

        valid_years = sorted([row.any for row in valid_years_df.collect()])
        print(f"Found {len(valid_years)} year(s) with valid data: {valid_years}")
        return valid_years
    except Exception as e:
        print(f"ERROR identifying valid years: {e}")
        traceback.print_exc()
        return []


def calculate_affordability_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates rent-to-income ratio numerically."""
    df_calc = df.copy()
    # print("Calculating affordability ratios...") # Reduced verbosity
    df_calc['yearly_rent'] = df_calc[RENT_COLUMN] * 12
    df_calc['affordability_pct'] = np.where(
        (df_calc[INCOME_COLUMN].notna()) & (df_calc[INCOME_COLUMN] > 0) & (df_calc['yearly_rent'].notna()),
        (df_calc['yearly_rent'] / df_calc[INCOME_COLUMN]) * 100,
        np.nan # Assign NaN if income is missing/zero or rent is missing
    )
    # print("Affordability calculation complete.") # Reduced verbosity
    return df_calc

def plot_affordability_map_continuous(year: int, stats_data_for_year: pd.DataFrame, shapefile_gdf: gpd.GeoDataFrame):
    """Generates and saves the continuous color choropleth map for a given year."""
    print(f"\n--- Generating Map for Year {year} ---")

    # 1. Calculate affordability percentage (already done on filtered data)
    # stats_with_affordability = calculate_affordability_numeric(stats_data_for_year)
    # No need to call calculate_affordability_numeric again if called before merge

    # 2. Merge geospatial data with statistical data
    print(f"Merging data for year {year} with shapefile...")
    # --- Ensure merge keys are strings ---
    shapefile_gdf[MUN_CODE_COLUMN_IN_SHAPEFILE] = shapefile_gdf[MUN_CODE_COLUMN_IN_SHAPEFILE].astype(str)
    stats_data_for_year['municipality_id'] = stats_data_for_year['municipality_id'].astype(str)

    # Add affordability calculation here before merging
    stats_with_affordability = calculate_affordability_numeric(stats_data_for_year)


    merged_gdf = shapefile_gdf.merge(
        stats_with_affordability, # Use the data with calculated affordability
        left_on=MUN_CODE_COLUMN_IN_SHAPEFILE,
        right_on='municipality_id',
        how='left' # Keep all map polygons
    )
    print(f"Merge complete. Plotting {len(merged_gdf)} base municipalities.")
    # 'affordability_pct' will be NaN for municipalities not in stats_data or with missing rent/income

    # 3. Plotting
    print("Creating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    merged_gdf.plot(
        column='affordability_pct', # Plot the numerical percentage
        cmap=COLOR_MAP,            # Use the continuous colormap
        linewidth=0.4,
        ax=ax,
        edgecolor='0.6',
        legend=True,               # Add a colorbar legend
        legend_kwds={
            'label': COLORBAR_LABEL,
            'orientation': "vertical", # Or "horizontal"
            'shrink': 0.6 # Adjust size of colorbar if needed
            },
        missing_kwds={             # How to draw areas with missing data
            "color": MISSING_DATA_COLOR,
            #"hatch": "///",       # Optional hatching
            "edgecolor": "grey",
            "label": MISSING_DATA_LABEL # This label might not show unless explicitly added to legend
            },
        vmin=VMIN,                 # Set the minimum value for the color scale
        vmax=VMAX                  # Set the maximum value for the color scale
        # scheme='quantiles', k=5 # Optional: Use classification scheme instead of pure continuous
    )

    # 4. Customize Appearance
    ax.set_axis_off()
    ax.set_title(f'Catalonia Rent Affordability - {year}', fontdict={'fontsize': '16', 'fontweight': '3'})
    # Add range annotation
    ax.annotate(f'Color scale: {VMIN}% - {VMAX}% ({COLOR_MAP} reversed)',
                xy=(0.1, .08), xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=8, color='#555555')
    # Add a legend entry for missing data manually if needed
    # ax.plot([], [], color=MISSING_DATA_COLOR, label=MISSING_DATA_LABEL)
    # ax.legend(loc='lower left')


    # 5. Save the figure
    output_filename = OUTPUT_DIR / f"catalonia_affordability_cont_{year}.png"
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Map successfully saved to {output_filename}")
    except Exception as e:
        print(f"Error saving map image: {e}")
    plt.close(fig) # Close figure to free memory


# --- Main Execution ---
if __name__ == "__main__":

    spark = None
    try:
        # --- Step 1: Initialize Spark ---
        spark = get_spark_session()

        # --- Step 2: Load Shapefile ---
        if not SHAPEFILE_PATH.is_file():
            print(f"ERROR: Shapefile not found at '{SHAPEFILE_PATH}'.")
            exit()
        print(f"Loading shapefile from {SHAPEFILE_PATH}...")
        catalunya_map_gdf = gpd.read_file(SHAPEFILE_PATH)
        print("Shapefile loaded successfully.")
        if MUN_CODE_COLUMN_IN_SHAPEFILE not in catalunya_map_gdf.columns:
             print(f"ERROR: Column '{MUN_CODE_COLUMN_IN_SHAPEFILE}' not found in shapefile.")
             print(f"Available columns: {list(catalunya_map_gdf.columns)}")
             exit()
        print(f"Using column '{MUN_CODE_COLUMN_IN_SHAPEFILE}' for municipality code.")

        # --- Step 3: Load Full Statistical Data from Spark ---
        stats_df_spark = load_and_prepare_stats_data_spark(spark, EXPLOITATION_DATA_PATH)

        if stats_df_spark is None:
             print("Exiting due to error loading statistical data.")
             exit()

        # --- Step 4: Identify Valid Years ---
        valid_years = identify_valid_years(stats_df_spark)

        if not valid_years:
             print("No years found with sufficient data for plotting. Exiting.")
             exit()

        # --- Step 5: Convert Full Relevant Data to Pandas ONCE ---
        print("\nConverting relevant Spark DataFrame data to Pandas DataFrame for plotting...")
        stats_df_pandas = stats_df_spark.toPandas()
        print(f"Conversion complete. Pandas DataFrame has {len(stats_df_pandas)} rows.")

        # --- Step 6: Loop Through Valid Years and Plot ---
        print(f"\n--- Starting Map Generation Loop for Years: {valid_years} ---")
        for year in valid_years:
            # Filter the PANDAS DataFrame for the current year
            stats_data_this_year = stats_df_pandas[stats_df_pandas['any'] == year].copy() # Use .copy() to avoid SettingWithCopyWarning

            if stats_data_this_year.empty:
                 print(f"Warning: No data after filtering Pandas DF for year {year}. Skipping plot.")
                 continue

            # Generate and Save the Map for this year
            try:
                plot_affordability_map_continuous(year, stats_data_this_year, catalunya_map_gdf)
            except KeyError as e:
                 print(f"\nERROR: Plotting failed for year {year} due to KeyError.")
                 print(f"Please check column names used in plotting match DataFrame columns.")
                 print(f"Stats DataFrame columns for year {year}: {list(stats_data_this_year.columns)}")
                 print(f"Specific error: {e}")
                 traceback.print_exc()
            except Exception as e:
                print(f"\nAn unexpected error occurred during plotting for year {year}: {e}")
                traceback.print_exc()
        print(f"\n--- Finished Map Generation Loop ---")

    except Exception as main_error:
        print(f"\nAn unexpected error occurred in the main execution block: {main_error}")
        traceback.print_exc()
    finally:
        if spark:
            print("\nStopping Spark Session.")
            spark.stop()
            print("Spark Session stopped.")

    print("\nScript finished.")