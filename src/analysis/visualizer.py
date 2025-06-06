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

# --- Spark Session Utility (Keep external or move inside if preferred) ---
from spark_session import get_spark_session
# -----------------------------------------------------------------------


class CatalanAffordabilityVisualizer:
    """
    Analyzes and visualizes Catalan rent affordability over multiple years using Spark and GeoPandas.

    Loads rent and income data from a Delta table, merges it with municipality
    shapefiles, calculates an affordability ratio, and generates choropleth maps
    for each year with valid data.
    """

    MUN_CODE_COLUMN_IN_SHAPEFILE = "CODI_INE"
    RENT_COLUMN = "avg_monthly_rent_eur"
    INCOME_COLUMN = "salary_per_capita_eur" # Using salary as the best available income proxy
    COLOR_MAP = 'RdYlGn_r'  # Red-Yellow-Green (Reversed: Low=Green, High=Red)
    COLORBAR_LABEL = "Affordability (% Monthly Rent * 12 / Annual Salary)"
    VMIN = 10  # Minimum percentage for color scale 
    VMAX = 50  # Maximum percentage for color scale
    MISSING_DATA_COLOR = 'lightgrey'
    MISSING_DATA_LABEL = 'Missing Data'
    # --------------------------------------------------------------

    def __init__(self, spark: SparkSession, exploitation_data_path: str = "./data/exploitation/municipal_annual", shapefile_path: str = "./geospatial/MUC_TM.shp", output_dir: str = "./data/analysis/visualizer"):
        """
        Initializes the visualizer with paths and optional custom configuration.

        Args:
            exploitation_data_path: Path to the exploitation zone Delta table.
            shapefile_path: Path to the municipality shapefile (.shp).
            output_dir: Directory to save the generated map images.
            config: Optional dictionary to override default configuration values
                    (e.g., column names, plot settings).
        """
        self.exploitation_data_path = exploitation_data_path
        self.shapefile_path = Path(shapefile_path)
        self.output_dir = Path(output_dir)
        # ------------------------------------

        self.spark = spark
        self.catalunya_map_gdf: Optional[gpd.GeoDataFrame] = None
        self.stats_df_spark: Optional[DataFrame] = None
        self.stats_df_pandas: Optional[pd.DataFrame] = None

        print("--- CatalanAffordabilityVisualizer Initialized ---")
        print(f"Exploitation Data Path: {self.exploitation_data_path}")
        print(f"Shapefile Path: {self.shapefile_path}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Using Rent Column: {self.RENT_COLUMN}")
        print(f"Using Income Column: {self.INCOME_COLUMN}")
        print(f"Using Shapefile Code Column: {self.MUN_CODE_COLUMN_IN_SHAPEFILE}")
        print("-------------------------------------------------")


    def _load_shapefile(self) -> bool:
        """Loads the shapefile into a GeoDataFrame."""
        print(f"\n--- Loading Shapefile ---")
        if not self.shapefile_path.is_file():
            print(f"ERROR: Shapefile not found at '{self.shapefile_path}'.")
            return False
        try:
            print(f"Loading shapefile from {self.shapefile_path}...")
            self.catalunya_map_gdf = gpd.read_file(self.shapefile_path)
            print("Shapefile loaded successfully.")

            # Validate municipality code column
            if self.MUN_CODE_COLUMN_IN_SHAPEFILE not in self.catalunya_map_gdf.columns:
                print(f"ERROR: Column '{self.MUN_CODE_COLUMN_IN_SHAPEFILE}' not found in shapefile.")
                print(f"Available columns: {list(self.catalunya_map_gdf.columns)}")
                return False
            print(f"Using column '{self.MUN_CODE_COLUMN_IN_SHAPEFILE}' for municipality code.")

            return True
        except Exception as e:
            print(f"ERROR loading shapefile: {e}")
            traceback.print_exc()
            return False

    def _load_and_prepare_stats_data_spark(self) -> bool:
        """Loads exploitation data via Spark, selects columns, returns success status."""
        if not self.spark:
             print("ERROR: Spark session not initialized.")
             return False

        print(f"\n--- Loading and Preparing Statistical Data from Spark ---")
        try:
            print(f"Reading Delta table from: {self.exploitation_data_path}")
            exploitation_df = self.spark.read.format("delta").load(self.exploitation_data_path)
            print("Exploitation data loaded. Schema:")
            exploitation_df.printSchema()

            # Check if required columns exist
            required_cols = ["municipality_id", "any", self.RENT_COLUMN, self.INCOME_COLUMN, "municipality_name"]
            if not all(col in exploitation_df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in exploitation_df.columns]
                print(f"ERROR: Exploitation data missing required columns: {missing}.")
                return False

            # Select necessary columns for all years
            self.stats_df_spark = exploitation_df.select(*required_cols) # Use list unpacking
            print("Required columns selected.")
            return True

        except Exception as e:
            print(f"ERROR loading or preparing statistical data from Spark: {e}")
            traceback.print_exc()
            return False

    def _identify_valid_years(self) -> List[int]:
        """Identifies years with at least one municipality having non-null rent AND income."""
        if self.stats_df_spark is None:
            print("ERROR: Spark statistical DataFrame not loaded.")
            return []

        print("\n--- Identifying valid years for plotting ---")
        try:
            valid_years_df = self.stats_df_spark.filter(
                F.col(self.RENT_COLUMN).isNotNull() & F.col(self.INCOME_COLUMN).isNotNull()
            ).select("any").distinct()

            valid_years = sorted([row.any for row in valid_years_df.collect()])
            print(f"Found {len(valid_years)} year(s) with valid data: {valid_years}")
            return valid_years
        except Exception as e:
            print(f"ERROR identifying valid years: {e}")
            traceback.print_exc()
            return []

    def _calculate_affordability_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates rent-to-income ratio numerically on a Pandas DataFrame."""
        df_calc = df.copy()
        df_calc['yearly_rent'] = df_calc[self.RENT_COLUMN] * 12
        df_calc['affordability_pct'] = np.where(
            (df_calc[self.INCOME_COLUMN].notna()) & (df_calc[self.INCOME_COLUMN] > 0) & (df_calc['yearly_rent'].notna()),
            (df_calc['yearly_rent'] / df_calc[self.INCOME_COLUMN]) * 100,
            np.nan # Assign NaN if income is missing/zero or rent is missing
        )
        return df_calc

    def _plot_affordability_map_continuous(self, year: int, stats_data_for_year: pd.DataFrame):
        """Generates and saves the continuous color choropleth map for a given year."""
        if self.catalunya_map_gdf is None:
            print(f"ERROR: Shapefile GeoDataFrame not loaded. Cannot plot year {year}.")
            return

        print(f"\n--- Generating Map for Year {year} ---")

        # 1. Calculate affordability percentage
        stats_with_affordability = self._calculate_affordability_numeric(stats_data_for_year)

        # 2. Merge geospatial data with statistical data
        print(f"Merging data for year {year} with shapefile...")
        # --- Ensure merge keys are strings ---
        map_gdf_copy = self.catalunya_map_gdf.copy() # Work on a copy to avoid modifying original
        map_gdf_copy[self.MUN_CODE_COLUMN_IN_SHAPEFILE] = map_gdf_copy[self.MUN_CODE_COLUMN_IN_SHAPEFILE].astype(str)
        stats_with_affordability['municipality_id'] = stats_with_affordability['municipality_id'].astype(str)

        merged_gdf = map_gdf_copy.merge(
            stats_with_affordability,
            left_on=self.MUN_CODE_COLUMN_IN_SHAPEFILE,
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
            cmap=self.COLOR_MAP,       # Use the continuous colormap
            linewidth=0.4,
            ax=ax,
            edgecolor='0.6',
            legend=True,               # Add a colorbar legend
            legend_kwds={
                'label': self.COLORBAR_LABEL,
                'orientation': "vertical", # Or "horizontal"
                'shrink': 0.6 # Adjust size of colorbar if needed
                },
            missing_kwds={             # How to draw areas with missing data
                "color": self.MISSING_DATA_COLOR,
                "edgecolor": "grey",
                "label": self.MISSING_DATA_LABEL # This label might not show unless explicitly added to legend
                },
            vmin=self.VMIN,            # Set the minimum value for the color scale
            vmax=self.VMAX             # Set the maximum value for the color scale
        )

        # 4. Customize Appearance
        ax.set_axis_off()
        ax.set_title(f'Catalonia Rent Affordability - {year}', fontdict={'fontsize': '16', 'fontweight': '3'})
        # Add range annotation
        ax.annotate(f'Color scale: {self.VMIN}% - {self.VMAX}% ({self.COLOR_MAP} reversed)',
                    xy=(0.1, .08), xycoords='figure fraction',
                    horizontalalignment='left', verticalalignment='top',
                    fontsize=8, color='#555555')


        # 5. Save the figure
        output_filename = self.output_dir / f"catalonia_affordability_cont_{year}.png"
        try:
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"Map successfully saved to {output_filename}")
        except Exception as e:
            print(f"Error saving map image for year {year}: {e}")
        plt.close(fig) # Close figure to free memory


    def run(self):
        """Executes the full pipeline: load data, process, and generate maps."""
        print("\n===== Starting Affordability Visualization Pipeline =====")
        try:
            # --- Step 1: Initialize Spark ---
            self.spark = get_spark_session()

            # --- Step 2: Load Shapefile ---
            if not self._load_shapefile():
                print("Exiting due to shapefile loading error.")
                return 

            # --- Step 3: Load Full Statistical Data from Spark ---
            if not self._load_and_prepare_stats_data_spark():
                print("Exiting due to error loading statistical data.")
                return # Exit run method

            # --- Step 4: Identify Valid Years ---
            valid_years = self._identify_valid_years()

            if not valid_years:
                print("No years found with sufficient data for plotting. Exiting.")
                return 

            # --- Step 5: Convert Full Relevant Data to Pandas ONCE ---
            print("\nConverting relevant Spark DataFrame data to Pandas DataFrame for plotting...")
            try:
                self.stats_df_pandas = self.stats_df_spark.toPandas()
                print(f"Conversion complete. Pandas DataFrame has {len(self.stats_df_pandas)} rows.")
            except Exception as e:
                print(f"ERROR converting Spark DataFrame to Pandas: {e}")
                traceback.print_exc()
                return # Exit run method

            # --- Step 6: Loop Through Valid Years and Plot ---
            print(f"\n--- Starting Map Generation Loop for Years: {valid_years} ---")
            for year in valid_years:
                # Filter the PANDAS DataFrame for the current year
                stats_data_this_year = self.stats_df_pandas[self.stats_df_pandas['any'] == year].copy() 

                if stats_data_this_year.empty:
                    print(f"Warning: No data after filtering Pandas DF for year {year}. Skipping plot.")
                    continue

                # Generate and Save the Map for this year
                try:
                    self._plot_affordability_map_continuous(year, stats_data_this_year)
                except KeyError as e:
                    print(f"\nERROR: Plotting failed for year {year} due to KeyError.")
                    print(f"Please check column names used in plotting match DataFrame columns.")
                    print(f"Stats DataFrame columns for year {year}: {list(stats_data_this_year.columns)}")
                    print(f"Specific error: {e}")
                    traceback.print_exc()
                    # Continue to the next year
                except Exception as e:
                    print(f"\nAn unexpected error occurred during plotting for year {year}: {e}")
                    traceback.print_exc()
                    # Continue to the next year

            print(f"\n--- Finished Map Generation Loop ---")

        except Exception as main_error:
            print(f"\nAn critical error occurred in the main execution pipeline: {main_error}")
            traceback.print_exc()

        print("\n===== Affordability Visualization Pipeline Finished =====")


# --- Main Execution ---
if __name__ == "__main__":
    from src.spark_session import get_spark_session

    # --- Run the visualization ---
    spark = get_spark_session()
    visualizer = CatalanAffordabilityVisualizer(spark=spark)
    visualizer.run()

    print("\nScript finished.")