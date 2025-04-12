# plot_catalonia_affordability.py

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Configuration ---

# 1. Path to your downloaded Shapefile (specifically the .shp file)
SHAPEFILE_PATH = Path("./geospatial/MUC_TM.shp")

# 2. *** IMPORTANT: Placeholder for Municipality Code Column ***
MUN_CODE_COLUMN_IN_SHAPEFILE = "CODI_INE" # <-- ### REPLACE THIS ###

# 3. Define affordability thresholds and corresponding colors/labels 
#    Intervals: [0, 20), [20, 30), [30, 40), [40, inf)
thresholds = [0, 20, 30, 40, np.inf]
colors = ['#2ca02c', '#ffdd71', '#ff7f0e', '#d62728'] # Green, Yellow, Orange, Red (Matplotlib Tableau Colors)
labels = ['< 20% (Affordable)', '20-30% (Manageable)', '30-40% (Stressed)', '> 40% (Severe)']
missing_data_color = 'lightgrey'
missing_data_label = 'Missing Data'

# 4. Year to plot
TARGET_YEAR = 2022 # Example Year

# 5. Output directory for the map image
OUTPUT_DIR = Path("./data/analysis_output/maps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Create if it doesn't exist

# --- Placeholder Data (Replace with reading your Exploitation Zone Delta table later) ---
# This mimics data you would get AFTER joining Idescat, Rent, Income in Exploitation Zone
placeholder_data = {
    'municipality_id': ['08019', '08101', '08279', '17079', '25120', '43148', '08901', '43010', '17001'], # Added a couple more dummy codes
    'year': [TARGET_YEAR] * 9,
    'municipality_name': ['Barcelona', 'L\'Hospitalet de Llobregat', 'Terrassa', 'Girona', 'Lleida', 'Tarragona', 'Badia del Vallès', 'Alió', 'Agullana'], # Example names
    'avg_monthly_rent_eur': [1100, 950, 780, 720, 550, 670, 600, 400, 350], # Invented rents (more variation)
    'avg_annual_income_eur': [32000, 25000, 28000, 29500, 25500, 27500, 20000, 22000, 24000] # Invented incomes (more variation)
}
stats_df = pd.DataFrame(placeholder_data)
# ------------------------------------------------------------------------------------

def calculate_affordability(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates rent-to-income ratio and assigns category."""
    df_calc = df.copy()
    print("Calculating affordability ratios...")
    df_calc['yearly_rent'] = df_calc['avg_monthly_rent_eur'] * 12
    df_calc['affordability_pct'] = np.where(
        (df_calc['avg_annual_income_eur'].notna()) & (df_calc['avg_annual_income_eur'] > 0) & (df_calc['yearly_rent'].notna()),
        (df_calc['yearly_rent'] / df_calc['avg_annual_income_eur']) * 100,
        np.nan # Assign NaN if income is missing/zero or rent is missing
    )

    print("Categorizing affordability...")
    df_calc['affordability_category'] = pd.cut(
        df_calc['affordability_pct'],
        bins=thresholds,
        labels=labels,
        right=False # Intervals are [min, max)
    )
    # Convert category to string and handle NaNs resulting from NaN percentages
    df_calc['affordability_category'] = df_calc['affordability_category'].astype(str).replace('nan', missing_data_label)
    print("Affordability calculation complete.")
    return df_calc

def plot_affordability_map(year: int, stats_data: pd.DataFrame, shapefile_gdf: gpd.GeoDataFrame):
    """Generates and saves the choropleth map for a given year."""
    print(f"Generating map for year {year}...")

    # 1. Prepare statistical data for the year
    yearly_stats = stats_data[stats_data['year'] == year].copy()
    if yearly_stats.empty:
        print(f"ERROR: No statistical data found for year {year}. Cannot generate map.")
        return
    yearly_stats = calculate_affordability(yearly_stats)

    # 2. Merge geospatial data with statistical data
    print("Merging statistical data with shapefile...")
    # --- Ensure merge keys are strings ---
    shapefile_gdf[MUN_CODE_COLUMN_IN_SHAPEFILE] = shapefile_gdf[MUN_CODE_COLUMN_IN_SHAPEFILE].astype(str)
    yearly_stats['municipality_id'] = yearly_stats['municipality_id'].astype(str)

    merged_gdf = shapefile_gdf.merge(
        yearly_stats,
        left_on=MUN_CODE_COLUMN_IN_SHAPEFILE,
        right_on='municipality_id',
        how='left' # Keep all map polygons
    )

    # Fill missing category for municipalities not found in stats data
    merged_gdf['affordability_category'] = merged_gdf['affordability_category'].fillna(missing_data_label)
    print(f"Merge complete. Plotting {len(merged_gdf)} municipalities.")

    # 3. Plotting
    print("Creating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12)) # Adjust figsize if needed

    # Define custom colors for categories
    category_colors = {label: color for label, color in zip(labels, colors)}
    category_colors[missing_data_label] = missing_data_color

    # Generate list of colors for each polygon based on its category
    plot_colors = [category_colors.get(cat, missing_data_color) for cat in merged_gdf['affordability_category']]

    merged_gdf.plot(
        color=plot_colors, # Use the generated list of colors
        linewidth=0.4,     # Thinner borders
        edgecolor='0.6',   # Lighter grey borders
        ax=ax
    )

    # 4. Customize Appearance
    ax.set_axis_off()
    ax.set_title(f'Catalonia Rent Affordability ({year})', fontdict={'fontsize': '16', 'fontweight': '3'})
    ax.annotate(f'Affordability = (Monthly Rent * 12) / Annual Income * 100%',
                xy=(0.1, .08), xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=8, color='#555555')

    # 5. Create Custom Legend (because assigning colors directly bypasses default legend)
    print("Creating custom legend...")
    legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in colors]
    legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=missing_data_color))
    ax.legend(legend_patches, labels + [missing_data_label],
              title="Affordability (% Rent/Income)",
              loc='lower left', fontsize='small', title_fontsize='medium')

    # 6. Save the figure
    output_filename = OUTPUT_DIR / f"catalonia_affordability_{year}.png"
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Map successfully saved to {output_filename}")
    except Exception as e:
        print(f"Error saving map image: {e}")
    plt.close(fig) # Close the figure


# --- Main Execution ---
if __name__ == "__main__":

    # --- Step 1: Load Shapefile and Inspect ---
    if not SHAPEFILE_PATH.is_file():
        print(f"ERROR: Shapefile not found at '{SHAPEFILE_PATH}'. Please ensure the path is correct.")
        exit()

    print(f"Loading shapefile from {SHAPEFILE_PATH}...")
    try:
        catalunya_map_gdf = gpd.read_file(SHAPEFILE_PATH)
        print("Shapefile loaded successfully.")
        print("Available columns in shapefile:", list(catalunya_map_gdf.columns))
        print("\nFirst 5 rows of shapefile attributes:")
        print(catalunya_map_gdf.head())
        print("-" * 30)
        print(f"IMPORTANT: Please verify that '{MUN_CODE_COLUMN_IN_SHAPEFILE}' is the correct column for the 5-digit municipality code.")
        print(f"If not, update the MUN_CODE_COLUMN_IN_SHAPEFILE variable at the top of this script.")
        print("-" * 30)

        # Optional: Check CRS and project if needed
        # print(f"Shapefile CRS: {catalunya_map_gdf.crs}")
        # if catalunya_map_gdf.crs and catalunya_map_gdf.crs != "EPSG:4326":
        #     print("Reprojecting shapefile to EPSG:4326 (WGS84)...")
        #     catalunya_map_gdf = catalunya_map_gdf.to_crs("EPSG:4326")

    except Exception as e:
        print(f"ERROR loading or inspecting shapefile: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- Step 2: Load or Prepare Statistical Data ---
    # For now, we use the placeholder `stats_df` defined above.
    # ** In your final pipeline, replace this with reading from your Exploitation Zone Delta table **
    # Example:
    # spark = get_spark_session(...) # Get your Spark session
    # exploitation_path = "./data/exploitation/municipal_yearly_analysis_view"
    # exploitation_df = spark.read.format("delta").load(exploitation_path)
    # Filter for needed columns and years if necessary
    # stats_df = exploitation_df.filter(F.col("year") == TARGET_YEAR)\
    #                           .select("municipality_id", "year", "municipality_name", "avg_monthly_rent_eur", "avg_annual_income_eur")\
    #                           .toPandas()
    print(f"Using placeholder statistical data for year {TARGET_YEAR}.")


    # --- Step 3: Generate and Save the Map ---
    try:
        plot_affordability_map(TARGET_YEAR, stats_df, catalunya_map_gdf)
    except KeyError as e:
         print(f"\nERROR: A KeyError occurred, likely related to column names.")
         print(f"Please double-check:")
         print(f"  1. MUN_CODE_COLUMN_IN_SHAPEFILE = '{MUN_CODE_COLUMN_IN_SHAPEFILE}' is correct for the shapefile.")
         print(f"  2. Your stats_df contains 'municipality_id', 'year', 'avg_monthly_rent_eur', 'avg_annual_income_eur'.")
         print(f"Specific error: {e}")
         import traceback
         traceback.print_exc()
    except Exception as e:
        print(f"\nAn unexpected error occurred during plotting: {e}")
        import traceback
        traceback.print_exc()

    print("\nScript finished.")