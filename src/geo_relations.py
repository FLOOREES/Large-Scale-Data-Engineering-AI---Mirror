# generate_geo_relations_final_v4.py

import geopandas as gpd
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import traceback
from collections import defaultdict

# --- CONFIGURATION ---
SHAPEFILE_PATH = "./geospatial/MUC_TM.shp"
SHAPEFILE_MUN_CODE_COL = "CODI_INE"

IDESCAT_LANDING_PARQUET = "./data/landing/idescat.parquet"
IDESCAT_MUN_CODE_COL = "municipality_id"
IDESCAT_MUN_NAME_COL = "municipality_name"
IDESCAT_COMARCA_NAME_COL = "comarca_name"

COMARCA_TO_PROVINCE_MAP = {
    "Alt Camp": "Tarragona", "Alt Empordà": "Girona", "Alt Penedès": "Barcelona",
    "Alt Urgell": "Lleida", "Alta Ribagorça": "Lleida", "Anoia": "Barcelona",
    "Bages": "Barcelona", "Baix Camp": "Tarragona", "Baix Ebre": "Tarragona",
    "Baix Empordà": "Girona", "Baix Llobregat": "Barcelona", "Baix Penedès": "Tarragona",
    "Barcelonès": "Barcelona", "Berguedà": "Barcelona", "Cerdanya": "Girona",
    "Conca de Barberà": "Tarragona", "Garraf": "Barcelona", "Garrigues": "Lleida",
    "Garrotxa": "Girona", "Gironès": "Girona", "Lluçanès": "Barcelona", "Maresme": "Barcelona",
    "Moianès": "Barcelona", "Montsià": "Tarragona", "Noguera": "Lleida",
    "Aran": "Lleida", "Osona": "Barcelona", "Pallars Jussà": "Lleida",
    "Pallars Sobirà": "Lleida", "Pla d'Urgell": "Lleida", "Pla de l'Estany": "Girona",
    "Priorat": "Tarragona", "Ribera d'Ebre": "Tarragona", "Ripollès": "Girona",
    "Segarra": "Lleida", "Segrià": "Lleida", "Selva": "Girona",
    "Solsonès": "Lleida", "Tarragonès": "Tarragona", "Terra Alta": "Tarragona",
    "Urgell": "Lleida", "Vallès Occidental": "Barcelona", "Vallès Oriental": "Barcelona"
}

OUTPUT_DIR = Path("./data/relations")
# --- END OF CONFIGURATION ---

def create_geodata_from_sources(idescat_path, shapefile_path):
    """Creates a master GeoDataFrame by merging Idescat data with shapefile geometry."""
    print("--- Step 1: Creating Master Geo-Data Source ---")

    print(f"Reading names and comarca links from: {idescat_path}")
    idescat_df = pd.read_parquet(idescat_path)
    geo_info_df = idescat_df[[
        IDESCAT_MUN_CODE_COL,
        IDESCAT_MUN_NAME_COL,
        IDESCAT_COMARCA_NAME_COL
    ]].drop_duplicates(subset=[IDESCAT_MUN_CODE_COL]).reset_index(drop=True)
    geo_info_df['province_name'] = geo_info_df[IDESCAT_COMARCA_NAME_COL].map(COMARCA_TO_PROVINCE_MAP)

    print(f"Loading geometry from shapefile: {shapefile_path}")
    gdf_shape = gpd.read_file(shapefile_path)

    print("\nNormalizing keys for merge...")
    geo_info_df[IDESCAT_MUN_CODE_COL] = geo_info_df[IDESCAT_MUN_CODE_COL].astype(str).str.strip().str.slice(stop=5)
    gdf_shape[SHAPEFILE_MUN_CODE_COL] = gdf_shape[SHAPEFILE_MUN_CODE_COL].astype(str).str.strip().str.slice(stop=5)
    gdf_shape = gdf_shape.drop_duplicates(subset=[SHAPEFILE_MUN_CODE_COL])
    
    print("\nMerging Idescat data with shapefile geometry...")
    master_gdf = geo_info_df.merge(
        gdf_shape[[SHAPEFILE_MUN_CODE_COL, 'geometry']],
        left_on=IDESCAT_MUN_CODE_COL,
        right_on=SHAPEFILE_MUN_CODE_COL,
        how="left"
    )
    master_gdf = gpd.GeoDataFrame(master_gdf, geometry='geometry')

    master_gdf.dropna(subset=['geometry'], inplace=True)
    print(f"Master GeoDataFrame created with {len(master_gdf)} municipalities with valid geometry.")
    return master_gdf

def generate_municipality_neighbors(gdf: gpd.GeoDataFrame) -> dict:
    """Calculates municipality neighbors using a robust intersection method."""
    print("\n--- Step 2: Calculating Municipality Neighbors (Robust Method) ---")
    
    gdf_indexed = gdf.set_index(IDESCAT_MUN_CODE_COL)
    if not gdf_indexed.has_sindex:
        print("Creating spatial index for faster neighbor search...")
        gdf_indexed.sindex

    neighbors_dict = {}
    buffer_distance = 1

    for mun_code, municipality in tqdm(gdf_indexed.iterrows(), total=gdf_indexed.shape[0], desc="Municipalities"):
        buffered_geom = municipality.geometry.buffer(buffer_distance)
        possible_matches_index = list(gdf_indexed.sindex.intersection(buffered_geom.bounds))
        possible_matches = gdf_indexed.iloc[possible_matches_index]
        neighbors = possible_matches[possible_matches.intersects(buffered_geom) & (possible_matches.index != mun_code)]
        
        neighbor_list = [{"id": neighbor_id, "name": neighbor_row[IDESCAT_MUN_NAME_COL]} for neighbor_id, neighbor_row in neighbors.iterrows()]
        neighbors_dict[mun_code] = {"name": municipality[IDESCAT_MUN_NAME_COL], "neighbors": neighbor_list}
        
    output_path = OUTPUT_DIR / "municipality_neighbors.json"
    print(f"Saving municipality neighbors to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(neighbors_dict, f, indent=4, ensure_ascii=False)
    print("Municipality neighbors JSON created successfully.")
    return neighbors_dict


def generate_comarca_relations(master_gdf: gpd.GeoDataFrame, municipality_neighbors: dict):
    """
    Derives comarca neighbors and comarca-to-province mappings using the
    pre-calculated municipality data.
    """
    print("\n--- Step 3: Generating Comarca-Level Relations (Derived Method) ---")

    # 1. Create a municipality ID -> comarca name lookup dictionary
    mun_to_comarca_map = master_gdf.set_index(IDESCAT_MUN_CODE_COL)[IDESCAT_COMARCA_NAME_COL].to_dict()

    # 2. Derive comarca neighbors by iterating through municipality neighbors
    comarca_adjacencies = defaultdict(set)
    print("Deriving comarca neighbors from municipality adjacencies...")
    for mun_id, data in municipality_neighbors.items():
        comarca_A = mun_to_comarca_map.get(mun_id)
        if not comarca_A: continue

        for neighbor in data['neighbors']:
            neighbor_id = neighbor['id']
            comarca_B = mun_to_comarca_map.get(neighbor_id)
            if not comarca_B: continue

            if comarca_A != comarca_B:
                comarca_adjacencies[comarca_A].add(comarca_B)
                comarca_adjacencies[comarca_B].add(comarca_A)

    # Convert sets to sorted lists for consistent JSON output
    comarca_neighbors_dict = {comarca: {"neighbors": sorted(list(neighbors))} for comarca, neighbors in comarca_adjacencies.items()}
    
    output_path_com = OUTPUT_DIR / "comarca_neighbors.json"
    print(f"Saving comarca neighbors to: {output_path_com}")
    with open(output_path_com, 'w', encoding='utf-8') as f:
        json.dump(comarca_neighbors_dict, f, indent=4, ensure_ascii=False)
    print("Comarca neighbors JSON created successfully.")

    # 3. Create comarca-to-province map
    comarca_to_province_map = master_gdf.drop_duplicates(subset=[IDESCAT_COMARCA_NAME_COL]).set_index(IDESCAT_COMARCA_NAME_COL)['province_name'].to_dict()
    output_path_prov = OUTPUT_DIR / "comarca_to_province.json"
    print(f"Saving comarca-to-province map to: {output_path_prov}")
    with open(output_path_prov, 'w', encoding='utf-8') as f:
        json.dump(comarca_to_province_map, f, indent=4, ensure_ascii=False)
    print("Comarca-to-province JSON created successfully.")


def main():
    """Main function to run the geo-relation generation process."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        master_gdf = create_geodata_from_sources(IDESCAT_LANDING_PARQUET, SHAPEFILE_PATH)
        
        if master_gdf.empty:
            print("\nProcess finished: master GeoDataFrame is empty.")
            return

        municipality_neighbors = generate_municipality_neighbors(master_gdf)
        
        if not municipality_neighbors:
             print("\nProcess finished: no municipality neighbors were generated.")
             return
        
        generate_comarca_relations(master_gdf, municipality_neighbors)
        
        print("\nProcess completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\nERROR: A required file was not found. Please check paths.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()