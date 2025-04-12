import requests
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Optional, Any

class IdescatLandingZone:
    """
    Handles fetching selected Idescat EMEX indicators for all municipalities
    and saving the results to the landing zone in Parquet format.
    """
    DEFAULT_MAX_WORKERS = 10 

    def __init__(self, max_workers: int = DEFAULT_MAX_WORKERS):
        self.chosen_indicators = [
            'f171', # Population
            'f36',  # Men
            'f42',  # Women
            'f187', # Births (Total)
            'f183', # Pop. Spanish Nationality
            'f261', # Surface area (km²)
            'f262', # Density (Pop/km²)
            'f328', # Longitude
            'f329', # Latitude
            'f308', # Total Registered Unemployment
            'f191', # Habitatges familiars
            'f270', # Biblioteques públiques
            'f293', # Pavellons
            'f294', # Pistes poliesportives
            'f301', # Piscines cobertes
        ]
        self.indicators_query_string = ','.join(self.chosen_indicators)
        self.api_url = "https://api.idescat.cat/emex/v1"
        self.max_workers = max_workers
        self.timeout = 20

    def _get_all_municipality_ids(self) -> List[str]:
        """Fetches the list of all municipality IDs from the Idescat API."""
        print("Fetching list of all municipalities...")
        nodes_url = f"{self.api_url}/nodes.json?tipus=mun"
        try:
            response = requests.get(nodes_url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            municipality_ids = []
            # Recursive function to handle nested structure
            def extract_municipalities(node):
                ids = []
                if node.get('scheme') == 'mun':
                    ids.append(node.get('id'))
                children = node.get('v', [])
                if isinstance(children, dict): children = [children] # Handle single child case
                if isinstance(children, list):
                    for child_node in children:
                        ids.extend(extract_municipalities(child_node))
                return ids

            root_nodes = data.get('fitxes', {}).get('v', [])
            if isinstance(root_nodes, dict): root_nodes = [root_nodes] # Handle single root node
            for root_node in root_nodes:
                 municipality_ids.extend(extract_municipalities(root_node))

            unique_ids = sorted(list(set(municipality_ids))) # Ensure uniqueness and sort
            if not unique_ids:
                 print("Warning: No municipality IDs extracted. Check API response structure.")
            else:
                 print(f"Found {len(unique_ids)} unique municipality IDs.")
            return unique_ids
        except requests.exceptions.RequestException as e:
            print(f"Error fetching municipality list: {e}")
            return []
        except json.JSONDecodeError:
            print(f"Error decoding JSON response for municipality list.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred getting municipality IDs: {e}")
            return []

    def _fetch_indicators_for_municipality(self, municipality_id: str) -> List[Dict[str, Any]]:
        """
        Fetches chosen indicators for a single municipality (thread-safe)
        and includes municipality and comarca names. Returns a list of dicts.
        """
        data_url = f"{self.api_url}/dades.json?id={municipality_id}&i={self.indicators_query_string}"
        try:
            response = requests.get(data_url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # --- Extract Names ---
            municipality_name: Optional[str] = None
            comarca_name: Optional[str] = None
            try:
                cols_data = data.get('fitxes', {}).get('cols', {}).get('col', [])
                if isinstance(cols_data, dict): cols_data = [cols_data]
                if isinstance(cols_data, list):
                    for col_info in cols_data:
                        scheme = col_info.get('@scheme', col_info.get('scheme'))
                        name_content = col_info.get('#content', col_info.get('content', col_info.get('c')))
                        if scheme == 'mun': municipality_name = name_content
                        elif scheme == 'com': comarca_name = name_content
            except Exception: pass # Ignore parsing errors for names, keep as None

            # --- Extract Indicator Data ---
            results = []
            indicators_data = data.get('fitxes', {}).get('indicadors', {}).get('i', [])
            if not isinstance(indicators_data, list): indicators_data = []

            for indicator in indicators_data:
                if 'error' in indicator: continue # Skip indicators the API reported errors for

                indicator_id = indicator.get('id')
                indicator_name_api = indicator.get('c')
                values_str = indicator.get('v')
                reference_year = indicator.get('r')
                updated_ts = indicator.get('updated')

                # Ensure essential IDs and values are present
                if values_str is None or indicator_id is None: continue

                values_list = values_str.split(',')
                mun_val = values_list[0] if len(values_list) > 0 else None
                com_val = values_list[1] if len(values_list) > 1 else None
                cat_val = values_list[2] if len(values_list) > 2 else None

                results.append({
                    "municipality_id": municipality_id,
                    "municipality_name": municipality_name,
                    "comarca_name": comarca_name,
                    "indicator_id": indicator_id,
                    "indicator_name": indicator_name_api, # Keep original API name
                    "reference_year": reference_year,
                    "municipality_value": mun_val if mun_val != '_' else None, # Convert API missing ('_') to None
                    "comarca_value": com_val if com_val != '_' else None,
                    "catalunya_value": cat_val if cat_val != '_' else None,
                    "source_update_timestamp": updated_ts
                })
            return results

        # Handle specific request/processing errors for this municipality
        except requests.exceptions.Timeout: return [] # Don't print warnings for common errors in parallel runs
        except requests.exceptions.HTTPError: return []
        except requests.exceptions.RequestException: return []
        except json.JSONDecodeError: return []
        except Exception: return [] # Catch-all for unexpected issues

    def _process_and_save_data(self, fetched_data: List[Dict[str, Any]], save_csv: bool = False):
        """
        Processes the fetched data into a Pandas DataFrame, cleans types,
        and saves it to Parquet (and optionally CSV).
        """
        print("\nProcessing fetched data and saving...")
        if not fetched_data:
            print("No data was successfully fetched. No output files generated.")
            return

        df_landing = pd.DataFrame(fetched_data)

        # --- Data Cleaning and Type Conversion ---
        numeric_cols = ['municipality_value', 'comarca_value', 'catalunya_value']
        for col in numeric_cols:
            df_landing[col] = pd.to_numeric(df_landing[col], errors='coerce')

        # Use nullable integer type for year
        df_landing['reference_year'] = pd.to_numeric(df_landing['reference_year'], errors='coerce').astype('Int64')
        # Parse timestamp, coerce errors, make timezone-aware (UTC)
        df_landing['source_update_timestamp'] = pd.to_datetime(df_landing['source_update_timestamp'], errors='coerce', utc=True)

        # --- Reorder Columns ---
        desired_order = [
            "municipality_id", "municipality_name", "comarca_name",
            "indicator_id", "indicator_name", "reference_year",
            "municipality_value", "comarca_value", "catalunya_value",
            "source_update_timestamp"
        ]
        existing_columns = [col for col in desired_order if col in df_landing.columns]
        df_landing = df_landing[existing_columns]

        # Save to Parquet (Primary Format)
        parquet_output_path = "./data/landing/idescat.parquet"
        try:
            df_landing.to_parquet(parquet_output_path, index=False, engine='pyarrow')
            print(f"Successfully saved landing data to PARQUET: {parquet_output_path}")
            print(f"Total rows saved: {len(df_landing)}")
        except Exception as e:
            print(f"Error saving DataFrame to Parquet file {parquet_output_path}: {e}")


    def run(self, save_csv: bool = False):
        """
        Executes the full landing zone process:
        1. Fetches all municipality IDs.
        2. Fetches chosen indicators for all municipalities in parallel.
        3. Processes and saves the collected data to Parquet (and optionally CSV).
        """
        print(f"--- Running Idescat Landing Zone (Indicators: {len(self.chosen_indicators)}) ---")

        municipality_ids = self._get_all_municipality_ids()
        if not municipality_ids:
            print("Aborting run: Could not retrieve municipality IDs.")
            return

        total_municipalities = len(municipality_ids)
        print(f"\nFetching indicators for {total_municipalities} municipalities using up to {self.max_workers} parallel workers...")

        all_data_for_landing: List[Dict[str, Any]] = []
        futures = []

        # --- Parallel Fetching ---
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for mun_id in municipality_ids:
                # Submit tasks using the instance method
                future = executor.submit(self._fetch_indicators_for_municipality, mun_id)
                futures.append(future)

            print("Fetching data progress:")
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=total_municipalities, unit="municipality"):
                try:
                    result = future.result() # Get results from completed future
                    if result: # Check if any data was returned
                        all_data_for_landing.extend(result)
                except Exception as exc:
                     # Catch potential errors during future.result() itself
                    print(f'\nCaught exception processing a future: {exc}')
        # --- End Parallel Fetching ---

        # --- Process and Save ---
        self._process_and_save_data(all_data_for_landing, save_csv)

        print("\n--- Idescat Landing Zone Task Complete ---")