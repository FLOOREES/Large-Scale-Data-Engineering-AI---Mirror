import requests
import pandas as pd
import time
import logging
import os
from typing import List, Dict, Any, Optional
import pyarrow # Ensure pyarrow is available

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Data Collector Class ---
class Lloguer:
    """
    Collects paginated rent data from an API and saves it to a Parquet file,
    overwriting any existing file. Uses Pandas and PyArrow.
    Relies on standard exceptions for error handling.
    """

    def __init__(self,
                 api_url: str ,
                 output_parquet_path: str,
                 request_limit: int = 1000,
                 request_delay: float = 0.5,
                 timeout: int = 30):
        """
        Initializes the Lloguer data collector.

        Args:
            api_url (str): The base URL of the API endpoint.
            output_parquet_path (str): Path for the output Parquet file.
            request_limit (int): Records per API call ($limit).
            request_delay (float): Delay between API calls (seconds).
            timeout (int): HTTP request timeout (seconds).
        """
        if not api_url or not output_parquet_path:
            raise ValueError("API URL and output path cannot be empty.")
        if request_limit <= 0 or request_delay < 0 or timeout <= 0:
            raise ValueError("Request limit/timeout must be positive. Delay cannot be negative.")

        self.api_url = api_url
        self.output_parquet_path = output_parquet_path
        self.request_limit = request_limit
        self.request_delay = request_delay
        self.timeout = timeout

        logger.info(f"Lloguer initialized:")
        logger.info(f"  API URL: {self.api_url}")
        logger.info(f"  Output Parquet: {self.output_parquet_path}")
        logger.info(f"  Settings: Limit={self.request_limit}, Delay={self.request_delay}s, Timeout={self.timeout}s")

    def _fetch_data_page(self, offset: int) -> List[Dict[str, Any]]:
        """
        Fetches a single page of data. Returns empty list on empty response.
        Raises exceptions on request/processing errors.
        """
        params = {"$limit": self.request_limit, "$offset": offset}
        logger.debug(f"Fetching page: offset={offset}, limit={self.request_limit}")
        try:
            response = requests.get(self.api_url, params=params, timeout=self.timeout)
            response.raise_for_status() # Check for HTTP 4xx/5xx errors

            if not response.content:
                logger.debug(f"Empty response body received for offset {offset}.")
                return [] # End of data for this page

            return response.json()

        # Catch specific requests exceptions first for potentially better context
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timed out (>{self.timeout}s) for offset {offset}", exc_info=True)
            raise # Re-raise the original exception
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error {e.response.status_code} for offset {offset}. Response: '{e.response.text[:200]}...'", exc_info=True)
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during request for offset {offset}: {e}", exc_info=True)
            raise
        except ValueError as e: # Catches JSONDecodeError
            logger.error(f"Failed to decode JSON response for offset {offset}", exc_info=True)
            raise

    def fetch_all_data(self) -> List[Dict[str, Any]]:
        """
        Fetches all data from the API using pagination.
        Raises exceptions if any page fetch fails.
        """
        all_records = []
        offset = 0
        logger.info("Starting full data fetch...")

        while True:
            # Error handling moved to _fetch_data_page, exceptions will propagate up
            data_page = self._fetch_data_page(offset)

            if not data_page: # Empty list signifies end of data
                logger.info(f"Received empty page at offset {offset}. Assuming end of data.")
                break
            else:
                page_record_count = len(data_page)
                all_records.extend(data_page)
                logger.info(f"Fetched {page_record_count} records (offset={offset}). Total so far: {len(all_records)}")

                if page_record_count < self.request_limit:
                    logger.info("Received fewer records than limit. Assuming fetch complete.")
                    break

                offset += self.request_limit
                if self.request_delay > 0:
                    logger.debug(f"Waiting {self.request_delay}s...")
                    time.sleep(self.request_delay)

        logger.info(f"Finished data fetch. Total records downloaded: {len(all_records)}")
        return all_records

    def save_to_parquet(self, data: List[Dict[str, Any]]):
        """
        Saves the provided data to a Parquet file, overwriting if it exists.
        Raises exceptions on errors.
        """
        if data is None:
            # Let caller handle None if needed, or raise error here
            raise ValueError("Input data cannot be None for saving.")
        if not data:
            logger.warning("No data provided (empty list). Skipping Parquet save.")
            return

        logger.info(f"Preparing to save {len(data)} records to {self.output_parquet_path} (overwrite mode).")

        try:
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning("Created DataFrame is empty. Skipping Parquet save.")
                return

            output_dir = os.path.dirname(self.output_parquet_path)
            if output_dir and not os.path.exists(output_dir):
                 logger.info(f"Creating output directory: {output_dir}")
                 os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Writing {len(df)} records using engine 'pyarrow'...")
            start_time = time.time()
            df.to_parquet(self.output_parquet_path, engine='pyarrow', index=False)
            end_time = time.time()
            logger.info(f"Successfully saved data to {self.output_parquet_path} in {end_time - start_time:.2f} seconds.")

        except (ValueError, TypeError) as e:
            logger.error(f"Error processing data for Parquet saving: {e}", exc_info=True)
            raise # Re-raise standard error
        except (IOError, OSError, pyarrow.lib.ArrowException) as e:
            logger.error(f"Error writing Parquet file to {self.output_parquet_path}: {e}", exc_info=True)
            raise
        except Exception as e: # Catch any other unexpected errors during save
             logger.error(f"An unexpected error occurred during Parquet save: {e}", exc_info=True)
             raise

    def run(self):
        """
        Executes the full pipeline: fetch all data and save (overwrite).
        Logs results or errors. Does not return a value.
        Raises exceptions on failure.
        """
        logger.info(f"--- Starting Lloguer Run ---")
        try:
            fetched_data = self.fetch_all_data()

            if fetched_data:
                 self.save_to_parquet(fetched_data)
                 logger.info(f"--- Lloguer Run Completed Successfully. Saved {len(fetched_data)} records. ---")
            else:
                 logger.info("--- Lloguer Run Completed: No data fetched or saved. ---")

        except Exception as e:
            # Specific errors logged in fetch/save methods. This catches the failure.
            logger.error(f"--- Lloguer Run Failed. See previous logs for error details. ---", exc_info=False) # Avoid duplicate traceback if already logged
            raise # Re-raise the exception to signal failure to the caller


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Configuration ---
    API_URL = "https://analisi.transparenciacatalunya.cat/resource/qww9-bvhh.json"
    OUTPUT_DIR = "./data/landing/"
    OUTPUT_FILENAME = "lloguer_catalunya.parquet"
    OUTPUT_PARQUET_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    # API Parameters
    REQUEST_LIMIT = 1000
    REQUEST_DELAY = 0.5
    TIMEOUT = 60
    logger.info("--- Initializing Lloguer Data Collection ---")

    try:
        # Initialize Collector
        collector = Lloguer(api_url=API_URL,output_parquet_path=OUTPUT_PARQUET_PATH,request_limit=REQUEST_LIMIT,request_delay=REQUEST_DELAY,timeout=TIMEOUT)
        collector.run()
        logger.info("--- Script finished successfully. ---")


    except ValueError as config_err:
        # Error during initialization
        logger.error(f"Initialization failed due to configuration error: {config_err}", exc_info=True)
    except Exception as e:
        # Catch any exception raised during collector.run()
        logger.error(f"Script execution failed during the run phase.", exc_info=True) # Log exception info here

    finally:
        logger.info("--- Lloguer Script Execution Attempt Finished ---")