import requests
import pandas as pd # Import pandas
import time
import logging
import os
from typing import List, Dict, Any, Optional
import pyarrow # Explicit import sometimes helps environments

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Custom Error Classes ---
class LloguerError(Exception):
    """
    Custom base exception for errors specific to the Lloguer data collection
    and processing tasks. Helps distinguish application-specific errors from
    generic Python or library errors.
    """
    def __init__(self, message: str, underlying_exception: Optional[Exception] = None):
        super().__init__(message)
        self.underlying_exception = underlying_exception
        # Log creation slightly differently for clarity
        log_msg = f"LloguerError: {message}"
        if underlying_exception:
            log_msg += f" (Caused by: {type(underlying_exception).__name__}: {underlying_exception})"
        logger.debug(log_msg) # Keep debug level for creation event

    def __str__(self) -> str:
        base_message = super().__str__()
        if self.underlying_exception:
            # Include exception details in __str__ for easier debugging when caught
            return f"{base_message} (Caused by: {type(self.underlying_exception).__name__}: {self.underlying_exception})"
        return base_message

class ReaderError(Exception):
    """
    Custom exception for errors specific to the ParquetReader class.
    Indicates issues during file reading or validation.
    """
    def __init__(self, message: str, underlying_exception: Optional[Exception] = None):
        """
        Initializes the ReaderError.

        Args:
            message (str): A descriptive error message.
            underlying_exception (Optional[Exception]): The original exception
                                                      that triggered this error, if any.
        """
        super().__init__(message)
        self.underlying_exception = underlying_exception
        # Log creation slightly differently for clarity
        log_msg = f"ReaderError: {message}"
        if underlying_exception:
            log_msg += f" (Caused by: {type(underlying_exception).__name__}: {underlying_exception})"
        logger.debug(log_msg) # Keep debug level for creation event


    def __str__(self) -> str:
        base_message = super().__str__()
        if self.underlying_exception:
             # Include exception details in __str__ for easier debugging when caught
            return f"{base_message} (Caused by: {type(self.underlying_exception).__name__}: {self.underlying_exception})"
        return base_message

# --- Data Collector Class ---
class Lloguer:
    """
    Collects data from a paginated API endpoint ('Lloguer' - Rent data)
    and saves it to a Parquet file using Pandas.

    Handles periodic execution by supporting 'overwrite' and 'append' modes.
    Requires `pandas` and `pyarrow` libraries.
    """

    def __init__(self,
                 api_url: str ,
                 output_parquet_path: str,
                 request_limit: int,
                 request_delay: float,
                 timeout: int,
                 logger: Optional[logging.Logger] = None):
        """
        Initializes the Lloguer data collector.

        Args:
            api_url (str): The base URL of the API endpoint.
            output_parquet_path (str): The local file system path for the output Parquet file.
            request_limit (int): The number of records to request per API call ($limit).
            request_delay (float): The delay in seconds between consecutive API calls.
            timeout (int): Timeout in seconds for the HTTP request.
        """
        if not api_url:
            raise ValueError("API URL cannot be empty.")
        if not output_parquet_path:
            raise ValueError("Output Parquet path cannot be empty.")
        if request_limit <= 0:
            raise ValueError("Request limit must be positive.")
        if request_delay < 0:
            raise ValueError("Request delay cannot be negative.")
        if timeout <= 0:
            raise ValueError("Timeout must be positive.")

        self.api_url = api_url
        self.output_parquet_path = output_parquet_path
        self.request_limit = request_limit
        self.request_delay = request_delay
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        self.last_fetch_count = 0 # Store count of last fetch

        self.logger.info(f"Lloguer collector initialized for API: {self.api_url}")
        self.logger.info(f"Output path (for Parquet): {self.output_parquet_path}") # Updated log
        self.logger.info(f"Request limit: {self.request_limit}, Delay: {self.request_delay}s")

    def _fetch_data_page(self, offset: int) -> Optional[List[Dict[str, Any]]]:
        """Fetches a single page of data from the API."""
        params = {
            "$limit": self.request_limit,
            "$offset": offset
        }
        try:
            response = requests.get(self.api_url, params=params, timeout=self.timeout)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            # Check for empty or non-JSON response before decoding
            if not response.content:
                logger.warning(f"Received empty response body for offset {offset}. Assuming end of data.")
                return None
            return response.json()
        except requests.exceptions.Timeout as e:
            msg = f"Request timed out after {self.timeout} seconds for offset {offset}."
            self.logger.error(msg)
            raise LloguerError(msg, underlying_exception=e) from e
        except requests.exceptions.HTTPError as e:
            # Log response body if available and not too large for debugging
            response_body_excerpt = ""
            if e.response is not None and e.response.content:
                 response_body_excerpt = e.response.content[:200].decode('utf-8', errors='ignore') # Show first 200 chars
            msg = f"HTTP error occurred: {e} (Status: {e.response.status_code}) for offset {offset}. Response excerpt: '{response_body_excerpt}...'"
            self.logger.error(msg)
            raise LloguerError(msg, underlying_exception=e) from e
        except requests.exceptions.RequestException as e:
            msg = f"Network error during request for offset {offset}: {e}"
            self.logger.error(msg)
            raise LloguerError(msg, underlying_exception=e) from e
        except ValueError as e: # Includes json.JSONDecodeError
             msg = f"Error decoding JSON response for offset {offset}. Response might not be valid JSON. Error: {e}"
             self.logger.error(msg)
             # Optionally log response text here too for debugging JSON errors
             try:
                 self.logger.debug(f"Response text causing JSON decode error (offset {offset}): {response.text[:500]}...") # Log beginning of text
             except Exception: # Ignore errors during logging the problematic text
                 pass
             raise LloguerError(msg, underlying_exception=e) from e

    def fetch_all_data(self) -> List[Dict[str, Any]]:
        """
        Fetches all data from the API endpoint by handling pagination.

        Returns:
            List[Dict[str, Any]]: A list containing all records fetched from the API.

        Raises:
            LloguerError: If any error occurs during fetching that prevents completion.
        """
        all_records = []
        offset = 0
        self.logger.info("Starting data fetch process...")

        while True:
            self.logger.debug(f"Fetching data page with offset: {offset}")
            try:
                data_page = self._fetch_data_page(offset)
            except LloguerError as e:
                self.logger.error(f"Failed to fetch page at offset {offset}. Aborting fetch. Error: {e}")
                raise # Re-raise the specific LloguerError

            # Handle case where _fetch_data_page returns None (e.g., empty response body)
            if data_page is None:
                self.logger.info(f"Received None or empty response for page at offset {offset}. Assuming end of data.")
                break

            # Handle case where API returns empty list explicitly
            if not data_page:
                self.logger.info(f"Received empty list from API at offset {offset}. Fetch complete.")
                break

            page_record_count = len(data_page)
            all_records.extend(data_page)
            self.logger.info(f"Fetched {page_record_count} records (offset={offset}). Total records so far: {len(all_records)}")

            # Check if the last page was received
            if page_record_count < self.request_limit:
                 self.logger.info(f"Received {page_record_count} records, which is less than the limit ({self.request_limit}). Assuming end of data.")
                 break

            offset += self.request_limit

            if self.request_delay > 0:
                self.logger.debug(f"Waiting for {self.request_delay} seconds before next request.")
                time.sleep(self.request_delay)

        self.last_fetch_count = len(all_records) # Store the count
        self.logger.info(f"Finished data fetch. Total records downloaded: {self.last_fetch_count}")
        return all_records

    def save_to_parquet(self, data: List[Dict[str, Any]], mode: str = 'overwrite'):
        """
        Saves the collected data to a Parquet file using Pandas.

        Args:
            data (List[Dict[str, Any]]): The data to save (list of dictionaries).
            mode (str): The save mode. Options:
                        'overwrite': Overwrites the file if it exists.
                        'append': Reads the existing file (if any), appends new data,
                                  and overwrites the file with the combined data.

        Raises:
            ValueError: If the mode is invalid or data is empty/None.
            LloguerError: If any error occurs during saving (I/O, Pandas, PyArrow).
            FileNotFoundError: If mode is 'append' and the file needs reading but doesn't exist (caught internally).
        """
        if data is None: # Explicitly check for None
            self.logger.warning("Input data is None. Skipping Parquet save.")
            return
        if not data:
            self.logger.warning("No data provided (empty list). Skipping Parquet save.")
            return

        if mode not in ['overwrite', 'append']:
            raise ValueError(f"Invalid save mode '{mode}'. Choose 'overwrite' or 'append'.")

        self.logger.info(f"Preparing to save {len(data)} records using Pandas to {self.output_parquet_path} in '{mode}' mode.")

        try:
            # --- DataFrame Creation ---
            self.logger.debug("Converting list of dictionaries to Pandas DataFrame...")
            new_df = pd.DataFrame(data)
            self.logger.debug(f"DataFrame created with shape: {new_df.shape}")

            if new_df.empty:
                 self.logger.warning("DataFrame created from data is empty. Skipping Parquet save.")
                 return

            final_df = new_df
            # --- Handle Append Mode ---
            if mode == 'append':
                if os.path.exists(self.output_parquet_path):
                    self.logger.info(f"Append mode: Reading existing data from {self.output_parquet_path}")
                    try:
                        existing_df = pd.read_parquet(self.output_parquet_path, engine='pyarrow')
                        self.logger.info(f"Read {len(existing_df)} existing records. Columns: {existing_df.columns.tolist()}")
                        self.logger.info(f"New data has {len(new_df)} records. Columns: {new_df.columns.tolist()}")

                        # Optional: Check for schema compatibility before concatenating
                        if not existing_df.columns.equals(new_df.columns):
                             self.logger.warning(f"Schema mismatch between existing file and new data. Appending may lead to unexpected results or errors.")
                             # Depending on requirements, you might want to:
                             # 1. Raise an error: raise LloguerError("Schema mismatch during append.")
                             # 2. Try to align columns (more complex)
                             # 3. Proceed with caution (current behavior of concat)

                        # Concatenate old and new data
                        final_df = pd.concat([existing_df, new_df], ignore_index=True)
                        self.logger.info(f"Concatenated data. New total records: {len(final_df)}")
                    except FileNotFoundError:
                         # Should not happen due to os.path.exists, but handle defensively
                         self.logger.warning(f"Append mode: File {self.output_parquet_path} disappeared between check and read. Creating new file.")
                    except (pd.errors.EmptyDataError, pyarrow.lib.ArrowInvalid) as read_err:
                        # Catch specific errors related to empty or corrupted files during read
                        msg = f"Error reading existing (potentially empty or corrupted) Parquet file {self.output_parquet_path} for append: {read_err}"
                        self.logger.error(msg)
                        raise LloguerError(msg, underlying_exception=read_err) from read_err
                    except Exception as read_err: # Catch other broad exceptions during read
                        msg = f"Unexpected error reading existing Parquet file {self.output_parquet_path} for append: {read_err}"
                        self.logger.exception(msg) # Log with traceback
                        raise LloguerError(msg, underlying_exception=read_err) from read_err
                else:
                    self.logger.info(f"Append mode: File {self.output_parquet_path} not found. Will create a new file.")

            # --- Ensure Directory Exists ---
            output_dir = os.path.dirname(self.output_parquet_path)
            if output_dir and not os.path.exists(output_dir):
                 self.logger.info(f"Creating output directory: {output_dir}")
                 os.makedirs(output_dir, exist_ok=True) # exist_ok=True prevents error if dir exists

            # --- Write Parquet File ---
            self.logger.info(f"Writing {len(final_df)} records to {self.output_parquet_path} (Engine: pyarrow)...")
            start_time = time.time()
            # Use pyarrow engine, specify index=False unless the index is meaningful
            final_df.to_parquet(self.output_parquet_path, engine='pyarrow', index=False)
            end_time = time.time()
            self.logger.info(f"Successfully saved data to {self.output_parquet_path} in {end_time - start_time:.2f} seconds.")

        except (ValueError, TypeError) as val_err: # Catch Pandas/PyArrow type/value errors
             msg = f"Data type or value error during DataFrame processing or Parquet saving: {val_err}"
             self.logger.error(msg)
             raise LloguerError(msg, underlying_exception=val_err) from val_err
        except IOError as io_err: # Catch file system errors during write
            msg = f"IOError saving Parquet file to {self.output_parquet_path}: {io_err}"
            self.logger.error(msg)
            raise LloguerError(msg, underlying_exception=io_err) from io_err
        except pyarrow.lib.ArrowException as arrow_err: # Catch specific PyArrow errors
            msg = f"PyArrow error during Parquet save to {self.output_parquet_path}: {arrow_err}"
            self.logger.error(msg)
            raise LloguerError(msg, underlying_exception=arrow_err) from arrow_err
        except Exception as e: # Catch other potential errors
            msg = f"An unexpected error occurred during Parquet save operation: {e}"
            self.logger.exception(msg) # Use exception to log traceback
            raise LloguerError(msg, underlying_exception=e) from e

    def run(self, save_mode: str = 'overwrite') -> Optional[int]:
        """
        Executes the full data collection and saving pipeline using Pandas.

        Args:
            save_mode (str): The mode for saving the data ('overwrite' or 'append').

        Returns:
            Optional[int]: The number of records fetched, or None if fetching failed or no data was fetched.
        """
        self.logger.info(f"Starting Lloguer data collection run (Save Mode: {save_mode})...")
        self.last_fetch_count = 0 # Reset count for the run
        try:
            fetched_data = self.fetch_all_data()
            if fetched_data: # Only save if data was actually fetched
                 self.save_to_parquet(fetched_data, mode=save_mode)
            elif fetched_data is None: # Fetch failed or returned None explicitly
                 self.logger.warning("Fetching process did not return data. Skipping save.")
                 return None # Indicate fetch issue
            else: # fetched_data is an empty list
                 self.logger.info("No data fetched from API (API returned empty list). Skipping save.")
            self.logger.info("Lloguer data collection run finished successfully.")
            return self.last_fetch_count # Return the number of records fetched in this run
        except LloguerError as e:
            self.logger.error(f"Lloguer data collection run failed: {e}")
            return None # Indicate failure
        except Exception as e:
            # Catch any unexpected error during the run sequence
            self.logger.exception(f"An unexpected critical error occurred during the run: {e}")
            return None # Indicate failure


# --- Parquet Reader Class ---
class ParquetReader:
    """
    A simple class to read a Parquet file using Pandas and PyArrow.
    """
    def __init__(self, parquet_path: str, logger: Optional[logging.Logger] = None):
        """
        Initializes the ParquetReader.

        Args:
            parquet_path (str): The path to the Parquet file to be read.
            logger (Optional[logging.Logger]): Logger instance.
        """
        if not parquet_path:
            raise ValueError("Parquet path cannot be empty.")
        self.parquet_path = parquet_path
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"ParquetReader initialized for path: {self.parquet_path}")

    def read_data(self) -> pd.DataFrame:
        """
        Reads the Parquet file specified during initialization using Pandas and PyArrow.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the data from the Parquet file.

        Raises:
            FileNotFoundError: If the parquet file does not exist at the path.
            ReaderError: If the file cannot be read due to corruption, permission issues,
                         or other PyArrow/Pandas errors during reading.
        """
        self.logger.info(f"Attempting to read Parquet file: {self.parquet_path}")
        if not os.path.exists(self.parquet_path):
            msg = f"Parquet file not found at path: {self.parquet_path}"
            self.logger.error(msg)
            # Raise built-in FileNotFoundError, as it's the standard exception for this
            raise FileNotFoundError(msg)

        try:
            start_time = time.time()
            # Explicitly use pyarrow engine for consistency with saving
            df = pd.read_parquet(self.parquet_path, engine='pyarrow')
            end_time = time.time()
            self.logger.info(f"Successfully read {len(df)} records from {self.parquet_path} in {end_time - start_time:.2f} seconds.")
            # Basic validation: Check if DataFrame is empty after reading a non-empty file
            # Note: A valid parquet file *can* be empty, so this is just a sanity check.
            if df.empty and os.path.getsize(self.parquet_path) > 0:
                 self.logger.warning(f"Read an empty DataFrame from a non-empty Parquet file: {self.parquet_path}. File size: {os.path.getsize(self.parquet_path)} bytes.")
                 # Depending on strictness, could raise ReaderError here
                 # raise ReaderError(f"Read empty DataFrame from non-empty file: {self.parquet_path}")

            return df
        except (pd.errors.EmptyDataError, pyarrow.lib.ArrowInvalid, pyarrow.lib.ArrowIOError) as e:
            # Catch specific errors indicating corrupted file, IO problems during read, etc.
            msg = f"Failed to read Parquet file {self.parquet_path} (potential corruption or I/O issue): {e}"
            self.logger.error(msg)
            raise ReaderError(msg, underlying_exception=e) from e
        except MemoryError as e:
             msg = f"MemoryError while reading Parquet file {self.parquet_path}. File might be too large for available memory."
             self.logger.error(msg)
             raise ReaderError(msg, underlying_exception=e) from e
        except Exception as e:
            # Catch any other unexpected exceptions during the read process
            msg = f"An unexpected error occurred while reading Parquet file {self.parquet_path}: {e}"
            self.logger.exception(msg) # Log traceback for unexpected errors
            raise ReaderError(msg, underlying_exception=e) from e


# --- Example Usage with Enhanced Verification ---
if __name__ == "__main__":

    # --- Define Configuration ---
    API_URL = "https://analisi.transparenciacatalunya.cat/resource/qww9-bvhh.json"
    # Use a temporary directory for outputs for cleaner testing
    TEST_OUTPUT_DIR = "temp_parquet_test_output"
    OUTPUT_PARQUET = os.path.join(TEST_OUTPUT_DIR, "lloguer_data_test.parquet")
    REQUEST_LIMIT = 500 # Smaller limit for faster testing
    REQUEST_DELAY = 0.1 # Shorter delay for testing
    TIMEOUT = 30      # Slightly longer timeout for potentially slow API

    # Ensure clean state for testing
    if os.path.exists(OUTPUT_PARQUET):
        logger.info(f"Removing existing test file: {OUTPUT_PARQUET}")
        try:
            os.remove(OUTPUT_PARQUET)
        except OSError as e:
             logger.warning(f"Could not remove existing file {OUTPUT_PARQUET}: {e}")
    if os.path.exists(TEST_OUTPUT_DIR) and not os.listdir(TEST_OUTPUT_DIR):
        try:
            os.rmdir(TEST_OUTPUT_DIR) # Remove dir only if empty
        except OSError as e:
             logger.warning(f"Could not remove empty test directory {TEST_OUTPUT_DIR}: {e}")
    elif not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR) # Create dir if it doesn't exist


    count1 = 0 # Initialize count after first run
    fetched_count1 = 0 # Count fetched in first run

    try:
        # --- Scenario 1: First time run (overwrite) ---
        logger.info(f"\n--- Running Scenario 1: Overwrite mode to {OUTPUT_PARQUET} ---")
        collector_overwrite = Lloguer(
            api_url=API_URL,
            output_parquet_path=OUTPUT_PARQUET,
            request_limit=REQUEST_LIMIT,
            request_delay=REQUEST_DELAY,
            timeout=TIMEOUT,
            logger=logger # Pass the main logger
        )
        run1_result = collector_overwrite.run(save_mode='overwrite')

        if run1_result is None:
            logger.error("Scenario 1 failed during Lloguer run. Aborting further tests.")
        else:
            fetched_count1 = run1_result
            logger.info(f"Scenario 1 fetched {fetched_count1} records.")

            # --- Verification S1 using ParquetReader ---
            logger.info("--- Verifying Scenario 1 Output (Overwrite) ---")
            if os.path.exists(OUTPUT_PARQUET):
                try:
                    reader_s1 = ParquetReader(parquet_path=OUTPUT_PARQUET, logger=logger)
                    df_check1 = reader_s1.read_data()
                    count1 = len(df_check1)
                    logger.info(f"Verification S1: ParquetReader successful. Read {count1} records.")

                    # Check if read count matches fetched count
                    if count1 == fetched_count1:
                        logger.info(f"Verification S1: OK - Record count in file ({count1}) matches fetched count ({fetched_count1}).")
                    else:
                        logger.warning(f"Verification S1: MISMATCH - Record count in file ({count1}) does NOT match fetched count ({fetched_count1}).")

                    # Basic integrity check
                    if not df_check1.empty:
                         logger.info("Verification S1: OK - DataFrame read is not empty.")
                    elif count1 == 0:
                         logger.info("Verification S1: OK - DataFrame is empty, matching zero records read/fetched.")
                    else: # Should not happen if count1 > 0
                         logger.error("Verification S1: FAILED - Read count > 0 but DataFrame is empty (unexpected).")

                    # --- Inspect Data (Scenario 1) ---
                    if count1 > 0: # Only inspect if data was actually read
                        logger.info("--- Inspecting Data from Scenario 1 ---")

                        # 1. Show Schema/Info
                        logger.info("\nDataFrame Info (Schema - Scenario 1):")
                        # Use a buffer to capture info() output for logging if needed, or just print
                        print("-------------------- Schema S1 --------------------")
                        df_check1.info()
                        print("---------------------------------------------------")


                        # 2. Show Sample Data
                        logger.info("\nSample Data (First 5 rows - Scenario 1):")
                        print("-------------------- Sample S1 --------------------")
                        # Use to_string() to avoid truncation in console width
                        print(df_check1.head().to_string())
                        print("---------------------------------------------------")

                        # 3. Count Distinct 'nom_territori'
                        territori_col = 'nom_territori' # Define column name once
                        if territori_col in df_check1.columns:
                            distinct_territori_count = df_check1[territori_col].nunique()
                            logger.info(f"\nDistinct '{territori_col}' count (Scenario 1): {distinct_territori_count}")
                            # Optional: print the unique values if not too many
                            # if distinct_territori_count < 20:
                            #    logger.info(f"Unique '{territori_col}' values: {df_check1[territori_col].unique().tolist()}")
                        else:
                            logger.warning(f"\nColumn '{territori_col}' not found in DataFrame for Scenario 1.")
                        logger.info("--- End Inspection (Scenario 1) ---")

                except (FileNotFoundError, ReaderError) as read_err:
                    logger.error(f"Verification S1 FAILED: Error reading Parquet file with ParquetReader: {read_err}")
                except Exception as e:
                     logger.error(f"Verification S1 FAILED: Unexpected error during verification: {e}", exc_info=True)
            else:
                 logger.error(f"Verification S1 FAILED: Parquet file {OUTPUT_PARQUET} not found after run.")


            # --- Scenario 2: Subsequent run (append) ---
            # Only run if Scenario 1 seemed successful (file exists, count >= 0)
            if os.path.exists(OUTPUT_PARQUET): # Check file exists from S1
                logger.info(f"\n--- Running Scenario 2: Append mode to {OUTPUT_PARQUET} ---")
                # Re-use the collector or create a new one
                collector_append = collector_overwrite
                run2_result = collector_append.run(save_mode='append')

                if run2_result is None:
                     logger.error("Scenario 2 failed during Lloguer run.")
                else:
                    fetched_count2 = run2_result
                    logger.info(f"Scenario 2 fetched {fetched_count2} records (to append).")

                    # --- Verification S2 using ParquetReader ---
                    logger.info("--- Verifying Scenario 2 Output (Append) ---")
                    if os.path.exists(OUTPUT_PARQUET):
                        try:
                            reader_s2 = ParquetReader(parquet_path=OUTPUT_PARQUET, logger=logger)
                            df_check2 = reader_s2.read_data()
                            count2 = len(df_check2)
                            logger.info(f"Verification S2: ParquetReader successful. Read {count2} records.")

                            # Check if count increased as expected
                            expected_count2 = count1 + fetched_count2
                            if count2 == expected_count2:
                                logger.info(f"Verification S2: OK - Record count ({count2}) matches expected count after append ({count1} + {fetched_count2} = {expected_count2}).")
                            else:
                                logger.warning(f"Verification S2: MISMATCH - Record count ({count2}) does NOT match expected count ({expected_count2}).")
                                logger.warning(f"  (Count before append was {count1}, records fetched in S2 were {fetched_count2})")

                            # Basic integrity check
                            if not df_check2.empty:
                                logger.info("Verification S2: OK - DataFrame read is not empty.")
                            elif count2 == 0:
                                 logger.info("Verification S2: OK - DataFrame is empty, matching zero total records.")
                            else:
                                 logger.error("Verification S2: FAILED - Read count > 0 but DataFrame is empty (unexpected).")


                            # --- Inspect Data (Scenario 2) ---
                            if count2 > 0: # Only inspect if data was actually read
                                logger.info("--- Inspecting Data from Scenario 2 (After Append) ---")

                                # 1. Show Schema/Info
                                logger.info("\nDataFrame Info (Schema - Scenario 2):")
                                print("-------------------- Schema S2 --------------------")
                                df_check2.info()
                                print("---------------------------------------------------")

                                # 2. Show Sample Data
                                logger.info("\nSample Data (First 5 rows - Scenario 2):")
                                print("-------------------- Sample S2 --------------------")
                                print(df_check2.head().to_string())
                                print("---------------------------------------------------")


                                # 3. Count Distinct 'nom_territori'
                                territori_col = 'nom_territori' # Use same variable name
                                if territori_col in df_check2.columns:
                                    distinct_territori_count_s2 = df_check2[territori_col].nunique()
                                    logger.info(f"\nDistinct '{territori_col}' count (Scenario 2): {distinct_territori_count_s2}")
                                    # Compare with S1 count if available
                                    try:
                                         if distinct_territori_count: # Check if S1 count exists
                                             if distinct_territori_count_s2 == distinct_territori_count:
                                                 logger.info(f"-> Distinct '{territori_col}' count is unchanged from Scenario 1.")
                                             else:
                                                 logger.warning(f"-> Distinct '{territori_col}' count changed from {distinct_territori_count} (S1) to {distinct_territori_count_s2} (S2).")
                                    except NameError: # Handle case where distinct_territori_count wasn't set in S1
                                         pass
                                else:
                                    logger.warning(f"\nColumn '{territori_col}' not found in DataFrame for Scenario 2.")
                                logger.info("--- End Inspection (Scenario 2) ---")


                        except (FileNotFoundError, ReaderError) as read_err:
                             logger.error(f"Verification S2 FAILED: Error reading Parquet file with ParquetReader after append: {read_err}")
                        except Exception as e:
                             logger.error(f"Verification S2 FAILED: Unexpected error during verification: {e}", exc_info=True)
                    else:
                        logger.error(f"Verification S2 FAILED: Parquet file {OUTPUT_PARQUET} not found after append run.")
            else:
                logger.warning("Skipping Scenario 2 because Scenario 1 verification indicated issues or file absence.")


    except LloguerError as app_err:
        logger.error(f"Application error during execution: {app_err}", exc_info=True) # Include traceback for app errors
    except ValueError as val_err:
         logger.error(f"Configuration or Value error: {val_err}", exc_info=True)
    except Exception as e:
        logger.exception(f"An critical unexpected error occurred in the main block: {e}") # Use exception for full traceback

    finally:
        # Optional: Clean up the created parquet file and directory after testing
        # Comment out if you want to inspect the file manually
        logger.info("--- Test Cleanup ---")
        if os.path.exists(OUTPUT_PARQUET):
            try:
                logger.info(f"Removing test file: {OUTPUT_PARQUET}")
                os.remove(OUTPUT_PARQUET)
            except OSError as e:
                logger.warning(f"Could not remove test file {OUTPUT_PARQUET}: {e}")
        if os.path.exists(TEST_OUTPUT_DIR):
             try:
                 # Only remove if empty after file removal
                 if not os.listdir(TEST_OUTPUT_DIR):
                     logger.info(f"Removing empty test directory: {TEST_OUTPUT_DIR}")
                     os.rmdir(TEST_OUTPUT_DIR)
                 else:
                      logger.warning(f"Test directory {TEST_OUTPUT_DIR} not empty, not removing.")
             except OSError as e:
                logger.warning(f"Could not remove test directory {TEST_OUTPUT_DIR}: {e}")
        logger.info("--- End of Script ---")