import requests
import time
import logging
import os
from typing import List, Dict, Any, Optional

# Import Spark specific components
try:
    from pyspark.sql import SparkSession, Row
    from pyspark.sql.utils import AnalysisException
    from pyspark.errors import PySparkException
except ImportError:
    print("---------------------------------------------------------------------")
    print("ERROR: PySpark is not installed or cannot be found.")
    print("Please install PySpark using 'pip install pyspark'")
    print("Ensure Spark is correctly configured in your environment (SPARK_HOME).")
    print("---------------------------------------------------------------------")
    # You might want to exit here depending on how the script is run
    # import sys
    # sys.exit(1)
    # For now, let the rest of the script potentially fail later if Spark is used
    SparkSession = None # Define as None to avoid NameError later if not imported


# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LloguerError(Exception):
    """
    Custom base exception for errors specific to the Lloguer data collection
    and processing tasks. Helps distinguish application-specific errors from
    generic Python or library errors.
    """
    def __init__(self, message: str, underlying_exception: Optional[Exception] = None):
        """
        Initializes the LloguerError.

        Args:
            message (str): A descriptive error message.
            underlying_exception (Optional[Exception]): The original exception that
                                                      caused this error, if any.
                                                      Useful for debugging.
        """
        super().__init__(message)
        self.underlying_exception = underlying_exception
        logger.debug(f"LloguerError created: {message} (Underlying: {type(underlying_exception).__name__ if underlying_exception else 'None'})")

    def __str__(self) -> str:
        """Provides a string representation including the underlying exception type."""
        base_message = super().__str__()
        if self.underlying_exception:
            return f"{base_message} (Caused by: {type(self.underlying_exception).__name__})"
        return base_message


class Lloguer:
    """
    Collects data from a paginated API endpoint ('Lloguer' - Rent data)
    and saves it to a Parquet file using Apache Spark.

    Handles periodic execution by supporting Spark's 'overwrite' and 'append' modes.
    """

    def __init__(self,
                 api_url: str ,
                 output_parquet_path: str,
                 request_limit: int,
                 request_delay: float,
                 timeout: int):
        """
        Initializes the Lloguer data collector.

        Args:
            api_url (str): The base URL of the API endpoint.
            output_parquet_path (str): The HDFS/S3/local path for the output Parquet data.
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
        # Spark handles paths (local, hdfs://, s3a:// etc.)
        self.output_parquet_path = output_parquet_path
        self.request_limit = request_limit
        self.request_delay = request_delay
        self.timeout = timeout

        logger.info(f"Lloguer collector initialized for API: {self.api_url}")
        logger.info(f"Output path (for Spark): {self.output_parquet_path}")
        logger.info(f"Request limit: {self.request_limit}, Delay: {self.request_delay}s")

    def _fetch_data_page(self, offset: int) -> Optional[List[Dict[str, Any]]]:
        """Fetches a single page of data from the API."""
        params = {
            "$limit": self.request_limit,
            "$offset": offset
        }
        try:
            response = requests.get(self.api_url, params=params, timeout=self.timeout)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.Timeout as e:
            msg = f"Request timed out after {self.timeout} seconds for offset {offset}."
            logger.error(msg)
            raise LloguerError(msg, underlying_exception=e) from e
        except requests.exceptions.HTTPError as e:
            msg = f"HTTP error occurred: {e} (Status: {response.status_code}) for offset {offset}"
            logger.error(msg)
            raise LloguerError(msg, underlying_exception=e) from e
        except requests.exceptions.RequestException as e:
            msg = f"Network error during request for offset {offset}: {e}"
            logger.error(msg)
            raise LloguerError(msg, underlying_exception=e) from e
        except ValueError as e: # Includes json.JSONDecodeError
             msg = f"Error decoding JSON response for offset {offset}: {e}"
             logger.error(msg)
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
        logger.info("Starting data fetch process...")

        while True:
            logger.debug(f"Fetching data page with offset: {offset}")
            try:
                data_page = self._fetch_data_page(offset)
            except LloguerError as e:
                # Log the error but re-raise it as the fetch failed
                logger.error(f"Failed to fetch page at offset {offset}. Aborting fetch. Error: {e}")
                raise # Re-raise the specific LloguerError

            if not data_page:
                logger.info("No more data received from API. Fetch complete.")
                break

            page_record_count = len(data_page)
            all_records.extend(data_page)
            logger.info(f"Fetched {page_record_count} records (offset={offset}). Total records so far: {len(all_records)}")

            # Stop if we received fewer records than the limit, indicating the last page
            if page_record_count < self.request_limit:
                 logger.info("Received fewer records than limit, assuming end of data.")
                 break

            offset += self.request_limit

            # Respectful delay
            if self.request_delay > 0:
                logger.debug(f"Waiting for {self.request_delay} seconds before next request.")
                time.sleep(self.request_delay)

        logger.info(f"Finished data fetch. Total records downloaded: {len(all_records)}")
        return all_records

    def save_to_spark_parquet(self, spark: SparkSession, data: List[Dict[str, Any]], mode: str = 'overwrite'): # type: ignore
        """
        Saves the collected data to a Parquet file/directory using Spark.

        Args:
            spark (SparkSession): The active SparkSession.
            data (List[Dict[str, Any]]): The data to save (list of dictionaries).
            mode (str): The Spark save mode. Typically 'overwrite' or 'append'.
                        Other options include 'ignore', 'errorifexists'.

        Raises:
            ValueError: If the mode is invalid or data is empty/None, or SparkSession is missing.
            LloguerError: If any Spark-related error occurs during saving.
            PySparkException: Can be raised from Spark operations.
        """
        if SparkSession is None:
             raise LloguerError("PySpark is not available. Cannot save data.")
        if not spark:
            raise ValueError("A valid SparkSession must be provided.")
        if not data:
            logger.warning("No data provided to save. Skipping Spark Parquet save.")
            return
        if not isinstance(spark, SparkSession):
             raise ValueError("The provided 'spark' argument is not a SparkSession instance.")

        valid_modes = ['overwrite', 'append', 'ignore', 'error', 'errorifexists']
        if mode not in valid_modes:
            raise ValueError(f"Invalid Spark save mode '{mode}'. Choose from: {valid_modes}")

        logger.info(f"Preparing to save {len(data)} records using Spark to {self.output_parquet_path} in '{mode}' mode.")

        try:
            # Convert Python list of dicts to Spark DataFrame
            # Spark will attempt to infer the schema.
            # For production, providing an explicit schema is more robust.
            # e.g., using spark.createDataFrame(data, schema=my_schema)
            # Consider potential performance impact for very large Python lists.
            start_time = time.time()
            logger.debug("Creating Spark DataFrame from fetched data...")
            # Using Row can sometimes help with schema inference consistency
            # spark_df = spark.createDataFrame(Row(**d) for d in data)
            spark_df = spark.createDataFrame(data) # Simpler approach

            # Optional: Cache if the DataFrame is reused, but not needed here
            # spark_df.cache()
            logger.debug(f"Spark DataFrame created in {time.time() - start_time:.2f} seconds. Schema:")
            spark_df.printSchema() # Log the inferred schema for debugging

            if spark_df.isEmpty():
                 logger.warning("Created Spark DataFrame is empty. Skipping Parquet save.")
                 return

            # Write using Spark DataFrameWriter
            logger.info(f"Writing Spark DataFrame ({spark_df.count()} rows) to {self.output_parquet_path}...")
            write_start_time = time.time()
            spark_df.write.mode(mode).parquet(self.output_parquet_path)
            logger.info(f"Successfully saved data using Spark in {time.time() - write_start_time:.2f} seconds.")

            # Optional: Unpersist if cached
            # spark_df.unpersist()

        except AnalysisException as e:
            # Errors during DataFrame analysis (e.g., path not found for append source)
            msg = f"Spark Analysis Exception during save to {self.output_parquet_path}: {e}"
            logger.error(msg)
            raise LloguerError(msg, underlying_exception=e) from e
        except PySparkException as e:
             # Catch other general PySpark errors during DataFrame creation or writing
             msg = f"PySpark error during save operation to {self.output_parquet_path}: {e}"
             logger.error(msg)
             raise LloguerError(msg, underlying_exception=e) from e
        except Exception as e: # Catch any other unexpected errors
            msg = f"An unexpected error occurred during Spark save to {self.output_parquet_path}: {e}"
            logger.exception(msg) # Use exception to log traceback
            raise LloguerError(msg, underlying_exception=e) from e

    def run(self, spark: SparkSession, save_mode: str = 'overwrite') -> None: # type: ignore
        """
        Executes the full data collection and saving pipeline using Spark.

        Args:
            spark (SparkSession): The active SparkSession.
            save_mode (str): The Spark mode for saving the data ('overwrite', 'append', etc.).
        """
        if SparkSession is None:
             logger.error("Cannot run: PySpark is not available.")
             return
        if not spark or not isinstance(spark, SparkSession):
             logger.error("Cannot run: A valid SparkSession must be provided.")
             return

        logger.info(f"Starting Lloguer data collection run (Save Mode: {save_mode})...")
        try:
            fetched_data = self.fetch_all_data()
            if fetched_data: # Only save if data was actually fetched
                 self.save_to_spark_parquet(spark, fetched_data, mode=save_mode)
            else:
                 logger.info("No data fetched from API, skipping Spark save.")
            logger.info("Lloguer data collection run finished successfully.")
        except LloguerError as e:
            # Log the specific error from our application logic
            logger.error(f"Lloguer data collection run failed: {e}")
            # Depending on the orchestration, you might want to raise it further
            # raise e
        except Exception as e:
            # Catch unexpected errors during the run
            logger.exception(f"An unexpected critical error occurred during the run: {e}")
            # Depending on the severity, might raise LloguerError or let it propagate
            # raise LloguerError("Critical unexpected error during run", underlying_exception=e) from e


# --- Example Usage ---
if __name__ == "__main__":

    if SparkSession is None:
        print("Cannot execute main block because PySpark is not available.")
    else:
        spark = None # Initialize spark variable
        try:
            # --- Initialize Spark Session ---
            # Adjust master URL and configurations as needed for your environment
            # local[*] uses all available cores locally
            spark = SparkSession.builder.appName("LloguerDataCollector").master("local[*]").config("spark.sql.parquet.compression.codec", "snappy").getOrCreate()

            logger.info(f"Spark Session created. Version: {spark.version}")
            sc = spark.sparkContext
            logger.info(f"Spark Context Web UI: {sc.uiWebUrl}")

            # --- Define Output Path ---
            # Use a local path for this example. Change for HDFS/S3 etc.
            # Spark will create a directory here, not a single file.
            output_dir = "lloguer_data_spark.parquet"


            # --- Scenario 1: First time run or overwrite ---
            logger.info(f"\n--- Running Scenario 1: Overwrite mode to {output_dir} ---")
            collector_overwrite = Lloguer(
                output_parquet_path=output_dir,
                request_limit=500 # Smaller limit for faster testing
            )
            collector_overwrite.run(spark, save_mode='overwrite')

            # Verification using Spark
            try:
                logger.info("Verifying Scenario 1 output...")
                df_check1 = spark.read.parquet(output_dir)
                count1 = df_check1.count()
                logger.info(f"Verification S1: Spark read successful. Found {count1} records.")
                # df_check1.show(5, truncate=False) # Show some data
            except Exception as read_err:
                 logger.error(f"Verification S1 failed: Error reading Parquet with Spark: {read_err}")


            # --- Scenario 2: Subsequent run with append ---
            # IMPORTANT: Spark's 'append' adds data. If the API always returns the *full*
            # dataset, running append will duplicate data. Use 'append' only if the API
            # provides *new* delta records or if duplication is acceptable/handled later.
            # For this example, we demonstrate append assuming new data might exist.
            logger.info(f"\n--- Running Scenario 2: Append mode to {output_dir} ---")
            collector_append = Lloguer(
                output_parquet_path=output_dir,
                request_limit=500
            )
            collector_append.run(spark, save_mode='append')

            # Verification using Spark
            try:
                logger.info("Verifying Scenario 2 output...")
                df_check2 = spark.read.parquet(output_dir)
                count2 = df_check2.count()
                logger.info(f"Verification S2: Spark read successful. Found {count2} records.")
                # Expect count2 to be roughly 2 * count1 if API data is static
                # Or count1 + new_records if API provides deltas.
                if count1 > 0 : # Avoid division by zero if first run failed
                     logger.info(f"Record count changed from {count1} to {count2} (Ratio: {count2/count1:.2f})")

            except Exception as read_err:
                 logger.error(f"Verification S2 failed: Error reading Parquet with Spark: {read_err}")

        except LloguerError as app_err:
            logger.error(f"Application error during execution: {app_err}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred in the main block: {e}")
        finally:
            # --- Stop Spark Session ---
            if spark:
                logger.info("Stopping Spark Session.")
                spark.stop()
                logger.info("Spark Session stopped.")