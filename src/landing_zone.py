# Goal: Periodically collect raw data from APIs and save it efficiently in Parquet format.

# Tools: Python (requests, json, pathlib), optional scheduler (schedule, cron, etc.), pandas, pyarrow.

# Actions: Write collectors per source (meteocat, idescat, solar), parse JSON/CSV, save as Parquet in date-partitioned folders.

from pyspark.sql import SparkSession, Row
from pyspark.sql.utils import AnalysisException
from pyspark.errors import PySparkException
import logging

import os
os.environ["PYSPARK_PYTHON"] = "C:/Users/aflon/OneDrive/Documentos/GitHub/large-scale-data-engineering-ai/.venv/Scripts/python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:/Users/aflon/OneDrive/Documentos/GitHub/large-scale-data-engineering-ai/.venv/Scripts/python.exe"

from landing.lloguer import Lloguer, LloguerError

from config import (LLOGUER_DEFAULT_API_URL, LLOGUER_DEFAULT_REQUEST_LIMIT, LLOGUER_DEFAULT_REQUEST_DELAY_SECONDS, LLOGUER_DEFAULT_TIMEOUT_SECONDS )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

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
            output_dir = "data/landing/lloguer_data_spark.parquet"


            # --- Scenario 1: First time run or overwrite ---
            logger.info(f"\n--- Running Scenario 1: Overwrite mode to {output_dir} ---")
            collector_overwrite = Lloguer(
                api_url=LLOGUER_DEFAULT_API_URL,
                request_delay=LLOGUER_DEFAULT_REQUEST_DELAY_SECONDS,
                timeout=LLOGUER_DEFAULT_TIMEOUT_SECONDS,
                output_parquet_path=output_dir,
                request_limit=LLOGUER_DEFAULT_REQUEST_LIMIT # Smaller limit for faster testing
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
                api_url=LLOGUER_DEFAULT_API_URL,
                request_delay=LLOGUER_DEFAULT_REQUEST_DELAY_SECONDS,
                timeout=LLOGUER_DEFAULT_TIMEOUT_SECONDS,
                output_parquet_path=output_dir,
                request_limit=LLOGUER_DEFAULT_REQUEST_LIMIT # Smaller limit for faster testing
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