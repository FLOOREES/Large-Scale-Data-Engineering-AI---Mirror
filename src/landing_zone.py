# Goal: Periodically collect raw data from APIs and save it efficiently in Parquet format.

# Tools: Python (requests, json, pathlib), optional scheduler (schedule, cron, etc.), pandas, pyarrow.

# Actions: Write collectors per source (meteocat, idescat, solar), parse JSON/CSV, save as Parquet in date-partitioned folders.

