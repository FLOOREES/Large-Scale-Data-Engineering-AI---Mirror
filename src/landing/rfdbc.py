import requests
import itertools
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

class RFDBC:
    """
    Fetches data from the Idescat RFDBC API and saves it to Parquet.
    API documentation: https://www.idescat.cat/pub/?id=rfdbc
    """
    def __init__(self) -> None:
        # URL for specific table: RFDB i RFDB per habitant. Municipis. 2010-2021.
        self.api_url = "https://api.idescat.cat/taules/v2/rfdbc/13301/14148/mun/data"

    def fetch_data(self):
        """Fetches data from the API."""
        try:
            response = requests.get(self.api_url)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from API: {e}")
            return None
        except requests.exceptions.JSONDecodeError:
            print(f"Error decoding JSON response from API.")
            print(f"Response text: {response.text[:500]}...") # Print first 500 chars for debugging
            return None


    def _transform_data(self, data):
        """Transforms the nested JSON data into a list of records (rows)."""
        if not data or 'dimension' not in data or 'value' not in data or 'id' not in data:
            print("Error: Input data structure is invalid or missing key components.")
            return []

        try:
            dimensions = data['id']
            dim_categories = {
                dim: data['dimension'][dim]['category']['index'] for dim in dimensions
            }
            dim_labels = {
                dim: data['dimension'][dim]['category'].get('label', {}) for dim in dimensions
            }
            values = data['value']

            category_lists = [dim_categories[dim] for dim in dimensions]
            combinations = itertools.product(*category_lists)

            records = []
            value_iter = iter(values)

            for combo in combinations:
                try:
                    raw_value = next(value_iter) # Renamed to raw_value
                except StopIteration:
                    print("Warning: Ran out of values sooner than expected based on dimension sizes.")
                    break

                record = {}
                combo_dict = dict(zip(dimensions, combo))

                for dim in dimensions:
                    code = combo_dict[dim]
                    record[f"{dim}_CODE"] = code
                    label = dim_labels[dim].get(code, code)
                    record[f"{dim}_LABEL"] = label

                # Convert the value to float if it's not None
                if raw_value is not None:
                    try:
                        record['VALUE'] = float(raw_value)
                    except (ValueError, TypeError):
                        # Handle cases where value might unexpectedly not be convertible
                        print(f"Warning: Could not convert value '{raw_value}' to float for combination {combo}. Setting VALUE to None.")
                        record['VALUE'] = None
                else:
                    record['VALUE'] = None # Keep None as None

                records.append(record)

            # Check if the total number of values matches the expected product of dimension sizes
            expected_size = 1
            for dim in dimensions:
                 expected_size *= len(dim_categories[dim])

            if len(values) != expected_size:
                 print(f"Warning: Number of values in API data ({len(values)}) does not match calculated expected size ({expected_size}).")
            elif len(records) != len(values):
                 print(f"Warning: Number of generated records ({len(records)}) does not match number of values ({len(values)}). Some combinations might be missing values or vice-versa.")


            return records

        except KeyError as e:
            print(f"Error: Missing expected key in JSON data during transformation: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during data transformation: {e}")
            return []


    def to_parquet(self, raw_data, path):
        """
        Transforms the raw JSON data and saves it to a Parquet file.
        """
        print("Transforming data...")
        records = self._transform_data(raw_data)

        if not records:
            print("No records generated from data transformation. Cannot save to Parquet.")
            return

        print(f"Generated {len(records)} records. Initializing Spark Session...")
        spark = None
        try:
            spark = SparkSession.builder \
                .appName("RFDBC Data to Parquet") \
                .getOrCreate()

            print("Defining DataFrame schema...")
            # Define schema explicitly for robustness, especially with potential nulls
            schema = StructType([
                StructField("YEAR_CODE", StringType(), True),
                StructField("YEAR_LABEL", StringType(), True),
                StructField("MUN_CODE", StringType(), True),
                StructField("MUN_LABEL", StringType(), True),
                StructField("CONCEPT_CODE", StringType(), True),
                StructField("CONCEPT_LABEL", StringType(), True),
                StructField("INDICATOR_CODE", StringType(), True),
                StructField("INDICATOR_LABEL", StringType(), True),
                StructField("VALUE", DoubleType(), True) # Use DoubleType for potential decimals or large numbers
            ])

            print("Creating Spark DataFrame...")
            # Ensure records match the schema order/names if using createDataFrame with schema
            # Adjusting record creation in _transform_data ensures names match
            df = spark.createDataFrame(records, schema=schema)

            print(f"Writing DataFrame to Parquet at {path}...")
            df.write.mode("overwrite").parquet(path) # Use overwrite for initial load, append for updates
            print("Successfully wrote data to Parquet.")

        except Exception as e:
            print(f"Error during Spark processing or Parquet writing: {e}")
        finally:
            if spark:
                print("Stopping Spark Session.")
                spark.stop()

    def run(self, output_path = "./data/landing/rfdbc.parquet"):
        """
        Main method to fetch, transform, and save data.
        """
        raw_api_data = self.fetch_data()
        if raw_api_data:
            self.to_parquet(raw_api_data, output_path)
        else:
            print("Failed to fetch data from API.")