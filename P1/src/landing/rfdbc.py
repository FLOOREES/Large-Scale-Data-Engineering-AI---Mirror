import requests
import json

class RFDBCLandingZone:
    def __init__(self) -> None:
        self.api_url = "https://api.idescat.cat/taules/v2/rfdbc/13301/14148/mun/data"

    def fetch_data(self):
        """Fetches data from the API."""
        try:
            response = requests.get(self.api_url)
            response.raise_for_status() 
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from API: {e}")
            return None
        except requests.exceptions.JSONDecodeError:
            print(f"Error decoding JSON response from API.")
            print(f"Response text: {response.text[:500]}...")
            return None

    def run(self, output_path = "./data/landing/rfdbc.json"):
        """
        Main method to fetch, transform, and save data.
        """
        raw_api_data = self.fetch_data()
        
        # Save data in json format (as it cannot be transformed to parquet because of the complex nested structure)
        with open(output_path, "w") as f:
            json.dump(raw_api_data, f, indent=4)

            