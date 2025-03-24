import requests
import json

class MeteoCat:
	"""
	A class to interact with the Meteocat API.
	"""
	
	def __init__(self, api_key: str):
		self.api_key = api_key
		
		self.base_url = 'https://api.meteo.cat/xema/v1'
		self.headers = {
			"Content-Type": "application/json",
			"X-Api-Key": self.api_key
		}

		self.stations_metadata = json.load(open('data/meteocat_stations_metadata.json', 'r', encoding='utf-8'))
		self.variables_metadata = json.load(open('data/meteocat_variables_metadata.json', 'r', encoding='utf-8'))

		self.sufix_stations_measurements = f'/estacions/mesurades/{id_station}/{year}/{month}/{day}'
		self.sufix_variable_measurements = f'/variables/mesurades/{id_variable}/{year}/{month}/{day}'

	def get_metadata(self):
		pass

	def get_request(self, url: str):
		return requests.get(url, headers=self.headers)
		
