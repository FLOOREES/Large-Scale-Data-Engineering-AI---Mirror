import requests

class ICAEN:
	""""
	"A class to interact with the ICAEN (Institut Català d'Energia) API.
	"""
	def __init__(self) -> None:
		self.base_url = 'https://analisi.transparenciacatalunya.cat/resource/8idm-becu.json'

	def request(self, year: int):
		url = f'{self.base_url}?any={year}'
		response = requests.get(url)
		print(response.status_code)
		return response.json()
	

if __name__ == "__main__":
		consumption = ICAEN()
		data = consumption.request(2022)
		print(data)
