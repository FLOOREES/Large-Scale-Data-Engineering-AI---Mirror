import requests



url = f'https://api.idescat.cat/emex/v1/dades.json?id={node_id}'
url = f'https://api.idescat.cat/emex/v1/nodes.json'



class IdesCat:
	"""
	A class to interact with the IdesCat (Institut d'Estadística de Catalunya) API.
	"""

	def __init__(self):
		self.base_url = 'https://api.idescat.cat/emex/v1'

	def get_nodes(self):
		"""
		Fetch the nodes from the IdesCat API.
		"""
		url = f'{self.base_url}/nodes.json'
		response = requests.get(url)
		print(response.status_code)
		return response.json()

	def get_info(self, node_id: str):
		url = f'{self.base_url}/dades.json?id={node_id}'
		response = requests.get(url)
		print(response.status_code)
		return response.json()
	
	def get_comarques(self):
		"""
		Get the list of comarques (counties) from the IdesCat API.
		"""
		comarques = idescat.get_nodes()['fitxes']['v']['v']

	