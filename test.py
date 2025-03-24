import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

 
key = os.getenv('API_KEY')
url = 'https://api.meteo.cat/referencia/v1/municipis'
 
response = requests.get(url, headers={"Content-Type": "application/json", "X-Api-Key": key})
 
print(response.status_code)  #statusCode
print(response.text) #valors de la resposta