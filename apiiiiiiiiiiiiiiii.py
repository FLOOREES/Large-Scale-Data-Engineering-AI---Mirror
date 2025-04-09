import requests
import time

# Endpoint de la API
BASE_URL = "https://analisi.transparenciacatalunya.cat/resource/qww9-bvhh.json"

# Parámetros de paginación
LIMIT = 1000
OFFSET = 0

# Lista donde almacenaremos todos los registros
all_records = []

while True:
    # Realizamos la petición
    params = {
        "$limit": LIMIT,
        "$offset": OFFSET
    }
    response = requests.get(BASE_URL, params=params)
    
    # Si hay error en la petición, salimos del bucle
    if response.status_code != 200:
        print(f"Error en la petición: {response.status_code}")
        break
    
    # Parseamos la respuesta
    data = response.json()
    
    # Si no hay más datos, salimos del bucle
    if not data:
        break

    # Añadimos los datos a nuestra lista general
    all_records.extend(data)

    print(f"Descargados {len(data)} registros (offset={OFFSET})")

    # Incrementamos el offset para la siguiente página
    OFFSET += LIMIT

    # Añadimos un pequeño retraso para evitar saturar la API
    time.sleep(0.5)

# Resultado final
print(f"\nTotal de registros descargados: {len(all_records)}")

# Finalemente guaradmos el JSON
import json

with open('data_preu_mitja.json', 'w') as f:
    json.dump(all_records, f, indent=4, ensure_ascii=True)

# Ahora, para guardar el JSON en formato Parquet, podemos usar pandas y pyarrow
# Lo pasamos a Parquet

