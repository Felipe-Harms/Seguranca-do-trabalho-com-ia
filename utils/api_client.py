import requests

BASE_URL = "http://127.0.0.1:8000"

def register_stream (name:str , source:str) -> dict :
    payload = {
        "name" : name,
        "source" : source
    }
    
    response = requests.post(f"{BASE_URL}/streams", json=payload)

    response.raise_for_status()

    return response.json()

register_stream("teste1","12.0.12.1")
