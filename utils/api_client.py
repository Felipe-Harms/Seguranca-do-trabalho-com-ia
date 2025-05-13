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

#register_stream("teste1","12.0.12.1")


def send_alert(stream_id: int, class_name: str, alert_type: str, timestamp: str):
    payload = {
        "stream_id" : stream_id,
        "class_name" : class_name,
        "alert_type" : alert_type,
        "timestamp" : timestamp
    }

    response = requests.post(f"{BASE_URL}/alerts", json=payload)
    response.raise_for_status()
    return response.json()

#send_alert(0, "NO-Hardhat", "safety_violation", "2025-05-12T22:55:00")

