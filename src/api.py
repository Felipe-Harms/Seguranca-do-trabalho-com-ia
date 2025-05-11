from fastapi import FastAPI, HTTPException
from typing import List
from api_models import Stream, Alert

#para realizar os testes, utilizar cd "C:\Users\focal\OneDrive\Área de Trabalho\VCST\src"
#e depois uvicorn api:app --reload  

app = FastAPI()

streams: List[Stream] = []
alerts: List[Alert] = []
next_stream_id = 0
next_alert_id = 0 

"""
POST /streams - criar uma nova stream
POST /alerts - criar um novo alerta
GET /streams/{id} - obter uma stream específica
GET /alerts/{id} - obter um alerta específico
GET /streams - obter todas as streams
GET /alerts - obter todos os alertas
DELETE /streams/{id} - excluir uma stream
DELETE /alerts/{id} - excluir um alerta
"""

@app.post("/streams", response_model=Stream)
def create_stream(stream: Stream):
    global next_stream_id
    stream.id = next_stream_id
    next_stream_id += 1
    #stream.status = "paused"
    streams.append(stream)
    return  stream

@app.get("/streams", response_model=List[Stream])
def list_streams():
    return streams

@app.get("/streams/{stream_id}", response_model=Stream)
def get_stream(stream_id: int):
    for s in streams:
        if s.id == stream_id:
            return s
    raise HTTPException(status_code=404, detail="Stream not found")


@app.delete("/streams/{stream_id}", status_code=204)
def delete_stream(stream_id:int):
    for s in streams:
        if s.id == stream_id:
            streams.remove(s)
            return 
    raise HTTPException(status_code=404, detail="Stream not found")

@app.post("/alerts", response_model=Alert)
def create_alert(alert:Alert):
    global next_alert_id
    alert.id = next_alert_id
    next_alert_id += 1
    alerts.append(alert)
    return alert 
    
@app.get("/alerts", response_model=List[Alert])
def list_alerts(stream_id: int = None):
    if stream_id is None:
        return alerts
    return [a for a in alerts if a.stream_id == stream_id]

@app.get("/alerts/{alert_id}", response_model=Alert)
def get_alert(alert_id: int):
    for a in alerts:
        if a.id == alert_id:
            return a
    raise HTTPException(status_code=404, detail="Alert not found")


@app.delete("/alerts/{alert_id}", status_code=204)
def delete_alert(alert_id:int):
    for a in alerts:
        if a.id == alert_id:
            alerts.remove(a)
            return 
    raise HTTPException(status_code=404, detail="Stream not found")