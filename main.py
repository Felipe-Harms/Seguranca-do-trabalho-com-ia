import os
import logging
import threading
from requests.exceptions import RequestException
from utils.process_stream import process_stream
from src.api_client import register_stream, send_alert,list_streams
from ultralytics import YOLO

# Configurações
stop_event = threading.Event()
model_path = "models\\best.pt"
video_path = "data\\video3.mp4"
inference_interval = 1  # Inferência a cada frame (ou de outro modo)
img_size = 1280         # Aumenta o tamanho da imagem para melhor qualidade
conf_threshold = 0.60   # Confiança mínima para a detecção
sms_interval = 30       # Intervalo entre o envio de SMS (em segundos)  

"""
#adcionar stream(origem do video)
try:
    stream_info = register_stream(name="Câmera Principal", source=video_path)
    stream_id = stream_info["id"]
    print(f"Stream registrada com ID {stream_id}")
except RequestException as e:
    print("Erro ao registrar a stream na API:", e)
    exit(1)
"""

#lista de streams
registered_streams = []

#loop para cadastro.

while True:
    answer = input("Deseja adicionar uma nova câmera ? (s/n): ").strip().lower()
    if answer != "s":
        break
    
    #pergunta nome e caminho da steam

    stream_name = input("  • Digite o nome da câmera: ").strip()
    stream_source = input("  • Digite a URL ou caminho da stream: ").strip()

    #chamar register_stream e guarda
    info = register_stream(name=stream_name, source=stream_source)
    registered_streams.append(info)
    print(f" -> '{info['name']}' cadastrada com ID {info['id']}")

#lista stream ja existentes

existing = list_streams()

registered_streams.extend(existing)

print("Streams disponíveis:")
for idx,s in enumerate(registered_streams):
    print(f"  {idx}: {s['name']} ({s['source']})")

choice = input("Escolha os números das câmeras para processar(ex:1,2,4,6,7): ")
index = [int(x.strip()) for x in choice.split(",") if x.strip().isdigit()]
selected_streams = [registered_streams[i] for i in index]

# Definir as classes relevantes (EPIs + Person e Safety Vest)

relevant_classes = ['Hardhat','Mask','NO-Hardhat','NO-Mask','NO-Safety Vest','Person','Safety Cone','Safety Vest','machinery','vehicle']
missing_ppe = ['NO-Hardhat','NO-Mask','NO-Safety Vest']

# Configuração de logs
logging.basicConfig(filename="detections.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Verifica se o arquivo do modelo existe
if not os.path.exists(model_path):
    print(f"Erro: Modelo não encontrado em {model_path}")
    exit()

# Carrega o modelo YOLO
model = YOLO(model_path, task="segment")

# Carrega o vídeo seperando-o em threads.

threads = []

for s in selected_streams:
    t = threading.Thread(
        target=process_stream,
        args=(s["source"], s["id"], s["name"], stop_event),
        daemon=True
    )
    t.start()
    threads.append(t)

# Espera todas as threads finalizarem
for t in threads:
    t.join()
