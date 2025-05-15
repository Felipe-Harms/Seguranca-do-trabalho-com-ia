import cv2
import os
import logging
import json
from time import time
from datetime import datetime
from requests.exceptions import RequestException
from utils.notifier import send_sms  # Importando a função de envio de SMS
from utils.video_tools import preprocess_frame  # Função de pré-processamento
from utils.infer import run_inference, draw_boxes  # Função de inferência
from src.api_client import register_stream, send_alert,list_streams
from ultralytics import YOLO

# Configurações
model_path = "models\\best.pt"
video_path = "data\\video3.mp4"
inference_interval = 1  # Inferência a cada frame (ou de outro modo)
img_size = 2464         # Aumenta o tamanho da imagem para melhor qualidade
conf_threshold = 0.60   # Confiança mínima para a detecção
sms_interval = 30       # Intervalo entre o envio de SMS (em segundos)  

#adcionar stream(origem do video)
try:
    stream_info = register_stream(name="Câmera Principal", source=video_path)
    stream_id = stream_info["id"]
    print(f"Stream registrada com ID {stream_id}")
except RequestException as e:
    print("Erro ao registrar a stream na API:", e)
    exit(1)

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

# Variáveis de controle
last_sms_time = 0  # Armazenar o tempo do último SMS enviado

# Verifica se o arquivo do modelo existe
if not os.path.exists(model_path):
    print(f"Erro: Modelo não encontrado em {model_path}")
    exit()

# Carrega o modelo YOLO
model = YOLO(model_path, task="segment")

# Verifica se o arquivo de vídeo existe
if not os.path.exists(video_path):
    print(f"Erro: Vídeo não encontrado em {video_path}")
    exit()

# Carrega o vídeo
caps = []
stream_ids = []
stream_names = []

for s in selected_streams:
    caps.append (cv2.VideoCapture(s["source"]))
    stream_ids.append(s["id"])
    stream_names.append(s["name"])

# Verifica se os vídeos abriram corretamente
for idx, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Erro: não foi possível abrir a câmera {stream_names[idx]}")
        exit(1)

frame_count = 0

while True:
    for idx,cap in enumerate(caps):
        ret,frame = cap.read()
        if not ret:
            continue

        # Aplica o pré-processamento no frame
        processed_frame = preprocess_frame(frame)

        frame_count += 1
        
        if frame_count % inference_interval == 0:
            # Faz a inferência no frame
            detect_boxes = run_inference(processed_frame, model, conf_threshold, relevant_classes)

            # Envia alertas para as detecções relevantes
            for box in detect_boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]

                if class_name in missing_ppe:
                    current_time = time()
                    if current_time - last_sms_time >= sms_interval:
                        timestamp = datetime.now().isoformat()
                        send_alert(stream_ids[idx], class_name, "safety_violation", timestamp)
                        send_sms(f"⚠️ Alerta: {class_name} detectado!")
                        last_sms_time = current_time  # Atualiza o tempo do último SMS
                        logging.info(f"SMS enviado: {class_name} detectado.")
                    else:
                        logging.info(f"Alerta {class_name} detectado, mas SMS não enviado (anti-spam).")

            # Gera a anotação (caixas ao redor das detecções)
            annotated = draw_boxes(processed_frame.copy(), detect_boxes, model)
        
            cv2.imshow(f"PPE - {stream_names[idx]}", annotated)
        
    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()

cv2.destroyAllWindows()
