import cv2
import os
import logging
from time import time
from datetime import datetime
from requests.exceptions import RequestException
from utils.notifier import send_sms  # Importando a função de envio de SMS
from utils.video_tools import preprocess_frame  # Função de pré-processamento
from utils.infer import run_inference, draw_boxes  # Função de inferência
from utils.api_client import register_stream, send_alert
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
cap = cv2.VideoCapture(video_path)

# Verifica se o vídeo abriu corretamente
if not cap.isOpened():
    print("Erro: Não foi possível abrir o vídeo.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou erro ao capturar frame.")
        break

    # Aplica o pré-processamento no frame
    processedframe = preprocess_frame(frame)

    frame_count += 1

    if frame_count % inference_interval == 0:
        # Faz a inferência no frame
        detect_boxes = run_inference(processedframe, model, conf_threshold, relevant_classes)

        # Envia alertas para as detecções relevantes
        for box in detect_boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]

            if class_name in missing_ppe:
                current_time = time()
                if current_time - last_sms_time >= sms_interval:
                    timestamp = datetime.now().isoformat()
                    send_alert(stream_id, class_name, "safety_violation", timestamp)
                    send_sms(f"⚠️ Alerta: {class_name} detectado!")
                    last_sms_time = current_time  # Atualiza o tempo do último SMS
                    logging.info(f"SMS enviado: {class_name} detectado.")
                else:
                    logging.info(f"Alerta {class_name} detectado, mas SMS não enviado (anti-spam).")

        # Gera a anotação (caixas ao redor das detecções)
        annotated = draw_boxes(processedframe.copy(), detect_boxes, model)
        
        cv2.imshow("PPE Detection", annotated)
        
    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
