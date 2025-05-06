import cv2
import os
import logging
from time import time
from notifier import send_sms  # Importando a função de envio de SMS
from ultralytics import YOLO

# Configurações
model_path = "models\\best.pt"
video_path = "data\\video3.mp4"
inference_interval = 1  # Inferência em todo frame para melhor detecção
img_size = 2464         # Aumenta o tamanho da imagem para melhor qualidade
conf_threshold = 0.60   # Aceita detecções mais fracas para pegar objetos pequenos
sms_interval = 30       # Intervalo entre o envio de SMS (em segundos)

# Definir as classes relevantes (EPIs + Person e Safety Vest)
relevant_classes = ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "Person", "Safety Vest"]

# Configurar o log
logging.basicConfig(filename="detections.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Variáveis de controle
last_sms_time = 0  # Armazenar o tempo do último SMS enviado

# Verifica se o arquivo do modelo existe
if not os.path.exists(model_path):
    print(f"Erro: Modelo não encontrado em {model_path}")
    exit()

# Carrega o modelo
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
annotated = None  # Inicializa a variável

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou erro ao capturar frame.")
        break

    # Redimensiona o frame para 1/3 do tamanho original
    frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))


    frame_count += 1

    if frame_count % inference_interval == 0:
        # Faz inferência
        results = model(frame, imgsz=img_size, conf=conf_threshold)[0]

        # Filtra as detecções para incluir apenas as classes relevantes
        keep = []
        for i, box in enumerate(results.boxes):
            class_id = int(box.cls)
            class_name = model.names[class_id]
            if class_name in relevant_classes:
                keep.append(i)

        # Filtra e desenha as caixas
        if len(keep) > 0:
            for i in keep:
                box = results.boxes[i]
                # Enviar SMS em caso de detecção de EPI faltando (negativa)
                if model.names[int(box.cls)] in ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]:
                    # Verifica se o tempo desde o último SMS é maior que o intervalo
                    current_time = time()
                    if current_time - last_sms_time >= sms_interval:
                        send_sms(f"⚠️ Alerta: {model.names[int(box.cls)]} detectado!")
                        last_sms_time = current_time  # Atualiza o tempo do último SMS
                        logging.info(f"SMS enviado: {model.names[int(box.cls)]} detectado.")  # Log do evento
                    else:
                        logging.info(f"Alerta {model.names[int(box.cls)]} detectado, mas SMS não enviado (anti-spam).")

        # Gera a anotação
        annotated = results.plot()

    if annotated is not None:
        cv2.imshow("PPE Detection", annotated)
    else:
        cv2.imshow("PPE Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
