import cv2
from time import time
from datetime import datetime
import logging
from requests.exceptions import RequestException
from utils.notifier import send_sms  # Importando a função de envio de SMS
from utils.video_tools import preprocess_frame  # Função de pré-processamento
from utils.infer import run_inference, draw_boxes  # Função de inferência
from src.api_client import register_stream, send_alert,list_streams
from ultralytics import YOLO

#configurações(ainda não descobri onde esses precisam estar)
inference_interval = 1  # Inferência a cada frame (ou de outro modo)
img_size = 1280         # Aumenta o tamanho da imagem para melhor qualidade
conf_threshold = 0.60   # Confiança mínima para a detecção
sms_interval = 30       # Intervalo entre o envio de SMS (em segundos)  
model_path = "models\\best.pt"
model = YOLO(model_path, task="segment")
relevant_classes = ['Hardhat','Mask','NO-Hardhat','NO-Mask','NO-Safety Vest','Person','Safety Cone','Safety Vest','machinery','vehicle']
missing_ppe = ['NO-Hardhat','NO-Mask','NO-Safety Vest']



def process_stream(source: str, stream_id: int, stream_name: str, stop_event):

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Erro: não foi possível abrir a câmera '{stream_name}' ({source})")
        return

    frame_count = 0
    last_sms_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Stream '{stream_name}' finalizada ou erro ao capturar frame.")
            break

        # Pré-processamento
        processed = preprocess_frame(frame)

        frame_count += 1
        if frame_count % inference_interval == 0:
            # Inferência
            boxes = run_inference(processed, model, conf_threshold, relevant_classes)

            # Alertas para infrações
            current_time = time()
            for box in boxes:
                class_name = model.names[int(box.cls)]
                if class_name in missing_ppe and (current_time - last_sms_time) >= sms_interval:
                    timestamp = datetime.now().isoformat()
                    send_alert(stream_id, class_name, "safety_violation", timestamp)
                    send_sms(f"⚠️ Alerta [{stream_name}]: {class_name} detectado!")
                    last_sms_time = current_time
                    logging.info(f"SMS enviado para {stream_name}: {class_name}")

            # Exibição
            annotated = draw_boxes(processed.copy(), boxes, model)
            cv2.imshow(f"PPE - {stream_name}", annotated)

        # Tecla 'q' encerra o vídeo

        if stop_event.is_set():
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyWindow(f"PPE - {stream_name}")
