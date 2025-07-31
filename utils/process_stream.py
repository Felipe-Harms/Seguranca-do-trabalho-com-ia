import cv2
from time import time
from datetime import datetime
import logging
import os
from requests.exceptions import RequestException
from src.api_client import send_alert
from utils.video_tools import preprocess_frame  # Função de pré-processamento
from utils.infer import run_inference, draw_boxes  # Função de inferência
from src.api_client import register_stream, send_alert, list_streams
from ultralytics import YOLO

# Configurações
inference_interval = 1      # Inferência a cada frame
img_size = 1280             # Aumenta o tamanho da imagem para melhor qualidade
conf_threshold = 0.60       # Confiança mínima para a detecção
save_interval = 15          # Intervalo periódico de salvamento de frames
save_frames = False         # True para salvar frames / False para nao salvar
alert_interval = 30
model_path = "models\\\\best.pt"

# Carrega o modelo YOLOv8 em modo segmentação
model = YOLO(model_path, task="segment")

# Classes relevantes e classes de violação
relevant_classes = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
    'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
]
missing_ppe = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']


def process_stream(source: str, stream_id: int, stream_name: str, stop_event):
    # Abre o stream de vídeo
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Erro: não foi possível abrir a câmera '{stream_name}' ({source})")
        return

    frame_count = 0
    last_alert_time = 0
    violation_active = False  # Flag para primeiro frame de cada evento de violação

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Stream '{stream_name}' finalizada ou erro ao capturar frame.")
            break

        # Pré-processamento do frame
        processed = preprocess_frame(frame)
        frame_count += 1

        # Realiza inferência em intervalos definidos
        if frame_count % inference_interval == 0:
            boxes = run_inference(processed, model, conf_threshold, relevant_classes)

            # Verifica infrações e envia alertas
            alert_sent = False
            current_time = time()
            current_violation = False
            for box in boxes:
                class_name = model.names[int(box.cls)]
                # Se violação e passou o intervalo de SMS
                if (class_name in missing_ppe and not alert_sent and (last_alert_time == 0 or (current_time - last_alert_time) >= alert_interval)):
                    ts = datetime.now().isoformat()
                    send_alert(stream_id,class_name , "safety_violation", ts)
                    last_alert_time = current_time
                    logging.info(f"[API] Alerta enviado para stream {stream_id} às {ts}")
                # Marca que há violação em algum box
                if class_name in missing_ppe:
                    current_violation = True

            # Desenha bounding boxes e monta nome do arquivo
            annotated = draw_boxes(processed.copy(), boxes, model)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{stream_id}_{timestamp}.jpg"

            # Decide se deve salvar o frame
            save_event = False
            # Salva o primeiro frame de cada evento de violação
            if current_violation and not violation_active:
                save_event = True
            # Salva periodicamente mesmo sem violação
            elif frame_count % save_interval == 0:
                save_event = True

            if save_frames and save_event :
                # Cria pasta para o stream, se não existir
                output_dir = os.path.join("frames_anotados", str(stream_id))
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename)

                # Salva frame anotado e registra log
                cv2.imwrite(output_path, annotated)
                logging.info(f"Frame salvo em {output_path}")

                # Atualiza flag de violação
                violation_active = current_violation

        # Permite parada externa do loop
        if stop_event.is_set():
            break

    cap.release()
   #cv2.destroyAllWindows() -não está sendo usado pois não é uma vesão com suporte para GUI
