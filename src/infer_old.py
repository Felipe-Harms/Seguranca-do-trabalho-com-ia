from ultralytics import YOLO
import cv2
import os

# Configurações
model_path = "models\\best.pt"
video_path = "data\\video3.mp4"
inference_interval = 1  # Inferência em todo frame para melhor detecção
img_size = 2464         # Aumenta o tamanho da imagem para melhor qualidade
conf_threshold = 0.60   # Aceita detecções mais fracas para pegar objetos pequenos

# Definir as classes relevantes (EPIs + Person e Safety Vest)
relevant_classes = ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "Person", "Safety Vest"]

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

    frame_count += 1

    if frame_count % inference_interval == 0:
        # Faz inferência
        results = model(frame, imgsz=img_size, conf=conf_threshold)[0]
        
        # Filtra as detecções para incluir apenas as classes relevantes
        filtered_results = []
        for box in results.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            if class_name in relevant_classes:
                filtered_results.append(box)

        # Cria uma anotação filtrada com as classes relevantes
        results.boxes = filtered_results
        annotated = results.plot()

    if annotated is not None:
        cv2.imshow("PPE Detection", annotated)
    else:
        cv2.imshow("PPE Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
