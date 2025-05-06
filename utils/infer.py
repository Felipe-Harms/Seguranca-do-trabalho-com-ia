#modulo resposável por processar o video.

def run_inference(frame,model,conf_treshold=0.60,relevant_classes=None):
    """
    função que realiza a inferência no frame usando o modelo best do YOLO e filtra 
    as detecções relevantes
    """

    if relevant_classes is None:
        relevant_classes = ['Hardhat','Mask','NO-Hardhat','NO-Mask','NO-Safety Vest','Person','Safety Cone','Safety Vest','machinery','vehicle']

    #faz a inferência no frame
    results = model(frame, imgsz =2464, conf = conf_treshold)[0]

    #lista para armazenar as detecções relevantes
    detect_boxes = []

    for i,box in enumerate(results.boxes):
        class_id = int(box.cls) #id da classe detectada
        class_name = model.names[class_id] #nome da classe detectada
        
        #verifica se a classe detectada é relevante
        if relevant_classes and class_name in relevant_classes:
            detect_boxes.append(box)

    return detect_boxes

import cv2

def draw_boxes(frame, boxes, model):
    # Mapeamento de cores por ID de classe (formato BGR)
    color_map = {
        0: (0, 255, 0),    # Hardhat - Verde
        1: (0, 255, 0),    # Mask - Verde
        2: (0, 0, 255),    # NO-Hardhat - Vermelho
        3: (0, 0, 255),    # NO-Mask - Vermelho
        4: (0, 0, 255),    # NO-Safety Vest - Vermelho
        6: (0, 165, 255),  # Safety Cone - Laranja
        7: (0, 255, 0),    # Safety Vest - Verde
        8: (0, 0, 0),      # Machinery - Preto
        9: (0, 255, 255),  # Vehicle - Amarelo
    }

    for box in boxes:
        # Pega as coordenadas da caixa (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls)
        class_name = model.names[class_id]

        # Obtém a cor a partir do mapeamento, padrão branco
        color = color_map.get(class_id, (255, 255, 255))

        # Desenha o bounding box no frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Escreve o nome da classe acima da caixa
        cv2.putText(
            frame,
            class_name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return frame
