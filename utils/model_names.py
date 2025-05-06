from ultralytics import YOLO

# Carregar o modelo (assumindo que o modelo já foi carregado)
model = YOLO("models\\best.pt", task="segment")

# Pegar todas as classes que o modelo pode detectar
all_classes = model.names

# Imprimir todas as classes
print(all_classes)
