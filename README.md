# VCST - Visão Computacional para Segurança do Trabalho

Este projeto usa visão computacional e IA para identificar EPIs e detectar riscos em canteiros de obras.

## Objetivos
- Detectar ausência de EPIs (capacete, luvas, óculos, colete)
- Identificar quedas, zonas de risco e objetos perigosos
- Enviar alertas por SMS a supervisores

## Tecnologias
- Python 3.10
- YOLOv8 (Ultralytics)
- Roboflow
- OpenCV
- Twilio
- FastAPI

## Como usar
1. Instale dependências
2. Baixe o modelo treinado (.pt)
3. rode o uvicorn api:app --reload  na pasta onde está o api.py 
4. Rode `main.py`

## Licença

Este projeto está licenciado sob uma Licença de Uso Não Comercial.

Você pode usar, modificar e distribuir este software **somente para fins não comerciais**.  
O uso comercial é **estritamente proibido** sem permissão expressa do autor.

Consulte o arquivo [LICENSE](./LICENSE) para mais detalhes.
