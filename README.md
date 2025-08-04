# VCST - Visão Computacional para Segurança do Trabalho

Este projeto usa visão computacional e IA para identificar EPIs e detectar riscos.

## Objetivos
- Detectar ausência de EPIs (capacete, luvas, óculos, colete)
- Identificar quedas, zonas de risco e objetos perigosos
- Enviar alertas via FASTAPI a supervisores

## Tecnologias
- Python 3.10
- YOLOv8 (Ultralytics)
- Roboflow
- OpenCV
- FastAPI

## 🚀 Como usar (Backend)

Siga os passos abaixo para executar o sistema de backend do VCST localmente:

### 1. Instale as dependências
Dentro da pasta `Backend_VCST`, crie e ative um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv .venv
# Ativação no Windows:
.venv\Scripts\activate
# Ou no Linux/macOS:
source .venv/bin/activate
```

Instale as dependências:
```bash
pip install -r requirements.txt
```

### 2. Baixe o modelo treinado (.pt)
Faça o download do modelo YOLOv8 segmentado (.pt) treinado via Roboflow e salve na pasta adequada (por exemplo: `models/`).

### 3. Inicie o servidor da API
Execute o Uvicorn dentro da pasta `Backend_VCST` para iniciar a FastAPI:
```bash
uvicorn src.api:app --reload
```

A API estará disponível em:
- http://127.0.0.1:8000
- Documentação interativa: http://127.0.0.1:8000/docs

### 4. Execute o sistema principal
Em outro terminal (com o ambiente virtual ativado):
```bash
python main.py
```

## Licença

Este projeto está licenciado sob uma Licença de Uso Não Comercial.

Você pode usar, modificar e distribuir este software **somente para fins não comerciais**.  
O uso comercial é **estritamente proibido** sem permissão expressa do autor.

Consulte o arquivo [LICENSE](./LICENSE) para mais detalhes.
