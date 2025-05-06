import cv2 

def resize_frame(frame,scale=1/3):
   width = int(frame.shape[1] * scale)
   height = int(frame.shape[0] * scale)
   return cv2.resize(frame,(width,height), interpolation=cv2.INTER_AREA)

def preprocess_frame(frame):
    """
    Pré-processamento que mantém a cor original e melhora o contraste:
    1) Redimensiona
    2) Converte para YCrCb
    3) Equaliza o canal Y (luminância)
    4) Reconstrói YCrCb e converte de volta para BGR
    """
    # 1) Redimensiona
    resized = resize_frame(frame)

    # 2) Converte para YCrCb
    ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)

    # 3) Separa canais e equaliza Y
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)

    # 4) Reúne e converte de volta para BGR
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    processed = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    return processed

def test_import():
   print("Deu bom")
