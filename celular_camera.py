import cv2

# Carregar o classificador de Haar Cascade para detecção de rosto
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a captura de vídeo com a URL do vídeo do celular

video = cv2.VideoCapture("http://10.46.154.28:8080/video")

while True:
    # Ler um frame do vídeo
    check, img = video.read()
    if not check:
        break
    
    # Redimensionar a imagem para 500x300
    img = cv2.resize(img, (600, 400))
    
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Desenhar retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Exibir a imagem
    cv2.imshow("img", img) 
    
    # Aguardar até que uma tecla seja pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar as janelas do OpenCV
video.release()
cv2.destroyAllWindows()


