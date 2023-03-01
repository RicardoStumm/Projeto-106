import cv2

# Carrega o classificador de corpos
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Inicia a captura de vídeo
cap = cv2.VideoCapture('walking.avi')

# Loop principal
while cap.isOpened():
    # Lê um frame do vídeo
    ret, frame = cap.read()

    # Detecta corpos no frame atual
    bodies = body_classifier.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.2, 3)

    # Desenha retângulos em torno dos corpos detectados
    for x, y, w, h in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exibe o frame atual com os retângulos desenhados
    cv2.imshow('Video', frame)

    # Aguarda a tecla 'q' ser pressionada para sair do loop
    if cv2.waitKey(1) == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
