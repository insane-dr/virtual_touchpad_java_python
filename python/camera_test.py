import cv2
import time
import keras as kr
import numpy as np

detector = kr.models.load_model("models/box.keras")

# Открываем камеру (0 – первая доступная)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ValueError

w = 120
h = 120
w_box = 60
h_box = 60


target_fps = 30
frame_interval = 1.0 / target_fps
tm = time.time()

while True:
    start_time = time.perf_counter()  # засекаем время начала обработки кадра

        # Читаем кадр
    ret, img = cap.read()
    if not ret:
        break
    
    #x = img1[int((time.time() - tm) / 2)].copy()
    
    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (w, h))
    x = x / 255
    x = np.reshape(x, (1, w, h))
    maska = detector.predict(x, verbose=False)[0].reshape(w_box, h_box)
    '''
    for i in range(h_box):
        for j in range(w_box):
            maska[i][j] = round(maska[i][j])
    '''
    maska = cv2.resize(maska, (w, h))

    img_x = x[0] * maska
    img_x = cv2.resize(img_x, (640, 480))
    cv2.imshow('Webcam', img_x)

    # Вычисляем затраченное время на захват и отображение
    elapsed = time.perf_counter() - start_time

    # Если обработали быстрее, чем интервал, ждём оставшееся время
    if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)

    # Проверяем нажатие клавиши 'q' для выхода (задержка 1 мс для обработки событий окна)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
