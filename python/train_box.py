import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPooling2D
import keras as kr
import numpy as np

x_train = np.load("dataset/x_train.npy")
box_train = np.load("dataset/boxes_train.npy")

w, h = x_train.shape[2], x_train.shape[1]
w_box, h_box = box_train.shape[2], box_train.shape[1]
print(x_train.shape, box_train.shape)


box_detector = kr.Sequential([
    Reshape((h, w, 1)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(15 * 15, activation="relu"),
    Dense(300, activation="relu"),
    Dense(500, activation="relu"),


    Dense(w_box * h_box, activation="sigmoid"),
    Reshape((h_box, w_box))
])

box_detector.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=["mae"])

history = box_detector.fit(x_train, box_train, batch_size=16, epochs=20)

box_detector.save("models/box_detector.keras")

plt.plot(history.history['loss'], label='Training Loss', color="blue")
# plt.plot(history.history['mse'], label='Validation Loss', color="red")
plt.title('График функции потерь')
plt.xlabel('Epochs')
plt.ylabel('Binary_crossentropy')
plt.legend()

plt.savefig("loss/loss.jpg")


