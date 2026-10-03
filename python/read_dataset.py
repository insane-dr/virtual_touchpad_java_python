from os import makedirs
import cv2
import numpy as np
import kagglehub
import os
import shutil


def false_info(y: np.array):
   # box = y[1: 5].reshape(2, 2)
   x_y_fl = y[5:].reshape(21, 3)
   # x_y = x_y_fl[:,:2]
   fl = x_y_fl[:, 2:]

   if (0 in fl):
      return 1
   return 0


def generate_train_files(folder: str, alph: float):
   m = os.listdir(folder)
   return [m[i][:-4]for i in range(int(len(m) * alph))]


def inicalisation(dataset="riondsilva21/hand-keypoint-dataset-26k"):
   # Ширина терминала
    terminal_size = shutil.get_terminal_size()
    width = terminal_size.columns

    # Load dataset
    path = kagglehub.dataset_download(dataset)
    print(f"Dataset: {path}")
    print("=" * width)
    return path


def y_give(folder: str, train_files:list, standart_size):
  y = None
  none = []


  for i in range(len(train_files)):
    f = open(f"{folder}/{train_files[i]}.txt", "r", encoding="utf-8")
    t = f.read().split().copy()
    y1 = np.array([float(j) for j in t])
    if false_info(y1):
       ind = int(train_files[i][-8:])
       none.append(ind)
       continue
    box = (y1[1:5]).reshape((2, 2))
    x_box = box[0][0] - box[1][0] // 2
    y_box = box[0][1] - box[1][1] // 2
    
    systavi = (y1[5:]).reshape((21, 3))
    systavi[:, 0] -= x_box
    systavi[:, 0] *= standart_size[0] / box[1][0]

    systavi[:, 1] -= y_box
    systavi[:, 1] *= standart_size[1] / box[1][1]

    f.close()

    systavi[:, 2] -= 1
    systavi = np.expand_dims(systavi, axis=0)

    if y is None: y = systavi.copy()
    else: y = np.append(y, systavi, axis=0)
  return y, none


def x_box_give(x_img: np.array, boxes, new_size):
  x = np.zeros((len(x_img), new_size[0], new_size[1]))

  for i in range(len(x_img)):
      img = x_img[i].copy()

      box = boxes[i]

      x_box = int(box[0][0] * new_size[1])
      y_box = int(box[0][1] * new_size[0])
      w = int(box[1][0] * new_size[1]) // 2
      h = int(box[1][1] * new_size[0]) // 2

      x1 = max(0, x_box - w)
      y1 = max(0, y_box - h)
      x2 = min(x_box + w, new_size[0] - 1)
      y2 = min(y_box + h, new_size[0] - 1)

      img = img[y1: y2, x1: x2]

      img = cv2.resize(img, new_size)


      x[i] = img.copy()

  return x

def x_give(folder: str, train_files, none, new_size):
  x = np.zeros((len(train_files) - len(none), new_size[0], new_size[1]))
  true_i = -1

  for i in range(len(train_files)):
      if int(train_files[i][-8:]) in none:
         continue
      else:
         true_i += 1

      img = cv2.imread(f"{folder}/{train_files[i]}.jpg")

      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      img = cv2.resize(img, new_size)

      x[true_i] = img.copy()

  x = x / 255
  return x


def box_give(folder: str, train_files, none, new_size):
  boxes = np.zeros((len(train_files) - len(none), new_size[0], new_size[1]))
  box_coord = np.zeros((len(train_files) - len(none), 2, 2))

  true_i = -1

  for i in range(len(train_files)):
    if int(train_files[i][-8:]) in none:
       continue
    else:
       true_i += 1

    f = open(f"{folder}/{train_files[i]}.txt", "r", encoding="utf-8")
    t = f.read().split().copy()
    box = np.array([float(j) for j in t[1:5]]).reshape((2, 2))
    box_img = np.zeros(new_size)

    x_box = int(box[0][0] * new_size[1])
    y_box = int(box[0][1] * new_size[0])
    w = int(box[1][0] * new_size[1]) // 2
    h = int(box[1][1] * new_size[0]) // 2

    x1 = max(0, x_box - w)
    y1 = max(0, y_box - h)
    x2 = min(x_box + w, new_size[0] - 1)
    y2 = min(y_box + h, new_size[0] - 1)

    for j in range(x1, x2):
       for k in range(y1, y2):
          box_img[k][j] = np.int64(1)
    f.close()

    boxes[true_i] = box_img.copy()

    box_coord[true_i] = box.copy()
  return boxes, box_coord


def save_info(x_train, x_test, x_box_train, x_box_test, y_train, y_test, fl_train, fl_test, boxes_train, boxes_test, none_train, none_test, alph):
    procent = (len(none_train) + len(none_test)) / (x_train[0] + x_test[0] + none_train[0] + none_test[0])

    info = f"Dataset_size: {x_train[0] + x_test[0]}\n"
    info += f"alp: {alph}\n"
    info += f"x_train: {x_train} |  x_test: {x_test}\n"
    info += f"y_train: {y_train} | y_test: {y_test}\n"
    info += f"fl_train: {fl_train} | fl_test: {fl_test}\n"
    info += f"boxes_train: {boxes_train} | boxes_test: {boxes_test}"
    info += f"x_box_train: {x_box_train} | x_box_test: {x_box_test}"
    info += f"none: {round(procent, 2)}"
    info += f"test_size: {round(x_test[0] / (x_train[0] + x_test[0]) * 100, 2)}%"

    with open("dataset/data_info.txt", "w", encoding="utf-8") as file:
        file.write(info)

def save_x(x_train, x_test):
    makedirs('dataset', exist_ok=True)

    np.save("dataset/x_train.npy", x_train)
    np.save("dataset/x_test.npy", x_test)

def save_y(y_train, y_test):
    makedirs('dataset', exist_ok=True)

    np.save("dataset/y_train.npy", y_train)
    np.save("dataset/y_test.npy", y_test)


def save_fl(fl_train, fl_test):
    makedirs('dataset', exist_ok=True)

    np.save("dataset/fl_train.npy", fl_train)
    np.save("dataset/fl_test.npy", fl_test)

def save_boxes(boxes_train, boxes_test):
    makedirs('dataset', exist_ok=True)

    np.save("dataset/boxes_train.npy", boxes_train)
    np.save("dataset/boxes_test.npy", boxes_test)

def save_x_boxes(x_box_train, x_box_test):
    makedirs('dataset', exist_ok=True)

    np.save("dataset/x_boxes_train.npy", x_box_train)
    np.save("dataset/x_boxes_test.npy", x_box_test)
