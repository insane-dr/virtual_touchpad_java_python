import numpy as np
import read_dataset as rd

path = rd.inicalisation()

alph = 0.2
new_size = (160, 160)
box_size = (80, 80)
in_box_size = (100, 100)

none_data = []
standart_size = (224, 224)

y_path = f"{path}/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/labels/"

files = rd.generate_train_files(f"{y_path}train", alph)
files2 = rd.generate_train_files(f"{y_path}val", alph)

y_train, none_train = rd.y_give(f"{y_path}train", files, standart_size)
y_test, none_test = rd.y_give(f"{y_path}val", files2, standart_size)
print(f"Dataset_size: {len(y_train) + len(y_test)}")
print(f"alp: {alph} | size_use: {(len(y_train) + len(y_test))}")
print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")


fl_train = y_train[:, :, 2]
y_train = np.delete(y_train, 2, axis=2)
fl_test = y_test[:, :, 2]
y_test = np.delete(y_test, 2, axis=2)
y_train_shape = y_train.shape
y_test_shape = y_test.shape
fl_train_shape = fl_train.shape
fl_test_shape = fl_test.shape
rd.save_y(y_train, y_test)
rd.save_fl(fl_train, fl_test)
del y_train, y_test, fl_train, fl_test


boxes_path = f"{path}/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/labels/"
boxes_train, coord_train = rd.box_give(f"{boxes_path}train", files, none_train, box_size)
boxes_test, coord_test = rd.box_give(f"{boxes_path}val", files2, none_test, box_size)
print(f"boxes_train: {boxes_train.shape} | boxes_test: {boxes_test.shape}")

boxes_train_shape = boxes_train.shape
boxes_test_shape = boxes_test.shape

rd.save_boxes(boxes_train, boxes_test)
del boxes_train, boxes_test


x_path = f"{path}/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/images/"
x_train = rd.x_give(f"{x_path}train", files, none_train, new_size)
x_test = rd.x_give(f"{x_path}val", files2, none_test, new_size)
print(f"x_train: {x_train.shape} | x_test: {x_test.shape}")


x_box_train = rd.x_box_give(x_train, coord_train, in_box_size)
x_box_test = rd.x_box_give(x_test, coord_test, in_box_size)
print(f"x_train: {x_box_train.shape} | x_test: {x_box_test.shape}")

rd.save_x(x_train, x_test)
rd.save_x_boxes(x_box_train, x_box_test)


procent = (len(none_train) + len(none_test)) / ((len(x_train) + len(x_test) + len(none_train) + len(none_test)))
print(f"test_size: {round(len(x_test) / (len(x_train) + len(x_test)) * 100, 2)}%")
print(f"none_data: {round(procent * 100, 2)}%")

rd.save_info(
    x_train.shape, x_test.shape,
    x_box_train.shape, x_box_test.shape,
    y_train_shape, y_test_shape,
    fl_train_shape, fl_test_shape,
    boxes_train_shape, boxes_test_shape,
    none_train, none_test,
    alph
)