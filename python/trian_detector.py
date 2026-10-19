"""
Stage 2: регрессор 21 ключевой точки (x, y) на кропе 80x80.

Данные в dataset/file/:
    x_box_train.npy : (N, 80, 80) или (N, 80, 80, 1)
    y_train.npy     : (N, 42) — 21×(x,y), нормированные в [0,1]

Сохраняет:
    detector_coord/              — папка SavedModel (для инференса)
    detector_coord.weights.h5    — лучшие веса (для дообучения)
"""

import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
from keras import layers, Model, callbacks

# ============================================================
# Конфигурация
# ============================================================
DATA_DIR       = "dataset"
X_FILE         = "x_box_train.npy"
Y_FILE         = "y_train.npy"

EXPORT_DIR     = "detector_coord"                # ← папка SavedModel
CKPT_FILE      = "detector_coord.weights.h5"     # ← чекпоинт весов

CROP_SIZE      = 80
NUM_KP         = 21
BATCH_SIZE     = 32
EPOCHS         = 100
LR             = 1e-3
VAL_FRACTION   = 0.15
SEED           = 42
USE_FLIP       = False


# ============================================================
# Потери и метрики
# ============================================================
def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff   = tf.abs(y_true - y_pred)
    c      = w * (1.0 - tf.math.log1p(w / epsilon))
    loss   = tf.where(diff < w,
                      w * tf.math.log1p(diff / epsilon),
                      diff - c)
    return tf.reduce_mean(loss)


def mean_pixel_error(y_true, y_pred, size=CROP_SIZE):
    diff = (y_true - y_pred) * float(size)
    diff = tf.reshape(diff, (-1, NUM_KP, 2))
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-9))


# ============================================================
# Модель
# ============================================================
def build_kp_model(input_shape=(CROP_SIZE, CROP_SIZE, 1), num_kp=NUM_KP):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_kp * 2, activation='sigmoid', name="keypoints")(x)
    return Model(inputs, outputs, name="kp_regressor")


# ============================================================
# Аугментация
# ============================================================
@tf.function
def augment_pair_kp(img, kp):
    if USE_FLIP:
        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)
            kp_2d = tf.reshape(kp, (NUM_KP, 2))
            x = 1.0 - kp_2d[:, 0]
            kp_2d = tf.stack([x, kp_2d[:, 1]], axis=-1)
            kp = tf.reshape(kp_2d, (NUM_KP * 2,))
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, kp


# ============================================================
# Данные
# ============================================================
def load_data(data_dir, x_file, y_file):
    x_path = os.path.join(data_dir, x_file)
    y_path = os.path.join(data_dir, y_file)
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Не найден файл с кропами: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Не найден файл с ключевыми точками: {y_path}")
    x = np.asarray(np.load(x_path, mmap_mode="r"))
    y = np.asarray(np.load(y_path, mmap_mode="r"))
    print(f"[stage2] x_box: {x.shape} {x.dtype} ← {x_path}")
    print(f"[stage2] y    : {y.shape} {y.dtype} ← {y_path}")
    return x, y


def preprocess(x, y):
    x = x.astype("float32")
    if x.max() > 1.0:
        x = x / 255.0
    if x.ndim == 3:
        x = x[..., None]
    elif x.ndim == 4 and x.shape[1] == 1:
        x = x.transpose(0, 2, 3, 1)

    y = y.astype("float32")
    if y.ndim == 3 and y.shape[-1] == 2:
        y = y.reshape(len(y), -1)
    elif y.ndim == 3 and y.shape[-2] == 2:
        y = y.transpose(0, 2, 1).reshape(len(y), -1)

    if y.max() > 1.5:
        y = y / float(CROP_SIZE)
    y = np.clip(y, 0.0, 1.0)
    return x, y


def make_ds(x, y, batch, shuffle, augment):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 2000), seed=SEED)
    if augment:
        ds = ds.map(augment_pair_kp, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--x-file",   default=X_FILE)
    parser.add_argument("--y-file",   default=Y_FILE)
    parser.add_argument("--epochs",   type=int, default=EPOCHS)
    parser.add_argument("--batch",    type=int, default=BATCH_SIZE)
    parser.add_argument("--export-dir", default=EXPORT_DIR)
    args = parser.parse_args()

    x, y = load_data(args.data_dir, args.x_file, args.y_file)
    x, y = preprocess(x, y)

    print(f"[stage2] crop={x.shape[1]}x{x.shape[2]}, num_kp={y.shape[1] // 2}, samples={len(x)}")

    n = len(x)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    val_n = max(1, int(VAL_FRACTION * n))
    val_idx, tr_idx = idx[:val_n], idx[val_n:]

    train_ds = make_ds(x[tr_idx], y[tr_idx], args.batch, shuffle=True,  augment=True)
    val_ds   = make_ds(x[val_idx], y[val_idx], args.batch, shuffle=False, augment=False)

    model = build_kp_model(input_shape=(x.shape[1], x.shape[2], 1), num_kp=y.shape[1] // 2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=wing_loss,
        metrics=[mean_pixel_error],
    )
    model.summary()

    cbs = [
        callbacks.ModelCheckpoint(
            CKPT_FILE,
            monitor="val_mean_pixel_error", mode="min",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_mean_pixel_error", mode="min",
            factor=0.5, patience=5, min_lr=1e-5, verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_mean_pixel_error", mode="min",
            patience=15, restore_best_weights=True, verbose=1,
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

    if os.path.exists(CKPT_FILE):
        try:
            model.load_weights(CKPT_FILE)
            print(f"[stage2] loaded best weights from {CKPT_FILE}")
        except Exception as e:
            print(f"[stage2] could not load checkpoint ({e}); exporting current weights")

    if os.path.isdir(args.export_dir):
        shutil.rmtree(args.export_dir)
    model.export(args.export_dir)
    print(f"[stage2] exported SavedModel → {args.export_dir}/")
    print(f"[stage2] best weights          → {CKPT_FILE}")


if __name__ == "__main__":
    main()