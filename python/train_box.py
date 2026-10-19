"""
Stage 1: U-Net для сегментации руки.

Данные в dataset/file/:
    x_train.npy      : (N, 160, 160) или (N, 160, 160, 1)
    boxes_train.npy  : (N, M, M)     или (N, M, M, 1), маска {0,1}, M = 160 или 80

Сохраняет:
    detector/                    — папка SavedModel (для инференса)
    detector.weights.h5          — лучшие веса (для дообучения)
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
X_FILE         = "x_train.npy"
Y_FILE         = "boxes_train.npy"

EXPORT_DIR     = "detector"                   # ← папка SavedModel
CKPT_FILE      = "detector.weights.h5"        # ← чекпоинт весов

IMG_SIZE       = 160
MASK_SIZE      = None
BATCH_SIZE     = 16
EPOCHS         = 40
LR             = 1e-3
VAL_FRACTION   = 0.15
SEED           = 42


# ============================================================
# Потери и метрики
# ============================================================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return (2.0 * inter + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(dice_coef(y_true, y_pred))


def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred, thresh=0.5):
    y_pred = tf.cast(y_pred > thresh, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - inter
    return tf.reduce_mean((inter + 1e-6) / (union + 1e-6))


# ============================================================
# Модель
# ============================================================
def build_unet(input_shape=(160, 160, 1), target_size=None):
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)

    u5 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(c5)

    u6 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(16, 3, activation='relu', padding='same')(c7)

    out = layers.Conv2D(1, 1, activation='sigmoid', name="mask")(c7)

    if target_size is not None and target_size != input_shape[0]:
        out = layers.Resizing(target_size, target_size, interpolation='bilinear',
                              name="mask_resize")(out)

    return Model(inputs, out, name="unet_seg")


# ============================================================
# Аугментация (исправленная)
# ============================================================
@tf.function
def augment_pair(img, mask):
    # Одно решение о флипе на пару — применяем независимо к обоим тензорам
    do_flip = tf.random.uniform([]) > 0.5
    img  = tf.cond(do_flip, lambda: tf.image.flip_left_right(img),  lambda: img)
    mask = tf.cond(do_flip, lambda: tf.image.flip_left_right(mask), lambda: mask)

    # Фотометрия — только к изображению
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, mask


# ============================================================
# Данные
# ============================================================
def load_data(data_dir, x_file, y_file):
    x_path = os.path.join(data_dir, x_file)
    y_path = os.path.join(data_dir, y_file)
    print(f"[debug] ls {data_dir}:",
          os.listdir(data_dir) if os.path.isdir(data_dir) else "NO DIR")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Не найден файл с изображениями: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Не найден файл с маской: {y_path}")
    x = np.asarray(np.load(x_path, mmap_mode="r"))
    y = np.asarray(np.load(y_path, mmap_mode="r"))
    print(f"[stage1] x: {x.shape} {x.dtype} ← {x_path}")
    print(f"[stage1] y: {y.shape} {y.dtype} ← {y_path}")
    return x, y


def preprocess(x, y):
    x = x.astype("float32")
    if x.max() > 1.0:
        x = x / 255.0
    if x.ndim == 3:
        x = x[..., None]
    elif x.ndim == 4 and x.shape[1] == 1:
        x = x.transpose(0, 2, 3, 1)

    y = (y > 0.5).astype("float32")
    if y.ndim == 3:
        y = y[..., None]
    elif y.ndim == 4 and y.shape[1] == 1:
        y = y.transpose(0, 2, 3, 1)
    return x, y


def make_ds(x, y, batch, shuffle, augment):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 2000), seed=SEED)
    if augment:
        ds = ds.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default=DATA_DIR)
    parser.add_argument("--x-file",     default=X_FILE)
    parser.add_argument("--y-file",     default=Y_FILE)
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch",      type=int, default=BATCH_SIZE)
    parser.add_argument("--mask-size",  type=int, default=None)
    parser.add_argument("--export-dir", default=EXPORT_DIR)
    args = parser.parse_args()

    x, y = load_data(args.data_dir, args.x_file, args.y_file)
    x, y = preprocess(x, y)

    mask_size = args.mask_size if args.mask_size else y.shape[1]
    img_size  = x.shape[1]
    print(f"[stage1] img_size={img_size}, mask_size={mask_size}, samples={len(x)}")

    n = len(x)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    val_n = max(1, int(VAL_FRACTION * n))
    val_idx, tr_idx = idx[:val_n], idx[val_n:]

    train_ds = make_ds(x[tr_idx], y[tr_idx], args.batch, shuffle=True,  augment=True)
    val_ds   = make_ds(x[val_idx], y[val_idx], args.batch, shuffle=False, augment=False)

    model = build_unet((img_size, img_size, 1), target_size=mask_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=bce_dice_loss,
        metrics=[iou_metric, dice_coef],
    )
    model.summary()

    cbs = [
        callbacks.ModelCheckpoint(
            CKPT_FILE,
            monitor="val_iou_metric", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_iou_metric", mode="max",
            factor=0.5, patience=5, min_lr=1e-5, verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_iou_metric", mode="max",
            patience=15, restore_best_weights=True, verbose=1,
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

    if os.path.exists(CKPT_FILE):
        try:
            model.load_weights(CKPT_FILE)
            print(f"[stage1] loaded best weights from {CKPT_FILE}")
        except Exception as e:
            print(f"[stage1] could not load checkpoint ({e}); exporting current weights")

    if os.path.isdir(args.export_dir):
        shutil.rmtree(args.export_dir)
    model.export(args.export_dir)
    print(f"[stage1] exported SavedModel → {args.export_dir}/")
    print(f"[stage1] best weights          → {CKPT_FILE}")


if __name__ == "__main__":
    main()