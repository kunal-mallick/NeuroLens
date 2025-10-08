# src/models/baseline_model.py
import os
import time
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------------
DATA_ROOT = Path("data/processed/Main_Training/train")
IMAGE_ROOT = DATA_ROOT / "image"
MASK_ROOT  = DATA_ROOT / "mark"
MODEL_DIR  = Path("models")
LOG_DIR    = Path("log")

BATCH_SIZE  = 2
EPOCHS      = 5
INPUT_SHAPE = (128, 128, 128, 4)

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "baseline_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
tf.get_logger().setLevel("ERROR")
# --------------------------------------------------

def patient_dirs(root: Path):
    """Return sorted list of patient-folder names (BraTS20_Training_XXX)."""
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def load_case(patient: str):
    """Return (image, mask) np arrays given patient id."""
    img_path = IMAGE_ROOT / patient / "image.npy"
    msk_path = MASK_ROOT  / patient / "mask.npy"
    if not img_path.exists() or not msk_path.exists():
        raise FileNotFoundError(f"Missing {img_path} or {msk_path}")
    image = np.load(img_path).astype(np.float32)
    mask  = np.load(msk_path).astype(np.float32)
    return image, mask[..., None]  # add channel dim

def build_tf_dataset(batch_size: int = 2, split: float = 0.8):
    patients = patient_dirs(IMAGE_ROOT)
    if not patients:
        raise RuntimeError(f"No patient folders found in {IMAGE_ROOT}")

    n_train = int(len(patients) * split)
    train_pat, val_pat = patients[:n_train], patients[n_train:]

    def generator(pat_list):
        for p in pat_list:
            try:
                yield load_case(p)
            except Exception as e:
                logging.warning("Skip %s: %s", p, e)
                continue

    def ds_from_list(pat_list):
        ds = tf.data.Dataset.from_generator(
            lambda: generator(pat_list),
            output_signature=(
                tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
                tf.TensorSpec(shape=INPUT_SHAPE[:-1] + (1,), dtype=tf.float32),
            ),
        )
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_from_list(train_pat), ds_from_list(val_pat), len(train_pat), len(val_pat)

# ---------------- U-Net ----------------
def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    c1 = layers.Conv3D(16, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPool3D()(c1)
    c2 = layers.Conv3D(32, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPool3D()(c2)
    c3 = layers.Conv3D(64, 3, activation='relu', padding='same')(p2)

    u1 = layers.UpSampling3D()(c3)
    u1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv3D(32, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling3D()(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv3D(16, 3, activation='relu', padding='same')(u2)

    outputs = layers.Conv3D(1, 1, activation='sigmoid')(c5)
    return models.Model(inputs, outputs)

# ---------------- train ----------------
def main():
    t0 = time.time()
    logging.info("=== start baseline training ===")
    MODEL_DIR.mkdir(exist_ok=True)

    train_ds, val_ds, n_train, n_val = build_tf_dataset(BATCH_SIZE)
    logging.info("Dataset ready: %d train / %d val", n_train, n_val)

    model = build_unet(INPUT_SHAPE)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model_path = MODEL_DIR / "baseline_3d_unet.h5"
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(model_path), save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
        ]
    )
    model.save(model_path)
    logging.info("Saved model to %s", model_path)
    logging.info("=== done in %.1f min ===", (time.time() - t0)/60)

if __name__ == "__main__":
    main()