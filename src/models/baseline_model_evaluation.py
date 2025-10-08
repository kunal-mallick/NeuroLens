import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    jaccard_score,
    f1_score,
    recall_score,
    precision_score,
)

# ---------- same data loader as baseline ----------
DATA_ROOT  = Path("data/processed/Main_Training/train")
IMAGE_ROOT = DATA_ROOT / "image"
MASK_ROOT  = DATA_ROOT / "mark"
MODEL_PATH = Path("models/baseline_3d_unet.h5")
METRICS_DIR = Path("metrics")

INPUT_SHAPE = (128, 128, 128, 4)

METRICS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def patient_dirs(root: Path):
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def load_case(patient: str):
    img_path = IMAGE_ROOT / patient / "image.npy"
    msk_path = MASK_ROOT  / patient / "mask.npy"
    image = np.load(img_path).astype(np.float32)
    mask  = np.load(msk_path).astype(np.float32)
    return image, mask[..., None]  # (H,W,D,1)

# ---------- metrics ----------
def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true, y_pred are flattened binary arrays (0,1).
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = np.logical_and(y_true, y_pred).sum()
    tn = np.logical_and(~y_true, ~y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()

    dice = f1_score(y_true, y_pred, zero_division=0.0)
    iou  = jaccard_score(y_true, y_pred, zero_division=0.0)
    acc  = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0.0)
    spec = precision_score(~y_true, ~y_pred, zero_division=0.0)

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Accuracy": float(acc),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
    }

# ---------- evaluation ----------
def evaluate(model, patients):
    results = []
    for p in patients:
        try:
            image, gt = load_case(p)
        except Exception as e:
            logging.warning("Skip %s: %s", p, e)
            continue

        pred = model.predict(image[None, ...], verbose=0)[0]  # (H,W,D,1)
        pred = (pred > 0.5).astype(np.uint8)

        res = metrics_for(gt.flatten(), pred.flatten())
        res["Patient"] = p
        results.append(res)
        logging.info("%s  Dice=%.3f  IoU=%.3f", p, res["Dice"], res["IoU"])

    df = pd.DataFrame(results).set_index("Patient")
    summary = df.mean().to_dict()
    return df, summary

# ---------- main ----------
def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Loaded model from %s", MODEL_PATH)

    # use same split as training script
    all_patients = patient_dirs(IMAGE_ROOT)
    n_train = int(0.8 * len(all_patients))
    val_patients = all_patients[n_train:]

    if not val_patients:
        logging.warning("No validation patients – nothing to do.")
        return

    df, summary = evaluate(model, val_patients)

    # save
    csv_path = METRICS_DIR / "evaluation.csv"
    json_path = METRICS_DIR / "evaluation.json"
    df.to_csv(csv_path)
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    logging.info("=== Validation summary ===")
    for k, v in summary.items():
        logging.info("%15s: %.4f", k, v)
    logging.info("Detailed table written to %s", csv_path)

if __name__ == "__main__":
    main()