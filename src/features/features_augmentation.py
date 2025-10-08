import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import random
import logging

# ---------- config ----------
RAW_ROOT  = Path("data/processed/Main_Training/train")
IMAGE_DIR = RAW_ROOT / "image"
MASK_DIR  = RAW_ROOT / "mark"
OUT_ROOT  = Path("data/features/Main_Training/train")
OUT_IM    = OUT_ROOT / "image_aug"
OUT_MK    = OUT_ROOT / "mask_aug"

OUT_IM.mkdir(parents=True, exist_ok=True)
OUT_MK.mkdir(parents=True, exist_ok=True)


# ------------------- Logging Configuration -------------------
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
log_filepath = os.path.join(LOG_DIR, "augmentation.log")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

fh = logging.FileHandler(log_filepath, encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# ---------- augmentations ----------
def rand_flip(im, mk):
    if random.random() > .5:
        im = np.flip(im, axis=1); mk = np.flip(mk, axis=1)
    if random.random() > .5:
        im = np.flip(im, axis=0); mk = np.flip(mk, axis=0)
    return im, mk

def rand_rot(im, mk):
    k = random.randint(0,3)
    return np.rot90(im, k=k), np.rot90(mk, k=k)

def rand_zoom(im, mk, z_range=(0.9,1.1)):
    z = random.uniform(*z_range)
    h,w = im.shape[:2]
    nh,nw = int(h*z), int(w*z)
    im_r = tf.image.resize(im, (nh,nw)).numpy()
    mk_r = tf.image.resize(mk.astype(float), (nh,nw), method='nearest').numpy()
    sh,sw = max(0,(nh-h)//2), max(0,(nw-w)//2)
    im_c = im_r[sh:sh+h, sw:sw+w]
    mk_c = mk_r[sh:sh+h, sw:sw+w]
    return im_c, mk_c.astype(mk.dtype)

def rand_bright(im, mk, delta=.1):
    return tf.image.random_brightness(im, delta).numpy(), mk

def augment(im, mk):
    im, mk = rand_flip(im, mk)
    im, mk = rand_rot(im, mk)
    im, mk = rand_zoom(im, mk)
    im, mk = rand_bright(im, mk)
    return im, mk

# ---------- single patient ----------
def process_one(patient: str):
    im_path = IMAGE_DIR / patient / "image.npy"
    mk_path = MASK_DIR  / patient / "mask.npy"
    if not (im_path.exists() and mk_path.exists()):
        logging.warning("Skipping %s – missing image or mask", patient)
        return

    image = np.load(im_path)
    mask  = np.load(mk_path)

    image_aug, mask_aug = augment(image, mask)

    np.save(OUT_IM / f"{patient}_aug.npy", image_aug)
    np.save(OUT_MK / f"{patient}_aug.npy", mask_aug)
    logging.info("Augmented  %s  →  %s_aug.npy", patient, patient)

# ---------- main ----------
def main():
    patients_img = {p.name for p in IMAGE_DIR.iterdir() if p.is_dir()}
    patients_mk  = {p.name for p in MASK_DIR.iterdir()  if p.is_dir()}
    patients = sorted(patients_img & patients_mk)   # only common ones
    logging.info("Found %d patients with both image and mask", len(patients))

    for p in patients:
        process_one(p)

    logging.info("Augmentation finished.")
    print("Augmentation done – logs in log/augmentation.log")

if __name__ == "__main__":
    main()