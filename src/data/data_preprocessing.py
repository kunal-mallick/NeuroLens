import os
import logging
import torch
import nibabel as nib
import numpy as np
from pathlib import Path


# ------------------------------
# Logging setup
# ------------------------------

LOG_DIR = Path("logs")
log_file = LOG_DIR / "preprocessing.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ------------------------------
# Config
# ------------------------------
INPUT_DIRS = {
    "train": "data/raw/train",
    "val": "data/raw/val"
}

OUTPUT_DIRS = {
    "train": "data/Processed/train",
    "val": "data/Processed/val"
}

TARGET_SHAPE = (160, 192, 155, 4)  # Images: H,W,D,4
MODALITIES = ['flair', 't1', 't1ce', 't2']

# ------------------------------
# Functions
# ------------------------------
def load_mri(file_path):
    """Load NIfTI file as numpy array."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        logging.info(f"Loaded {file_path} with shape {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        return None

def crop_or_pad(data, target_shape):
    """
    Crop or pad 3D/4D data to target shape.
    data: (H,W,D,C) or (H,W,D)
    target_shape: (H,W,D,C_target)
    """
    n_channels = data.shape[3] if data.ndim == 4 else 1
    result = np.zeros(target_shape, dtype=np.float32)

    for i in range(n_channels):
        vol = data[..., i] if data.ndim == 4 else data
        shape = vol.shape

        # Crop
        slices = []
        for dim, t_dim in zip(shape, target_shape[:3]):
            if dim < t_dim:
                slices.append(slice(0, dim))
            else:
                start = (dim - t_dim) // 2
                end = start + t_dim
                slices.append(slice(start, end))
        cropped = vol[slices[0], slices[1], slices[2]]

        # Pad
        pad_width = []
        for dim, t_dim in zip(cropped.shape, target_shape[:3]):
            total_pad = t_dim - dim
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width.append((pad_before, pad_after))
        cropped_padded = np.pad(cropped, pad_width, mode='constant')

        result[..., i] = cropped_padded

    return result

def normalize(data):
    """Normalize data per volume."""
    mean = data.mean()
    std = data.std()
    if std > 0:
        data = (data - mean) / std
    return data

def process_patient(patient_folder, output_dir):
    """Process both images and masks for one patient."""
    image_folder = Path(patient_folder) / "image"
    mask_folder = Path(patient_folder) / "mask"

    # ----------------- Images -----------------
    volumes = []
    for mod in MODALITIES:
        file = list(image_folder.glob(f"*_{mod}.nii*"))
        if not file:
            logging.warning(f"No {mod} file in {image_folder}")
            return
        vol = load_mri(file[0])
        if vol is None:
            return
        volumes.append(vol)
    data = np.stack(volumes, axis=-1)  # (H,W,D,4)
    data = crop_or_pad(data, TARGET_SHAPE)
    data = normalize(data)

    # ----------------- Mask -----------------
    mask_files = list(mask_folder.glob("*.nii*"))
    if not mask_files:
        logging.warning(f"No mask file in {mask_folder}")
        return
    mask = load_mri(mask_files[0])
    mask = crop_or_pad(np.expand_dims(mask, -1), TARGET_SHAPE[:3] + (1,))
    mask = mask.astype(np.uint8)

    # ----------------- Save -----------------
    out_folder = Path(output_dir) / Path(patient_folder).name
    os.makedirs(out_folder, exist_ok=True)
    torch.save(torch.from_numpy(data), out_folder / "image.pt")
    torch.save(torch.from_numpy(mask), out_folder / "mask.pt")
    logging.info(f"Saved preprocessed data for {Path(patient_folder).name}")

def process_split(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    patient_folders = [f for f in Path(input_dir).iterdir() if f.is_dir()]
    logging.info(f"Found {len(patient_folders)} patients in {input_dir}")
    for patient in patient_folders:
        process_patient(patient, output_dir)

# ------------------------------
# Main
# ------------------------------
def main():
    logging.info("Starting preprocessing...")
    for split in ["train", "val"]:
        process_split(INPUT_DIRS[split], OUTPUT_DIRS[split])
    logging.info("Preprocessing finished.")

if __name__ == "__main__":
    main()