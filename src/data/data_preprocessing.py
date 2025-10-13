import os
import logging
import torch
import nibabel as nib
import numpy as np
from pathlib import Path

# ------------------------------
# Logging setup
# ------------------------------
log_file = "log/data_preprocessing.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ------------------------------
# Config
# ------------------------------
INPUT_DIRS = {
    "train": "data/raw/train",
    "val": "data/raw/val"
}

OUTPUT_DIRS = {
    "train": "data/processed/train",
    "val": "data/processed/val"
}

TARGET_SHAPE = (160, 192, 128, 4)  # (H, W, D, modalities)
MODALITIES = ['flair', 't1', 't1ce', 't2']

# ------------------------------
# Functions
# ------------------------------
def load_mri(file_path):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        logging.info(f"Loaded {file_path} with shape {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        return None

def crop_or_pad(data, target_shape):
    result = np.zeros(target_shape, dtype=np.float32)
    for i in range(4):
        vol = data[..., i] if data.ndim == 4 else data
        shape = vol.shape
        slices = []
        for dim, t_dim in zip(shape, target_shape[:3]):
            if dim < t_dim:
                slices.append(slice(0, dim))
            else:
                start = (dim - t_dim) // 2
                end = start + t_dim
                slices.append(slice(start, end))
        cropped = vol[slices[0], slices[1], slices[2]]
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
    mean = data.mean()
    std = data.std()
    if std > 0:
        data = (data - mean) / std
    return data

def process_patient(patient_folder, output_dir):
    # Collect modality files
    files = []
    for mod in MODALITIES:
        file = list(Path(patient_folder).glob(f"*_{mod}.nii*"))
        if not file:
            logging.warning(f"No file found for modality {mod} in {patient_folder}")
            return
        files.append(file[0])
    # Load and stack modalities
    volumes = []
    for f in files:
        vol = load_mri(f)
        if vol is None:
            return
        volumes.append(vol)
    data = np.stack(volumes, axis=-1)  # shape: (H,W,D,4)
    data = crop_or_pad(data, TARGET_SHAPE)
    data = normalize(data)
    out_file = Path(output_dir) / (Path(patient_folder).name + ".pt")
    torch.save(torch.from_numpy(data), out_file)
    logging.info(f"Saved preprocessed data to {out_file}")

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
