import os
import random
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import numpy as np
from pathlib import Path

# ------------------------------
# Config
# ------------------------------
RAW_DIR = "data/raw/train"
PROCESSED_DIR = "data/Processed/train"
FIG_DIR = "reports/figures/processed_vs_raw"
MODALITIES = ['flair', 't1', 't1ce', 't2']

os.makedirs(FIG_DIR, exist_ok=True)

# ------------------------------
# Functions
# ------------------------------
def load_raw_patient(patient_folder):
    volumes = []
    for mod in MODALITIES:
        file = list(Path(patient_folder).glob(f"*_{mod}.nii*"))
        if not file:
            raise FileNotFoundError(f"No {mod} file in {patient_folder}")
        vol = nib.load(file[0]).get_fdata()
        volumes.append(vol)
    data = np.stack(volumes, axis=-1)
    return data

def load_processed_patient(pt_file):
    return torch.load(pt_file).numpy()

def visualize_comparison(raw_data, processed_data, patient_name):
    # Randomize slice along axial axis (Z)
    z_slice = random.randint(0, min(raw_data.shape[2], processed_data.shape[2]) - 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Patient: {patient_name} | Slice: {z_slice}", fontsize=16)

    for i, mod in enumerate(MODALITIES):
        # Raw
        axes[0, i].imshow(raw_data[:, :, z_slice, i], cmap='gray')
        axes[0, i].set_title(f"Raw {mod}")
        axes[0, i].axis('off')
        # Processed
        axes[1, i].imshow(processed_data[:, :, z_slice, i], cmap='gray')
        axes[1, i].set_title(f"Processed {mod}")
        axes[1, i].axis('off')

    plt.tight_layout()
    save_path = Path(FIG_DIR) / f"{patient_name}_slice{z_slice}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {save_path}")

# ------------------------------
# Main
# ------------------------------
def main():
    # Random patient
    patient_folders = [f for f in Path(RAW_DIR).iterdir() if f.is_dir()]
    patient_folder = random.choice(patient_folders)
    patient_name = patient_folder.name
    raw_data = load_raw_patient(patient_folder)

    # Corresponding processed file
    processed_file = Path(PROCESSED_DIR) / f"{patient_name}.pt"
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_file}")
    processed_data = load_processed_patient(processed_file)

    visualize_comparison(raw_data, processed_data, patient_name)

if __name__ == "__main__":
    main()