import os
import torch
import random
import logging
import warnings
from pathlib import Path

# ---------------------------
# Settings
# ---------------------------
# Ignore warnings
warnings.filterwarnings("ignore")

# Input and output folders
PROCESSED_TRAIN = "data/processed/train"
PROCESSED_VAL = "data/processed/val"
FEATURES_TRAIN = "data/features/train"
FEATURES_VAL = "data/features/val"

# Logging setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/augmentation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------
# Augmentation function
# ---------------------------
def augment_tensor(tensor):
    """
    Applies augmentation to a 4D tensor [H, W, D, C].
    Only flips and brightness to preserve original shape.
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor [H,W,D,C], got {tensor.shape}")

    H, W, D, C = tensor.shape
    augmented = torch.empty_like(tensor)
    
    for c in range(C):
        channel_tensor = tensor[:, :, :, c]

        # Random flips along axes
        if random.random() > 0.5:
            channel_tensor = torch.flip(channel_tensor, dims=[0])  # H-axis
        if random.random() > 0.5:
            channel_tensor = torch.flip(channel_tensor, dims=[1])  # W-axis
        if random.random() > 0.5:
            channel_tensor = torch.flip(channel_tensor, dims=[2])  # D-axis

        # Random brightness scaling
        factor = 0.8 + 0.4 * random.random()  # [0.8,1.2]
        channel_tensor = channel_tensor * factor

        augmented[:, :, :, c] = channel_tensor

    return augmented

# ---------------------------
# Process patient folders
# ---------------------------
def process_folder(input_folder, output_folder):
    """
    Process all patient folders in input_folder:
    - Augment image.pt
    - Copy mask.pt
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for patient_dir in input_folder.iterdir():
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        out_patient_dir = output_folder / patient_id
        out_patient_dir.mkdir(parents=True, exist_ok=True)

        # --- IMAGE AUGMENTATION ---
        image_file = patient_dir / "image.pt"
        if image_file.exists():
            tensor = torch.load(image_file, weights_only=True)
            try:
                augmented_tensor = augment_tensor(tensor)
                torch.save(augmented_tensor, out_patient_dir / "image.pt")
                logging.info(f"[{patient_id}] Saved augmented image.pt")
            except Exception as e:
                logging.error(f"[{patient_id}] Failed to augment image: {e}")
        else:
            logging.warning(f"[{patient_id}] image.pt not found, skipping")

        # --- COPY MASK ---
        mask_file = patient_dir / "mask.pt"
        if mask_file.exists():
            mask_tensor = torch.load(mask_file, weights_only=True)
            torch.save(mask_tensor, out_patient_dir / "mask.pt")
            logging.info(f"[{patient_id}] Copied mask.pt")
        else:
            logging.warning(f"[{patient_id}] mask.pt not found, skipping")

# ---------------------------
# Main function
# ---------------------------
def main():
    logging.info("Starting augmentation process...")
    process_folder(PROCESSED_TRAIN, FEATURES_TRAIN)
    process_folder(PROCESSED_VAL, FEATURES_VAL)
    logging.info("Augmentation process completed.")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    main()