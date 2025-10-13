import os
import shutil
import random
import logging
from pathlib import Path

# ====== CONFIG ======
DATASET_DIR = Path(r"C:\Users\evilk\Downloads\Compressed\Main_Training\train")  # Change to your dataset path
OUTPUT_DIR = Path("data/raw")
VAL_SPLIT = 0.2  # 20% validation
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "data_ingestion.log"

# ====== SETUP LOGGING ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("ingestion")

# ====== CREATE OUTPUT DIRS ======
train_dir = OUTPUT_DIR / "train"
val_dir = OUTPUT_DIR / "val"
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# ====== UTILS ======
def get_patient_id(name: str) -> str:
    """Extract patient number from folder/file name."""
    return name.replace("BraTS20_Training_", "").replace("BraTS2021_", "")

def rename_and_copy(patient_list, dest_dir):
    """Rename patients and copy to destination directory."""
    for patient in patient_list:
        try:
            new_name = f"patient_{get_patient_id(patient.name)}"
            dest_path = dest_dir / new_name

            if patient.is_dir():
                shutil.copytree(patient, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy2(patient, dest_path)

            log.info(f"{patient.name} → {new_name}")
        except Exception as e:
            log.error(f"Failed to copy {patient.name}: {e}")

# ====== MAIN PROCESS ======
def main():
    log.info("📂 Collecting dataset...")
    patients = [p for p in DATASET_DIR.iterdir() if p.is_dir() or p.suffix in [".nii", ".nii.gz"]]

    if not patients:
        log.warning("No patient files/folders found in the dataset directory!")
        return

    random.shuffle(patients)
    split_idx = int(len(patients) * (1 - VAL_SPLIT))
    train_patients = patients[:split_idx]
    val_patients = patients[split_idx:]

    log.info(f"Total patients: {len(patients)} | Train: {len(train_patients)} | Val: {len(val_patients)}")
    log.info("📂 Processing train set...")
    rename_and_copy(train_patients, train_dir)

    log.info("📂 Processing val set...")
    rename_and_copy(val_patients, val_dir)

    log.info("🎉 Dataset prepared successfully in 'data/raw/'")

if __name__ == "__main__":
    main()