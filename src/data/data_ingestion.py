import shutil
import random
import logging
from pathlib import Path

# ====== CONFIG ======
DATASET_DIR = Path(r"C:\Users\evilk\Downloads\Compressed\Main_Training\train")  # Change this
OUTPUT_DIR = Path("data/raw")
VAL_SPLIT = 0.2
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "ingestion.log"

# ====== LOGGING ======
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

# ====== CONSTANTS ======
REQUIRED_MODALITIES = ["t1", "t1ce", "t2", "flair"]

# ====== UTILS ======
def get_patient_id(name: str) -> str:
    return name.replace("BraTS20_Training_", "").replace("BraTS2021_", "")

def has_all_modalities(patient_path: Path) -> bool:
    """Check if all required MRI modalities are present"""
    present_modalities = [f.stem.lower() for f in patient_path.iterdir() if f.is_file()]
    return all(any(mod in p for p in present_modalities) for mod in REQUIRED_MODALITIES)

def copy_patient(patient_path: Path, dest_root: Path):
    """Copy images and mask to patient folder in train/val"""
    patient_id = get_patient_id(patient_path.name)
    patient_dir = dest_root / f"patient_{patient_id}"
    image_dir = patient_dir / "image"
    mask_dir = patient_dir / "mask"

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for f in patient_path.iterdir():
        if f.is_file():
            name_lower = f.stem.lower()
            if any(mod in name_lower for mod in REQUIRED_MODALITIES):
                shutil.copy2(f, image_dir / f.name)
            elif "seg" in name_lower or "mask" in name_lower:
                shutil.copy2(f, mask_dir / f.name)

    log.info(f"✅ Saved patient {patient_id}")

# ====== MAIN PROCESS ======
def main():
    log.info("📂 Collecting dataset...")
    patients = [p for p in DATASET_DIR.iterdir() if p.is_dir()]
    if not patients:
        log.warning("No patient folders found!")
        return

    # Filter patients with all modalities
    valid_patients = [p for p in patients if has_all_modalities(p)]
    random.shuffle(valid_patients)

    split_idx = int(len(valid_patients) * (1 - VAL_SPLIT))
    train_patients = valid_patients[:split_idx]
    val_patients   = valid_patients[split_idx:]

    log.info(f"Total patients with all modalities: {len(valid_patients)} | Train: {len(train_patients)} | Val: {len(val_patients)}")

    # Copy train
    log.info("📂 Processing train set...")
    for p in train_patients:
        copy_patient(p, OUTPUT_DIR / "train")

    # Copy val
    log.info("📂 Processing val set...")
    for p in val_patients:
        copy_patient(p, OUTPUT_DIR / "val")

    log.info("🎉 Dataset prepared successfully in 'data/raw/'")

if __name__ == "__main__":
    main()