import os
import shutil
import logging

# ------------------- Logging Configuration -------------------
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
log_filepath = os.path.join(LOG_DIR, "data_ingestion.log")

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

# ------------------- Main Script -------------------
def copy_local_folders():
    source_dir = r"C:\Users\NeuroLens\Downloads\Compressed"  # Windows absolute path
    dest_dir = "data/raw"
    os.makedirs(dest_dir, exist_ok=True)

    folders = ["Main_Training", "Generalization_Test", "Federated_Simulation"]

    for folder in folders:
        src_path = os.path.join(source_dir, folder)
        dest_path = os.path.join(dest_dir, folder)

        try:
            if not os.path.exists(src_path):
                logger.warning(f"Folder '{folder}' not found in {source_dir}")
                continue

            # Remove existing folder if exists
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
                logger.info(f"Removed existing folder: {dest_path}")

            # Copy the entire folder
            shutil.copytree(src_path, dest_path)
            logger.info(f"Copied folder '{folder}' to {dest_path}")

        except Exception as e:
            logger.error(f"Failed to copy folder '{folder}': {e}")

def main():
    logger.info("Starting local data ingestion from C:\\Users\\NeuroLens\\Downloads\\Compressed...")
    copy_local_folders()
    logger.info("✅ Data ingestion completed successfully!")

if __name__ == "__main__":
    main()