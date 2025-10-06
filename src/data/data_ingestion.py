import os
import logging
import gdown

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

fh = logging.FileHandler(log_filepath)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# ------------------- Main Download Script -------------------
def download_gdrive_folders():
    save_dir = "data/raw"
    os.makedirs(save_dir, exist_ok=True)

    folders = {
        "Main_Training": "13_T_sEudFGzkiJlIogeqzle2AKeIOKXA",
        "Generalization_Test": "1FWvFkIGkRwq51dpr5HahUO59kuBx4Ta-",
        "Federated_Simulation": "1WuX_qrM0hmoWWNLtjFQSn9KKEXMvx8DS"
    }

    for name, folder_id in folders.items():
        output_path = os.path.join(save_dir, name)
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Downloading {name}...")
        try:
            gdown.download_folder(
                f"https://drive.google.com/drive/folders/{folder_id}",
                output=output_path,
                quiet=False,
                use_cookies=False
            )
            logger.info(f"{name} downloaded successfully to {output_path}")
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")

def main():
    logger.info("Starting data ingestion...")
    download_gdrive_folders()
    logger.info("Data ingestion completed successfully!")

if __name__ == "__main__":
    main()