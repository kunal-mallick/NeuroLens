import os
import json
import logging
import shutil
import subprocess


def setup_logging():
    """Configures logging to save to a file and print to the console."""
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, "data_ingestion.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def setup_kaggle_credentials():
    """Loads Kaggle API credentials from kaggle.json and sets environment variables."""
    try:
        with open('kaggle.json') as f:
            kaggle_creds = json.load(f)
        os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
        os.environ['KAGGLE_KEY'] = kaggle_creds['key']
        logging.info("Kaggle API credentials set successfully.")
        return True
    except FileNotFoundError:
        logging.error("kaggle.json not found. Please place it in the same directory as the script.")
        return False


def download_and_extract(name, kaggle_id, base_dir):
    """Downloads a dataset from Kaggle and extracts it."""
    logging.info(f"Starting download for {name} dataset...")

    download_command = [
        "kaggle", "datasets", "download", "-d", kaggle_id,
        "-p", base_dir, "--unzip"
    ]

    logging.info(f"Executing command: {' '.join(download_command)}")
    try:
        subprocess.run(download_command, check=True)
        logging.info(f"{name} dataset downloaded and extracted successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to download {name}: {e}")


def copy_dataset_to_mixed(source_dir, mixed_dir, dataset_name):
    """
    Copies patient folders from source_dir into mixed_dir.
    Renames folders if duplicates exist.
    """
    copied_count = 0

    if not os.path.exists(source_dir):
        logging.warning(f"Source directory {source_dir} not found. Skipping.")
        return 0

    for patient_folder in os.listdir(source_dir):
        src_patient_path = os.path.join(source_dir, patient_folder)
        if not os.path.isdir(src_patient_path):
            continue  # skip non-folder files

        dst_patient_path = os.path.join(mixed_dir, patient_folder)

        # Handle duplicates: rename with dataset prefix
        if os.path.exists(dst_patient_path):
            new_name = f"{dataset_name}_{patient_folder}"
            dst_patient_path = os.path.join(mixed_dir, new_name)
            logging.warning(f"Duplicate folder {patient_folder} found. Renamed to {new_name}.")

        shutil.copytree(src_patient_path, dst_patient_path)
        copied_count += 1

    logging.info(f"Copied {copied_count} patient folders from {dataset_name}.")
    return copied_count


def main():
    """Main function to orchestrate the download, extraction, and merging of BraTS datasets."""
    setup_logging()

    if not setup_kaggle_credentials():
        return

    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)

    datasets = {
        "BraTS2020": "awsaf49/brats20-dataset-training-validation",
        "BraTS2021": "dschettler8845/brats-2021-task1"
    }

    # Step 1: Download datasets
    for name, kaggle_id in datasets.items():
        download_and_extract(name, kaggle_id, data_dir)

    # Step 2: Merge into BraTS_Mixed
    mixed_dir = os.path.join(data_dir, "BraTS_Mixed")
    os.makedirs(mixed_dir, exist_ok=True)
    logging.info(f"Created merged directory at: {mixed_dir}")

    total_copied = 0

    for name in datasets.keys():
        source_dir = os.path.join(data_dir, name)

        # Look for common nested folders
        nested_training = os.path.join(source_dir, f"MICCAI_BraTS_{name[-4:]}_TrainingData")
        nested_validation = os.path.join(source_dir, f"MICCAI_BraTS_{name[-4:]}_ValidationData")

        folders_to_copy = []
        if os.path.exists(nested_training):
            folders_to_copy.append(nested_training)
        if os.path.exists(nested_validation):
            folders_to_copy.append(nested_validation)

        if not folders_to_copy and os.path.exists(source_dir):
            folders_to_copy = [os.path.join(source_dir, item) for item in os.listdir(source_dir)]

        for folder in folders_to_copy:
            total_copied += copy_dataset_to_mixed(folder, mixed_dir, name)

    logging.info(f"✅ Finished merging datasets. Total patient folders copied: {total_copied}")


if __name__ == "__main__":
    main()