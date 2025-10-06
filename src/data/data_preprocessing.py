import os
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import zipfile
import logging

# ---------------- Logging ----------------
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
log_filepath = os.path.join(LOG_DIR, "preprocessing.log")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(log_filepath)
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# ---------------- Parameters ----------------
TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_SHAPE = (128, 128, 128)
RAW_DIR = "data/raw/Main_Training"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------- Helper Functions ----------------
def unzip_dataset(zip_path, extract_to):
    logger.info(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info(f"Extraction completed: {extract_to}")


def resample_image(img, target_spacing=TARGET_SPACING):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    return resampler.Execute(img)


def resample_mask(mask, target_spacing=TARGET_SPACING):
    original_spacing = mask.GetSpacing()
    original_size = mask.GetSize()
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputDirection(mask.GetDirection())
    resampler.SetOutputOrigin(mask.GetOrigin())
    return resampler.Execute(mask)


def zscore_normalize(volume):
    volume = sitk.GetArrayFromImage(volume).astype(np.float32)
    mean = np.mean(volume)
    std = np.std(volume)
    volume = (volume - mean) / (std + 1e-8)
    return volume


def crop_or_pad(volume, target_shape=TARGET_SHAPE):
    shape = volume.shape
    pad_width = [(0, max(0, target_shape[i] - shape[i])) for i in range(3)]
    volume = np.pad(volume, pad_width, mode='constant', constant_values=0)
    crop_slices = tuple(slice(0, target_shape[i]) for i in range(3))
    volume = volume[crop_slices]
    return volume


def process_case(image_path, mask_path, save_dir):
    img = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    img = resample_image(img)
    mask = resample_mask(mask)

    img = zscore_normalize(img)
    mask = sitk.GetArrayFromImage(mask)

    img = crop_or_pad(img)
    mask = crop_or_pad(mask)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    np.save(os.path.join(save_dir, f"{base_name}_image.npy"), img)
    np.save(os.path.join(save_dir, f"{base_name}_mask.npy"), mask)
    logger.info(f"Processed {base_name}")


def process_dataset(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
    os.makedirs(processed_dir, exist_ok=True)
    # Extract all zip files
    for file in os.listdir(raw_dir):
        if file.endswith(".zip"):
            unzip_dataset(os.path.join(raw_dir, file), raw_dir)

    # Process all image-mask pairs
    for root, _, files in os.walk(raw_dir):
        images = sorted([f for f in files if "t1" in f.lower() or "flair" in f.lower()])
        masks = sorted([f for f in files if "seg" in f.lower()])

        for img_file, mask_file in zip(images, masks):
            image_path = os.path.join(root, img_file)
            mask_path = os.path.join(root, mask_file)
            process_case(image_path, mask_path, processed_dir)


# ---------------- TensorFlow Dataset ----------------
def build_tf_dataset(processed_dir=PROCESSED_DIR, batch_size=2, shuffle=True):
    image_files = sorted([os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if "image" in f])
    mask_files = sorted([os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if "mask" in f])

    def load_npy(image_path, mask_path):
        image = np.load(image_path.decode())
        mask = np.load(mask_path.decode())
        return image[..., np.newaxis], mask[..., np.newaxis]

    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    dataset = dataset.map(lambda x, y: tf.py_function(load_npy, [x, y], [tf.float32, tf.float32]),
                            num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_files))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------- Main ----------------
def main():
    logger.info("Starting preprocessing...")
    process_dataset()
    logger.info("Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
