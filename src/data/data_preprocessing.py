import os
import time
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool, cpu_count, Manager, get_context
from pathlib import Path
import logging

# ---------------- LOGGING ----------------
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "preprocessing.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- CONFIG ----------------
RAW_BASE = Path("data/raw/Main_Training")
PROC_BASE = Path("data/processed/Main_Training")
TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_SHAPE = (128, 128, 128)
MODALITIES = ["t1", "t1ce", "t2", "flair"]

# Use fewer cores than max to prevent memory pressure
N_PROCESSES = max(1, min(6, cpu_count() - 2))

# ---------------- UTILITY FUNCTIONS ----------------
def load_nifti(path):
    return sitk.ReadImage(str(path))

def resample_image(image, target_spacing=TARGET_SPACING, is_mask=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)

def zscore_normalize(volume):
    arr = sitk.GetArrayFromImage(volume).astype(np.float32)
    mean, std = arr.mean(), arr.std()
    if std > 0:
        arr = (arr - mean) / std
    return arr

def crop_or_pad(volume, target_shape=TARGET_SHAPE):
    current_shape = np.array(volume.shape)
    pad = np.maximum((np.array(target_shape) - current_shape) // 2, 0)
    crop = np.maximum((current_shape - np.array(target_shape)) // 2, 0)
    volume = np.pad(volume, [(p, p) for p in pad], mode="constant")
    slices = tuple(slice(c, c + ts) for c, ts in zip(crop, target_shape))
    return volume[slices]

# ---------------- PROCESS SINGLE PATIENT ----------------
def preprocess_patient(patient_dir, out_image_dir, out_mask_dir=None, require_seg=True, stats_dict=None):
    start_time = time.time()
    patient_name = os.path.basename(patient_dir)
    status = "skipped"
    has_seg = False

    try:
        img_out = out_image_dir / patient_name
        if img_out.exists():
            status = "already_processed"
            return status

        # Load and process modalities
        images = []
        for m in MODALITIES:
            file = list(Path(patient_dir).glob(f"*{m}*.nii*"))
            if not file:
                logging.warning(f"{patient_name}: Missing modality {m}")
                return "missing_modality"
            img = load_nifti(file[0])
            img = resample_image(img)
            arr = zscore_normalize(img)
            arr = crop_or_pad(arr)
            images.append(arr)

        image_stack = np.stack(images, axis=-1)
        os.makedirs(img_out, exist_ok=True)
        np.save(img_out / "image.npy", image_stack)

        # Save mask only if available and required
        if out_mask_dir:
            seg_file = list(Path(patient_dir).glob("*seg*.nii*"))
            if seg_file and os.path.exists(seg_file[0]):
                mask = load_nifti(seg_file[0])
                mask = resample_image(mask, is_mask=True)
                mask_arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
                mask_arr = crop_or_pad(mask_arr)
                os.makedirs(out_mask_dir / patient_name, exist_ok=True)
                np.save(out_mask_dir / patient_name / "mask.npy", mask_arr)
                has_seg = True
            elif require_seg:
                logging.warning(f"{patient_name}: Missing segmentation.")
                return "missing_seg"

        duration = time.time() - start_time
        status = "processed"
        logging.info(f"{patient_name}: {status}, seg={has_seg}, time={duration:.2f}s")

        if stats_dict is not None:
            stats_dict[patient_name] = {"status": status, "has_seg": has_seg, "time_s": round(duration, 2)}

        return status

    except Exception as e:
        logging.error(f"{patient_name}: Error {e}")
        if stats_dict is not None:
            stats_dict[patient_name] = {"status": "error", "has_seg": False, "time_s": 0}
        return "error"

# ---------------- PROCESS SPLIT ----------------
def process_split(split, require_seg=True):
    input_dir = RAW_BASE / split
    output_image_dir = PROC_BASE / split / "image"
    os.makedirs(output_image_dir, exist_ok=True)

    output_mask_dir = None
    if require_seg:
        output_mask_dir = PROC_BASE / split / "mark"
        os.makedirs(output_mask_dir, exist_ok=True)

    patient_dirs = [str(p) for p in input_dir.iterdir() if p.is_dir()]
    logging.info(f"{split}: Found {len(patient_dirs)} patients")

    manager = Manager()
    stats_dict = manager.dict()

    start_total = time.time()
    ctx = get_context("spawn")  # Safer multiprocessing on Windows Server
    with ctx.Pool(processes=N_PROCESSES) as pool:
        for p in patient_dirs:
            pool.apply_async(preprocess_patient, args=(p, output_image_dir, output_mask_dir, require_seg, stats_dict))
        pool.close()
        pool.join()
    end_total = time.time()

    processed = sum(1 for v in stats_dict.values() if v["status"]=="processed")
    skipped = sum(1 for v in stats_dict.values() if v["status"]=="already_processed")
    errors = sum(1 for v in stats_dict.values() if v["status"]=="error")
    no_seg = sum(1 for v in stats_dict.values() if v["has_seg"]==False and v["status"]=="processed")

    logging.info(f"===== {split.upper()} Summary =====")
    logging.info(f"Processed: {processed}, Skipped: {skipped}, No seg: {no_seg}, Errors: {errors}")
    logging.info(f"{split} total time: {(end_total - start_total)/60:.2f} min")
    logging.info("Per-patient runtimes:")
    for name, s in stats_dict.items():
        logging.info(f"{name}: {s}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    total_start = time.time()
    logging.info("===== Preprocessing Started (Azure VM Optimized) =====")
    process_split("train", require_seg=True)
    process_split("val", require_seg=False)  # Skip masks for val
    total_end = time.time()
    logging.info(f"===== All Done in {(total_end - total_start)/60:.2f} min =====")