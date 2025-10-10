import os
import random
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np # Import numpy

# ---------------- CONFIG ----------------
modalities = ["flair", "t1", "t1ce", "t2"]
cmap = 'gray'

case_names = ["BraTS2021_01557", "BraTS2021_01558", "BraTS2021_01559"]

raw_base_dir = r"data\raw\Main_Training\Train"
processed_base_dir = r"data\processed\Main_Training\train\image"
save_dir = r"reports\figures\processed"

# Create output directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

def load_nifti(filepath):
    nii = nib.load(filepath)
    return nii.get_fdata()

# ---------------- VISUALIZATION ----------------
for case_name in case_names:
    print(f"\n==============================")
    print(f"Comparing RAW vs PROCESSED for case: {case_name}")
    print(f"==============================")

    raw_images = {}
    proc_image_data = None # Initialize processed image data to None

    # ---- Load raw and processed ----
    for modality in modalities:
        raw_path = os.path.join(raw_base_dir, case_name, f"{case_name}_{modality}.nii.gz")
        if os.path.exists(raw_path):
            raw_images[modality] = load_nifti(raw_path)
        else:
            print(f"[WARN] Missing raw: {raw_path}")

    # Load the processed data once per case
    proc_path = os.path.join(processed_base_dir, case_name, "image.npy")
    if os.path.exists(proc_path):
        try:
            proc_image_data = np.load(proc_path)
            # Assuming the shape is [depth, height, width, modalities]
            # Check if the number of modalities in the loaded data matches the expected number
            if proc_image_data.shape[-1] != len(modalities):
                print(f"[WARN] Number of modalities in {proc_path} ({proc_image_data.shape[-1]}) does not match expected ({len(modalities)}).")
                proc_image_data = None # Set to None if mismatch
        except Exception as e:
            print(f"[ERROR] Could not load processed data from {proc_path}: {e}")
            proc_image_data = None
    else:
        print(f"[WARN] Missing processed data file: {proc_path}")


    # ---- Skip if no data ----
    if not raw_images and proc_image_data is None:
        print(f"[INFO] No data found for {case_name}")
        continue

    # ---- Determine slice index for RAW data ----
    raw_slice_idx = None
    if raw_images:
        # Use the first available raw modality to determine slice index
        raw_ref_image = next(iter(raw_images.values()))
        if len(raw_ref_image.shape) == 3:
             raw_max_slice_idx = raw_ref_image.shape[2] - 1
             raw_slice_idx = random.randint(0, raw_max_slice_idx)
        else:
            print(f"[WARN] Unexpected raw image shape for slice determination: {raw_ref_image.shape}")


    # ---- Determine slice index for PROCESSED data ----
    proc_slice_idx = None
    if proc_image_data is not None:
        # Assuming processed data shape is [depth, height, width, modalities]
        if len(proc_image_data.shape) == 4:
            proc_max_slice_idx = proc_image_data.shape[0] - 1
            proc_slice_idx = random.randint(0, proc_max_slice_idx)
        else:
            print(f"[WARN] Unexpected processed image shape for slice determination: {proc_image_data.shape}")


    # ---- Determine available modalities for plotting ----
    available_modalities_for_plotting = [m for m in modalities if m in raw_images or proc_image_data is not None]
    num_modalities_to_plot = len(available_modalities_for_plotting)

    if num_modalities_to_plot == 0:
        print(f"[INFO] No modalities available to plot for {case_name}")
        continue


    # ---- Create figure (2 rows: raw on top, processed below) ----
    plt.figure(figsize=(num_modalities_to_plot * 4, 8))
    # Use raw slice index in the title if available, otherwise use processed slice index
    title_slice_idx = raw_slice_idx if raw_slice_idx is not None else proc_slice_idx
    plt.suptitle(f"{case_name} — Slice {title_slice_idx}", fontsize=14, y=0.95)

    for i, modality in enumerate(available_modalities_for_plotting):
        # ----- RAW -----
        plt.subplot(2, num_modalities_to_plot, i + 1)
        if modality in raw_images and raw_slice_idx is not None:
            # Assuming raw data shape is [height, width, depth]
            plt.imshow(raw_images[modality][:, :, raw_slice_idx], cmap=cmap)
            plt.title(f"RAW - {modality.upper()}")
        else:
            plt.text(0.5, 0.5, "Missing Raw", ha="center", va="center")
        plt.axis("off")

        # ----- PROCESSED -----
        plt.subplot(2, num_modalities_to_plot, num_modalities_to_plot + i + 1)
        if proc_image_data is not None and proc_slice_idx is not None and modality in modalities:
             # Find the index of the current modality in the expected modalities list
             try:
                 modality_idx = modalities.index(modality)
                 # Assuming processed data shape is [depth, height, width, modalities]
                 plt.imshow(proc_image_data[proc_slice_idx, :, :, modality_idx], cmap=cmap)
                 plt.title(f"PROCESSED - {modality.upper()}")
             except ValueError:
                  plt.text(0.5, 0.5, f"Modality {modality.upper()} not in processed data", ha="center", va="center")

        else:
            plt.text(0.5, 0.5, "Missing Processed", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # ---- Save the figure ----
    # Ensure the save_dir exists before saving
    os.makedirs(save_dir, exist_ok=True)
    # Include both raw and processed slice indices in the filename if available, or just one if only one is available
    if raw_slice_idx is not None and proc_slice_idx is not None:
        save_path = os.path.join(save_dir, f"{case_name}_comparison_raw_slice_{raw_slice_idx}_proc_slice_{proc_slice_idx}.png")
    elif raw_slice_idx is not None:
         save_path = os.path.join(save_dir, f"{case_name}_comparison_raw_slice_{raw_slice_idx}.png")
    elif proc_slice_idx is not None:
         save_path = os.path.join(save_dir, f"{case_name}_comparison_proc_slice_{proc_slice_idx}.png")
    else:
        save_path = os.path.join(save_dir, f"{case_name}_comparison.png") # Fallback filename

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"[SAVED] Comparison figure saved to: {save_path}")

    # Optionally display
    plt.show()