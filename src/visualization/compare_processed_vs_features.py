import os
import random
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Paths
PROCESSED_PATH = "data/processed/train"
FEATURES_PATH = "data/features/train"
SAVE_PATH = "reports/figures/processed_vs_features"

# Create folder if it doesn't exist
os.makedirs(SAVE_PATH, exist_ok=True)

# Logging
logging.basicConfig(
    filename="log/visualize_features.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_pt_files(folder):
    data_dict = {}
    for f in Path(folder).glob("*.pt"):
        try:
            tensor = torch.load(f, weights_only=True)  # safe loading
            data_dict[f.stem] = tensor
        except Exception as e:
            logging.warning(f"Failed to load {f}: {e}")
    return data_dict

def get_random_slice(tensor):
    if tensor.ndim == 3:
        C, H, W = tensor.shape
        slice_idx = random.randint(0, C-1)
        return tensor[slice_idx]
    elif tensor.ndim == 2:
        return tensor
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

def visualize_comparison(processed_dict, features_dict):
    for key in processed_dict:
        if key not in features_dict:
            logging.warning(f"{key} not found in features folder")
            continue

        processed_slice = get_random_slice(processed_dict[key])
        features_slice = get_random_slice(features_dict[key])

        processed_np = processed_slice.detach().cpu().numpy()
        features_np = features_slice.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(processed_np, cmap='gray')
        axes[0].set_title("Processed")
        axes[0].axis('off')

        axes[1].imshow(features_np, cmap='gray')
        axes[1].set_title("Features")
        axes[1].axis('off')

        plt.tight_layout()
        save_file = os.path.join(SAVE_PATH, f"{key}.png")
        fig.savefig(save_file, bbox_inches='tight')
        plt.close(fig)

        logging.info(f"Saved comparison for {key} -> {save_file}")

def main():
    processed_dict = load_pt_files(PROCESSED_PATH)
    features_dict = load_pt_files(FEATURES_PATH)
    visualize_comparison(processed_dict, features_dict)

if __name__ == "__main__":
    main()
