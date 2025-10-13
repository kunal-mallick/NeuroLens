import os
import torch
import torchvision.transforms as transforms
import logging
import random
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Setup logging
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "augmentation.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_data(path):
    """Load .pt files from a directory."""
    files = [f for f in os.listdir(path) if f.endswith(".pt")]
    data_list = []
    for f in files:
        file_path = os.path.join(path, f)
        tensor = torch.load(file_path, weights_only=True)  # avoid FutureWarning
        data_list.append((tensor, f))
    logging.info(f"Loaded {len(data_list)} files from {path}")
    return data_list

def augment_tensor(tensor):
    """
    Apply random flip, 0 or 180-degree rotation, and brightness.
    Works for 2D (H, W), 3D (C, H, W), or 4D (N, C, H, W) tensors.
    Does NOT stack 4D tensors; returns list for batch if needed.
    """
    tensor = tensor.float()

    # Handle 2D -> 3D
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)  # C, H, W

    # Handle 4D batch tensors
    if tensor.ndim == 4:
        N, C, H, W = tensor.shape
        # Return list of augmented 3D tensors
        return [augment_tensor(tensor[i]) for i in range(N)]

    # 3D tensor: C, H, W
    C, H, W = tensor.shape

    # Random horizontal flip
    if random.random() > 0.5:
        tensor = torch.flip(tensor, dims=[2])
    # Random vertical flip
    if random.random() > 0.5:
        tensor = torch.flip(tensor, dims=[1])
    # Random rotation (0 or 180 degrees)
    if random.random() > 0.5:
        tensor = torch.rot90(tensor, k=2, dims=[1, 2])
    # Random brightness
    brightness_factor = random.uniform(0.7, 1.3)
    tensor = tensor * brightness_factor
    tensor = torch.clamp(tensor, 0, 1)

    return tensor



def save_data(data_list, save_path):
    """Save list of augmented tensors to .pt files."""
    os.makedirs(save_path, exist_ok=True)
    for tensor, filename in data_list:
        save_file = os.path.join(save_path, filename)
        torch.save(tensor, save_file)
    logging.info(f"Saved {len(data_list)} augmented files to {save_path}")

def augment_and_save(input_path, output_path):
    data_list = load_data(input_path)
    os.makedirs(output_path, exist_ok=True)
    for tensor, fname in data_list:
        augmented = augment_tensor(tensor)
        # If 4D tensor returned a list, save each separately with suffix
        if isinstance(augmented, list):
            for idx, t in enumerate(augmented):
                save_file = os.path.join(output_path, f"{fname[:-3]}_aug{idx}.pt")
                torch.save(t, save_file)
        else:
            save_file = os.path.join(output_path, fname)
            torch.save(augmented, save_file)


def main():
    logging.info("=== Data Augmentation Started ===")
    augment_and_save("data/processed/train", "data/features/train")
    augment_and_save("data/processed/val", "data/features/val")
    logging.info("=== Data Augmentation Completed ===")

if __name__ == "__main__":
    main()
