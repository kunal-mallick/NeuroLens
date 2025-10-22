import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import mlflow
import logging
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1️⃣ Logging setup
# ============================================================
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    filename="log/validate_clinical_model.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# 2️⃣ Dataset Loader
# ============================================================
class ClinicalValDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.patients = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        img_path = os.path.join(self.data_dir, patient, "image.pt")
        mask_path = os.path.join(self.data_dir, patient, "mask.pt")

        image = torch.load(img_path, weights_only=True).permute(3, 0, 1, 2)  # [C,H,W,D]
        mask = torch.load(mask_path, weights_only=True).unsqueeze(0)          # [1,H,W,D]

        # Normalize
        image = (image - image.mean()) / (image.std() + 1e-5)
        return image.float(), mask.float()

# ============================================================
# 3️⃣ Model Definition (same as training)
# ============================================================
class ClinicalModel(nn.Module):
    def __init__(self, in_channels=4, num_classes=1, embed_dim=256, num_heads=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=2
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        b, c, h, w, d = x.shape
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.transformer(x)
        x = self.fc(x.mean(dim=1))
        return torch.sigmoid(x)

# ============================================================
# 4️⃣ Dice Score Metric
# ============================================================
def dice_score(preds, targets, eps=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum() + eps)

# ============================================================
# 5️⃣ Validation Function
# ============================================================
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_acc = 0, 0, 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs.squeeze(), masks.mean(dim=(2,3,4)))  # match output shape

            preds = outputs.detach().cpu()
            targets = masks.detach().cpu()

            dice = dice_score(preds, targets)
            acc = (preds.round() == targets).float().mean()

            total_loss += loss.item()
            total_dice += dice.item()
            total_acc += acc.item()

    n = len(loader)
    return total_loss / n, total_dice / n, total_acc / n

# ============================================================
# 6️⃣ Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dir = "data/features/val"
    model_path = "models/clinical/clinical_model.pt"

    # Load validation dataset
    val_dataset = ClinicalValDataset(val_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Model + Loss
    model = ClinicalModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.BCELoss()

    # MLflow
    mlflow.set_experiment("Clinical_Model_NeuroLens_Validation")
    with mlflow.start_run(run_name="Validation_Run"):
        val_loss, val_dice, val_acc = validate(model, val_loader, criterion, device)

        mlflow.log_metrics({
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_accuracy": val_acc
        })
        mlflow.log_param("model_path", model_path)

        logging.info(f"Validation Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, Acc: {val_acc:.4f}")
        print(f"✅ Validation Complete | Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | Acc: {val_acc:.4f}")

if __name__ == "__main__":
    main()
