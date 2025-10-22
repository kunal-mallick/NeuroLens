import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import logging
from tqdm import tqdm

# ========== Logging Setup ==========
LOG_PATH = "logs/clinical_model.log"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)

# ========== Dataset ==========
class ClinicalDataset2D(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for patient in os.listdir(root_dir):
            img_path = os.path.join(root_dir, patient, "image.pt")
            mask_path = os.path.join(root_dir, patient, "mask.pt")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                image = torch.load(img_path, weights_only=True)
                mask = torch.load(mask_path, weights_only=True)
                for i in range(image.shape[2]):
                    self.samples.append((image[:, :, i, :], mask[:, :, i, :]))
        logging.info(f"Total slices loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, mask = self.samples[idx]
        return image.permute(2, 0, 1).float(), mask.permute(2, 0, 1).float()

# ========== Model ==========
class ClinicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ========== Training Function ==========
def train_epoch(model, loader, criterion, optimizer, device, scaler, epoch, total_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", ncols=100)):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with autocast("cuda", enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            logging.info(f"Epoch [{epoch}], Batch [{batch_idx}/{len(loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    logging.info(f"Epoch [{epoch}] completed. Avg Loss: {avg_loss:.4f}")
    return avg_loss

# ========== Main ==========
def main():
    logging.info("Starting Clinical Model Training")

    data_params = {
        "train_dir": "data/processed/train",
        "val_dir": "data/processed/val",
        "batch_size": 32,
        "epochs": 5,
        "lr": 1e-3
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    train_dataset = ClinicalDataset2D(data_params["train_dir"])
    val_dataset = ClinicalDataset2D(data_params["val_dir"])

    train_loader = DataLoader(train_dataset, batch_size=data_params["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=data_params["batch_size"], shuffle=False, num_workers=4)

    model = ClinicalModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=data_params["lr"])
    scaler = GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, data_params["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, data_params["epochs"])
        logging.info(f"Epoch [{epoch}/{data_params['epochs']}] - Train Loss: {train_loss:.4f}")

    os.makedirs("models/clinical", exist_ok=True)
    torch.save(model.state_dict(), "models/clinical/clinical_model.pt")
    logging.info("Model saved at models/clinical/clinical_model.pt")

    # Lite version (optimized for mobile)
    lite_model = torch.jit.script(model)
    lite_model.save("models/clinical/clinical_model_lite.pt")
    logging.info("Lite model saved at models/clinical/clinical_model_lite.pt")

if __name__ == "__main__":
    main()
