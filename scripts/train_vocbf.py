import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
import random

# ------------------ 1. Define the Neural Network ------------------
class VOCBFNet(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN for the 64x64 RGB image
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # The 64x64 image reduces to 64 * 8 * 8 = 4096 features
        # We add the 2 joint angles (Pan, Tilt), making it 4098
        self.mlp = nn.Sequential(
            nn.Linear(4096 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs the predicted continuous distance
        )

    def forward(self, img, joints):
        img_features = self.cnn(img)
        combined = torch.cat((img_features, joints), dim=1)
        return self.mlp(combined)

# ------------------ 2. Define the Dataset ------------------
class SafetyDataset(Dataset):

    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            raw_data = pickle.load(f)
            
        # Separate the data into "Hits" (distance < 8.0) and "Misses" (distance == 8.0)
        bed_hits = [item for item in raw_data if item["distance"] < 8.0]
        misses = [item for item in raw_data if item["distance"] >= 8.0]
        
        print(f"Original Data: {len(bed_hits)} hits, {len(misses)} misses.")
        
        # Balance the dataset: take an equal number of misses as hits
        # This prevents the network from becoming "lazy"
        balanced_misses = random.sample(misses, min(len(bed_hits), len(misses)))
        
        self.data = bed_hits + balanced_misses
        random.shuffle(self.data)
        
        print(f"Balanced Data: {len(self.data)} total samples used for training.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert image to PyTorch format: (Channels, Height, Width) and normalize to [0, 1]
        img = torch.tensor(item["image"], dtype=torch.float32).permute(2, 0, 1) / 255.0
        joints = torch.tensor(item["q"], dtype=torch.float32)
        
        # Target is the continuous distance we raycasted
        target = torch.tensor([item["distance"]], dtype=torch.float32)
        
        return img, joints, target

# ------------------ 3. Training Loop ------------------
def train():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_path = PROJECT_ROOT / "data" / "offline_dataset.pkl"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = SafetyDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = VOCBFNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # Mean Squared Error to learn the exact distance

    epochs = 20
    for epoch in range(epochs):
        total_loss = 0
        for imgs, joints, targets in dataloader:
            imgs, joints, targets = imgs.to(device), joints.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(imgs, joints)
            
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

    # 4. Freeze and Save Weights
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), model_dir / "vocbf_weights.pth")
    print("Training complete. Weights frozen and saved to models/vocbf_weights.pth")

if __name__ == "__main__":
    train()