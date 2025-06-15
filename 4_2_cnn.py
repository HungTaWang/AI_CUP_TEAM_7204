# ============================================================================= LOAD DATA =====================================================================================
import gdown
import zipfile
import pandas as pd
from collections import Counter
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from PIL import Image


# Images
file_id = "1QUbpT1GARof37I0AE87WLnSxo0fQYxCZ"
url = f"https://drive.google.com/uc?id={file_id}"
train_spectrogram_image = 'train_spectrogram_images.zip'
gdown.download(url, train_spectrogram_image, quiet=False)

file_id = "18kHQzRa95mEBKQDwB07ySwDL0unx19Pk"
url = f"https://drive.google.com/uc?id={file_id}"
test_spectrogram_images = 'test_spectrogram_images.zip'
gdown.download(url, test_spectrogram_images, quiet=False)

with zipfile.ZipFile('/content/train_spectrogram_images.zip', 'r') as zip_ref:
    zip_ref.extractall('train_spectrogram_images')
with zipfile.ZipFile('/content/test_spectrogram_images.zip', 'r') as zip_ref:
    zip_ref.extractall('test_spectrogram_images')

# CSV file
file_id = "1k3cRLWCbm1DIKUxkoNbt2PVYlrZqksQd"
url = f"https://drive.google.com/uc?id={file_id}"
train_csv = 'train.csv'
gdown.download(url, train_csv, quiet=False)
train_df = pd.read_csv(train_csv)



# ============================================================================= TRAIN =====================================================================================
def compute_class_weights(dataset, label_key, num_classes):
    counts = Counter()
    for _, labels, _ in dataset:
        counts[labels[label_key].item()] += 1

    total = sum(counts.values())
    weights = [0] * num_classes
    for i in range(num_classes):
        weights[i] = total / (num_classes * counts[i])
    return torch.tensor(weights, dtype=torch.float)

class MultiSensorDataset(Dataset):
    def __init__(self, images_root, labels_csv, sensor_names=['Ax','Ay','Az','Gx','Gy','Gz'], transform=None):
        self.images_root = images_root
        self.sensor_names = sensor_names
        self.transform = transform

        # Load label data (if duplicates exist, keep only the first occurrence for each unique_id)
        labels_df = pd.read_csv(labels_csv)
        labels_df = labels_df.drop_duplicates(subset=['unique_id'], keep='first')

        # Create a mapping from unique_id to only play years and level targets
        self.labels_map = {
            str(row['unique_id']): {
                'play years': int(row['play years']),
                'level': int(row['level'])
            }
            for _, row in labels_df.iterrows()
        }

        # Collect sample identifiers from the subdirectories in images_root
        self.sample_ids = [
            d for d in os.listdir(images_root)
            if os.path.isdir(os.path.join(images_root, d))
        ]
        # Optionally, filter sample_ids to those that have corresponding labels
        self.sample_ids = [uid for uid in self.sample_ids if uid in self.labels_map]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        unique_id = self.sample_ids[idx]
        sample_dir = os.path.join(self.images_root, unique_id)

        # Read 7 spectrogram images
        sensor_images = []
        for sensor in self.sensor_names:
            img_path = os.path.join(sample_dir, f"{sensor}_spectrogram.png")
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            sensor_images.append(img)

        # Raw labels for this sample
        raw = self.labels_map[unique_id]

        # Level mapping
        level_mapping = {2: 0, 3: 1, 4: 2, 5: 3}

        # Convert labels to tensors
        mapped = {
            "play_years": torch.tensor(raw["play years"], dtype=torch.long),
            "level": torch.tensor(level_mapping[raw["level"]], dtype=torch.long),
        }

        return sensor_images, mapped, unique_id


class MultiSensorResNet(nn.Module):
    def __init__(self, sensor_count=7, pretrained=True):
        super(MultiSensorResNet, self).__init__()
        self.sensor_count = sensor_count

        # Backbone for play_years and level (shared branch)
        resnet_shared = models.resnet18(pretrained=pretrained)
        self.backbone_shared = nn.Sequential(*list(resnet_shared.children())[:-1])

        # Attention mechanism for the shared branch (512 channels)
        self.attention_fc_shared = nn.Linear(512, 1)

        # Fusion for the shared branch
        self.fusion_shared = nn.Linear(512, 512)

        # Classification heads
        self.head_play_years = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Linear(16, 3)
        )
        self.head_level = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Linear(16, 4)
        )

    def forward(self, images_list):
        shared_features = []

        for img in images_list:
            feat_shared = self.backbone_shared(img).squeeze()
            if feat_shared.dim() == 1:
                feat_shared = feat_shared.unsqueeze(0)
            shared_features.append(feat_shared)

        shared_tensor = torch.stack(shared_features, dim=1)

        attn_scores = self.attention_fc_shared(shared_tensor).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        aggregated_shared = torch.sum(shared_tensor * attn_weights, dim=1)
        fused_shared = self.fusion_shared(aggregated_shared)

        play_years_out = self.head_play_years(fused_shared)
        level_out = self.head_level(fused_shared)

        return {
            "play_years": play_years_out,
            "level": level_out
        }

    def extract_features(self, images_list):
        """
        This method extracts intermediate features from the ResNet backbone,
        before the classification heads.
        Returns:
            fused_shared: tensor of shape [B, 512]
        """
        shared_features = []

        for img in images_list:
            feat_shared = self.backbone_shared(img).squeeze()
            if feat_shared.dim() == 1:
                feat_shared = feat_shared.unsqueeze(0)
            shared_features.append(feat_shared)

        shared_tensor = torch.stack(shared_features, dim=1)

        attn_scores = self.attention_fc_shared(shared_tensor).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        aggregated_shared = torch.sum(shared_tensor * attn_weights, dim=1)
        fused_shared = self.fusion_shared(aggregated_shared)

        return fused_shared


# Define preprocessing transforms to match the expected input of ResNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create the dataset
dataset = MultiSensorDataset(
    images_root='/content/train_spectrogram_images/train_spectrogram_images',
    labels_csv='train.csv',
    sensor_names=['Ax','Ay','Az','Gx','Gy','Gz','sum_AG'],
    transform=preprocess
)

# Split into train and validation sets
total_samples = len(dataset)
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size

# Ensure reproducibility
torch.manual_seed(10)
train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize model
model = MultiSensorResNet(pretrained=True)
model.to(device)
model.train()

# Loss function
weights_play_years = compute_class_weights(train_dataset, 'play_years', 3).to(device)
weights_level = compute_class_weights(train_dataset, 'level', 4).to(device)
criterion_play_years = nn.CrossEntropyLoss(weight=weights_play_years)
criterion_level = nn.CrossEntropyLoss(weight=weights_level)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 25

best_val_loss = float('inf')
model_save_path = 'best_model.pth'

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (imgs_batch, labels, _) in enumerate(train_loader):
        imgs_batch = [x.to(device) for x in imgs_batch]
        target_play_years = labels['play_years'].to(device)
        target_level = labels['level'].to(device)

        optimizer.zero_grad()
        outputs = model(imgs_batch)

        loss_play_years = criterion_play_years(outputs["play_years"], target_play_years)
        loss_level = criterion_level(outputs["level"], target_level)


        total_loss = loss_play_years + loss_level

        total_loss.backward()
        optimizer.step()

        batch_loss = total_loss.item()
        epoch_loss += batch_loss

        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {batch_loss:.4f}")

    avg_train_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0

    # New lists to collect predictions and labels for ROC AUC
    all_level_probs = []
    all_level_targets = []

    with torch.no_grad():
        for sensor_images_batch, labels, _ in val_loader:
            sensor_images_batch = [img.to(device) for img in sensor_images_batch]
            target_play_years = labels['play_years'].to(device)
            target_level = labels['level'].to(device)

            outputs = model(sensor_images_batch)

            loss_play_years = criterion_play_years(outputs["play_years"], target_play_years)
            loss_level = criterion_level(outputs["level"], target_level)

            total_val_loss = loss_play_years + loss_level
            val_loss += total_val_loss.item()

            # Collect predictions and true labels for ROC AUC
            probs_level = F.softmax(outputs["level"], dim=1)  # [B, 4]
            all_level_probs.append(probs_level.cpu())
            all_level_targets.append(target_level.cpu())

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

    # ROC AUC computation
    all_level_probs = torch.cat(all_level_probs, dim=0).numpy()
    all_level_targets = torch.cat(all_level_targets, dim=0).numpy()
    y_true_one_hot = label_binarize(all_level_targets, classes=[0, 1, 2, 3])
    roc_auc = roc_auc_score(y_true_one_hot, all_level_probs, average="micro", multi_class="ovr")
    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Micro-averaged ROC AUC (Level): {roc_auc:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"New best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")



# ============================================================================= TEST =====================================================================================
file_id = "1PMhlLbWSIDBWXATdR3dngm4R-rJe0iTY"
url = f"https://drive.google.com/uc?id={file_id}"
test_csv = 'your_file.csv'
gdown.download(url, test_csv, quiet=False)
test_df = pd.read_csv(test_csv)


# Define a test dataset class that only loads images without labels
class TestMultiSensorDataset(Dataset):
    def __init__(self, images_root, sensor_names=['Ax','Ay','Az','Gx','Gy','Gz'], transform=None):
        """
        Args:
            images_root (str): Root directory containing subdirectories for each unique_id.
            sensor_names (list): List of sensor names to match the image filenames.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.images_root = images_root
        self.sensor_names = sensor_names
        self.transform = transform

        # Collect sample identifiers from the subdirectories in images_root
        self.sample_ids = [
            d for d in os.listdir(images_root)
            if os.path.isdir(os.path.join(images_root, d))
        ]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        unique_id = self.sample_ids[idx]
        sample_dir = os.path.join(self.images_root, unique_id)

        # Read 6 spectrogram images
        sensor_images = []
        for sensor in self.sensor_names:
            img_path = os.path.join(sample_dir, f"{sensor}_spectrogram.png")
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            sensor_images.append(img)

        return sensor_images, unique_id

# Define preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create test dataset and dataloader
test_dataset = TestMultiSensorDataset(
    images_root='/content/test_spectrogram_images/test_spectrogram_images',
    sensor_names=['Ax','Ay','Az','Gx','Gy','Gz','sum_AG'],
    transform=preprocess
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model
model = MultiSensorResNet(pretrained=False)  # Set pretrained to False for inference
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()
print("Loaded best model for prediction.")

# Initialize containers for results
all_ids = []
play_years_pred_proba = []
level_pred_proba = []

# Run inference
with torch.no_grad():
    for sensor_images_batch, unique_ids in test_loader:
        sensor_images_batch = [img.to(device) for img in sensor_images_batch]
        outputs = model(sensor_images_batch)

        # Convert unique_ids to list if it's not already
        if isinstance(unique_ids, torch.Tensor):
            unique_ids = unique_ids.tolist()
        all_ids.extend(unique_ids)

        # For multiclass heads, apply softmax
        play_probs = torch.softmax(outputs['play_years'], dim=1).cpu().numpy()  # shape: [B,3]
        level_probs = torch.softmax(outputs['level'], dim=1).cpu().numpy()  # shape: [B,4]

        play_years_pred_proba.extend(play_probs)
        level_pred_proba.extend(level_probs)

# Convert to numpy arrays
play_years_pred_proba = np.array(play_years_pred_proba)
level_pred_proba = np.array(level_pred_proba)

# Create results dataframe
results_df = pd.DataFrame({'unique_id': all_ids})

# Save probabilities for play years (3 classes)
for i in range(play_years_pred_proba.shape[1]):
    results_df[f'play years_{i}'] = play_years_pred_proba[:, i]

# Save probabilities for level (4 classes, mapped from {2, 3, 4, 5})
for i in range(level_pred_proba.shape[1]):
    # Adding i+2 to match original level labels (2, 3, 4, 5)
    results_df[f'level_{i+2}'] = level_pred_proba[:, i]

# Round values for readability
results_df = results_df.round(6)

# Save results
results_df = results_df.sort_values(by='unique_id', ascending=True)
results_df.to_csv("test_predictions.csv", index=False)
print("Saved test predictions to test_predictions.csv")
