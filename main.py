import os
import pandas as pd
import numpy as np
import gdown
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, find_peaks

# ============================== DOWNLOAD & PREPARE ==============================

def download_and_prepare_data():
    files = [
        # (file_id, output_name)
        ("1A7UHfKQs9nX26xX1o7xDfQ3W1OO7gSrH", "train.csv"),
        ("18X-uh5egqy-YBICcKtNn9RrNRMSpdfJ7", "test.csv"),
        ("1aLZSzB1wZAt_mbI0h0MO4aJudQCQdGpX", "best_cnn_model.zip"),
        ("18kHQzRa95mEBKQDwB07ySwDL0unx19Pk", "test_spectrogram_images.zip"),
        ("1NODVZQlCPTGo-UhIZ0wz4Cr2WCj9ajUD", "sample_submission.csv"),
    ]

    for file_id, output in files:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

    # Unzip CNN model
    with zipfile.ZipFile("best_cnn_model.zip", "r") as zip_ref:
        zip_ref.extractall(".")

    # Unzip test images
    with zipfile.ZipFile("test_spectrogram_images.zip", "r") as zip_ref:
        zip_ref.extractall(".")

# ============================== CNN MODEL ==============================

class MultiSensorResNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        resnet_shared = models.resnet18(pretrained=pretrained)
        self.backbone_shared = nn.Sequential(*list(resnet_shared.children())[:-1])
        self.attention_fc_shared = nn.Linear(512, 1)
        self.fusion_shared = nn.Linear(512, 512)
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
        shared_features = [self.backbone_shared(img).squeeze() for img in images_list]
        shared_tensor = torch.stack(shared_features, dim=1)
        attn_scores = self.attention_fc_shared(shared_tensor).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        aggregated_shared = torch.sum(shared_tensor * attn_weights, dim=1)
        fused_shared = self.fusion_shared(aggregated_shared)
        return {
            "play_years": self.head_play_years(fused_shared),
            "level": self.head_level(fused_shared)
        }

class TestMultiSensorDataset(Dataset):
    def __init__(self, images_root, sensor_names, transform):
        self.sample_ids = [d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))]
        self.images_root = images_root
        self.sensor_names = sensor_names
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        unique_id = self.sample_ids[idx]
        sample_dir = os.path.join(self.images_root, unique_id)
        images = []
        for sensor in self.sensor_names:
            img = Image.open(os.path.join(sample_dir, f"{sensor}_spectrogram.png")).convert("RGB")
            images.append(self.transform(img))
        return images, unique_id

def cnn_predict_wrapper():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiSensorResNet(pretrained=False)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = TestMultiSensorDataset(
        images_root="test_spectrogram_images",
        sensor_names=['Ax','Ay','Az','Gx','Gy','Gz','sum_AG'],
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    results = []
    with torch.no_grad():
        for imgs, ids in loader:
            imgs = [i.to(device) for i in imgs]
            out = model(imgs)
            play_probs = F.softmax(out["play_years"], dim=1).cpu().numpy()
            level_probs = F.softmax(out["level"], dim=1).cpu().numpy()
            for i, uid in enumerate(ids):
                row = {'unique_id': uid}
                for j in range(3):
                    row[f'play years_{j}'] = play_probs[i, j]
                for j in range(4):
                    row[f'level_{j+2}'] = level_probs[i, j]
                results.append(row)
    df = pd.DataFrame(results)
    return df

# ============================== LIGHTGBM ==============================

def aggregate_features(df):
    sensor_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    agg_funcs = ['mean', 'std', 'min', 'max', 'median', lambda x: x.max() - x.min()]
    agg_df = df.groupby('unique_id').agg({col: agg_funcs for col in sensor_cols})
    agg_df.columns = [f"{c}_{f.__name__ if callable(f) else f}" for c,f in agg_df.columns]
    agg_df = agg_df.reset_index()
    return agg_df

def lgbm_predict_wrapper():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    X_train = aggregate_features(train_df)
    X_test = aggregate_features(test_df)
    target_df = train_df.groupby('unique_id')[['gender', 'hold racket handed', 'play years', 'level']].first().reset_index()
    X_train = pd.merge(X_train, target_df, on='unique_id')

    output = pd.DataFrame({'unique_id': X_test['unique_id']})

    for target, obj, num_class in [('gender','binary',None), ('hold racket handed','binary',None),
                                   ('play years','multiclass',3), ('level','multiclass',4)]:
        X = X_train.drop(columns=['unique_id','gender','hold racket handed','play years','level'])
        y = X_train[target]
        model = lgb.LGBMClassifier(objective=obj, num_class=num_class, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X_test.drop(columns=['unique_id']))
        if obj == 'binary':
            output[target] = proba[:, list(model.classes_).index(1)]
        else:
            for i,cls in enumerate(model.classes_):
                output[f'{target}_{cls}'] = proba[:,i]
    return output

# ============================== MERGE AND SAVE ==============================

def merge_and_save_submission(cnn_df, lgbm_df):

    # 確保 unique_id 都轉成字串（或都轉 int 也可，只要一致）
    lgbm_df["unique_id"] = lgbm_df["unique_id"].astype(str)
    cnn_df["unique_id"] = cnn_df["unique_id"].astype(str)

    # 再 merge
    merged_df = pd.merge(lgbm_df, cnn_df, on="unique_id", how="outer")


    merged_df = merged_df.sort_values(by="unique_id").reset_index(drop=True)
    merged_df.to_csv("Submission.csv", index=False, float_format="%.10f")
    print("已輸出 Submission.csv")

# ============================== MAIN ==============================

def main():
    download_and_prepare_data()
    cnn_df = cnn_predict_wrapper()
    lgbm_df = lgbm_predict_wrapper()
    merge_and_save_submission(cnn_df, lgbm_df)

if __name__ == "__main__":
    main()
