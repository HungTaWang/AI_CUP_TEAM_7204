import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lightgbm as lgb
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, find_peaks
import gdown
import zipfile

# === Download Data ===

file_id = "1A7UHfKQs9nX26xX1o7xDfQ3W1OO7gSrH"
url = f"https://drive.google.com/uc?id={file_id}"
output = "train_tennis.csv"

gdown.download(url, output, quiet=False)

file_id = "18X-uh5egqy-YBICcKtNn9RrNRMSpdfJ7"
url = f"https://drive.google.com/uc?id={file_id}"
output = "test_tennis.csv"

gdown.download(url, output, quiet=False)


file_id = "1aLZSzB1wZAt_mbI0h0MO4aJudQCQdGpX"
url = f"https://drive.google.com/uc?id={file_id}"
output = "best_model.pth"

gdown.download(url, output, quiet=False)

file_id = "18kHQzRa95mEBKQDwB07ySwDL0unx19Pk"
url = f"https://drive.google.com/uc?id={file_id}"
output = "test_spectrogram_images.zip"
gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(".")

file_id = "1NODVZQlCPTGo-UhIZ0wz4Cr2WCj9ajUD"
url = f"https://drive.google.com/uc?id={file_id}"
submission_csv = 'sample_submission.csv'
gdown.download(url, submission_csv, quiet=False)

# === CNN ===
class MultiSensorResNet(torch.nn.Module):
    def __init__(self, sensor_count=7, pretrained=False):
        super().__init__()
        from torchvision import models
        resnet_shared = models.resnet18(pretrained=pretrained)
        self.backbone_shared = torch.nn.Sequential(*list(resnet_shared.children())[:-1])
        self.attention_fc_shared = torch.nn.Linear(512, 1)
        self.fusion_shared = torch.nn.Linear(512, 512)
        self.head_play_years = torch.nn.Linear(512, 3)
        self.head_level = torch.nn.Linear(512, 4)

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

def cnn_predict(model_path, images_root, sensor_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiSensorResNet(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = TestMultiSensorDataset(images_root, sensor_names, transform)
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
    return pd.DataFrame(results)

# === LGBM ===
def aggregate_features(df):
    sensor_cols = ['Ax','Ay','Az','Gx','Gy','Gz']
    agg_funcs = ['mean','std','min','max','median', lambda x: x.max()-x.min()]
    agg_df = df.groupby('unique_id').agg({col: agg_funcs for col in sensor_cols})
    agg_df.columns = [f"{c}_{f.__name__ if callable(f) else f}" for c,f in agg_df.columns]
    agg_df = agg_df.reset_index()
    return agg_df

def lgbm_predict(train_df, test_df):
    X_train = aggregate_features(train_df)
    X_test = aggregate_features(test_df)

    target_df = train_df.groupby('unique_id')[['gender', 'hold racket handed', 'play years', 'level']].first().reset_index()
    X_train = pd.merge(X_train, target_df, on='unique_id')

    output = pd.DataFrame({'unique_id': X_test['unique_id']})

    for target, obj, num_class in [('gender','binary',None), ('hold racket handed','binary',None),
                                   ('play years','multiclass',3), ('level','multiclass',4)]:
        X = X_train.drop(columns=['unique_id','gender','hold racket handed','play years','level'])
        y = X_train[target]
        X_tr,X_val,y_tr,y_val = train_test_split(X,y,test_size=0.2,random_state=42)
        model = lgb.LGBMClassifier(objective=obj,num_class=num_class,random_state=42)
        model.fit(X_tr,y_tr)
        proba = model.predict_proba(X_test.drop(columns=['unique_id']))
        if obj == 'binary':
            output[target] = proba[:, list(model.classes_).index(1)]
        else:
            for i,cls in enumerate(model.classes_):
                output[f'{target}_{cls}'] = proba[:,i]
    return output

# === MAIN PROCESS ===
def main():
    # CNN 
    cnn_df = cnn_predict(
        model_path="best_model.pth",
        images_root="test_spectrogram_images/test_spectrogram_images",
        sensor_names=['Ax','Ay','Az','Gx','Gy','Gz','sum_AG']
    )

    # LGBM 
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    lgbm_df = lgbm_predict(train_df, test_df)

    # MERGE
    final_df = pd.merge(lgbm_df, cnn_df, on="unique_id", how="outer")
    final_df = final_df.sort_values(by="unique_id").reset_index(drop=True)

    # OUTPUT
    final_df.to_csv("Submission.csv", index=False, float_format='%.10f')
    print("已輸出 Submission.csv")

if __name__ == "__main__":
    main()
