import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from scipy.stats import entropy
import pandas as pd
import lightgbm as lgb

train_data = pd.read_csv("train_tennis_resized.csv")
test_data = pd.read_csv("test_tennis_resized.csv")
submission_template = pd.read_csv("39_Test_Dataset/sample_submission.csv")

import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, find_peaks

# === 2. 標籤轉換 ===
train_data['gender'] = train_data['gender'].replace({1: 1, 2: 0}).astype(int)
train_data['hold racket handed'] = train_data['hold racket handed'].replace({1: 1, 2: 0}).astype(int)
train_data['play years'] = train_data['play years'].astype(int)
train_data['level'] = train_data['level'].astype(int)

# === 譜熵計算 ===
def spectral_entropy(signal, fs=85):
    freqs, psd = welch(signal, fs=fs)
    psd_norm = psd / np.sum(psd)
    return entropy(psd_norm)

# === Dominant Frequency ===
def dominant_frequency(signal, fs=85):
    freqs, psd = welch(signal, fs=fs)
    return freqs[np.argmax(psd)]

# === Spectral Centroid ===
def spectral_centroid(signal, fs=85):
    freqs, psd = welch(signal, fs=fs)
    return np.sum(freqs * psd) / np.sum(psd)

# === Bandwidth ===
def spectral_bandwidth(signal, fs=85):
    freqs, psd = welch(signal, fs=fs)
    centroid = spectral_centroid(signal, fs)
    return np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd))

# === Zero Crossing Rate ===
def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)

# === Vector Magnitude ===
def vector_magnitude(ax, ay, az):
    return np.sqrt(ax ** 2 + ay ** 2 + az ** 2)

# === 特徵聚合函數 (加強版) ===
def aggregate_features(df):
    sensor_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    agg_funcs = ['mean', 'std', 'min', 'max', 'median', lambda x: x.max() - x.min()]
    agg_dict = {col: agg_funcs for col in sensor_cols}
    
    agg_df = df.groupby('unique_id').agg(agg_dict)
    agg_df.columns = [f"{col}_{func.__name__ if callable(func) else func}" for col, func in agg_df.columns]
    agg_df = agg_df.reset_index()

    # 高階統計、頻域、形狀特徵
    extra_features = []
    for col in sensor_cols:
        extra = df.groupby('unique_id')[col].agg([
            lambda x: skew(x, bias=False),
            lambda x: kurtosis(x, bias=False),
            lambda x: spectral_entropy(x),
            lambda x: dominant_frequency(x),
            lambda x: spectral_centroid(x),
            lambda x: spectral_bandwidth(x),
            lambda x: zero_crossing_rate(x),
            lambda x: np.sqrt(np.mean(x**2)),   # RMS
            lambda x: np.sum(x**2),              # Energy
            lambda x: np.mean(np.abs(x)),        # Mean Absolute Value
            lambda x: np.percentile(x, 25),      # 25th percentile
            lambda x: np.percentile(x, 75),      # 75th percentile
            lambda x: np.percentile(x, 75) - np.percentile(x, 25), # IQR
            lambda x: len(find_peaks(x)[0]),     # Peak Count
            #lambda x: x[find_peaks(x)[0]].max() if len(find_peaks(x)[0]) > 0 else 0  # Max Peak Amplitude
            lambda x: x.iloc[find_peaks(x)[0]].max() if len(find_peaks(x)[0]) > 0 else 0

        ])
        extra.columns = [
            f'{col}_skew', f'{col}_kurtosis', f'{col}_spec_entropy',
            f'{col}_dominant_freq', f'{col}_spectral_centroid', f'{col}_spectral_bandwidth',
            f'{col}_zero_crossing_rate', f'{col}_rms', f'{col}_energy', f'{col}_mav',
            f'{col}_q25', f'{col}_q75', f'{col}_iqr',
            f'{col}_peak_count', f'{col}_max_peak'
        ]
        extra_features.append(extra)

    extra_df = pd.concat(extra_features, axis=1).reset_index()
    combined = pd.merge(agg_df, extra_df, on='unique_id')

    # Vector Magnitude
    vm_df = df.copy()
    vm_df['VM'] = vector_magnitude(vm_df['Ax'], vm_df['Ay'], vm_df['Az'])
    vm_agg = vm_df.groupby('unique_id')['VM'].agg([
        'mean', 'std', 'min', 'max', 'median',
        lambda x: x.max() - x.min(),
        lambda x: skew(x, bias=False),
        lambda x: kurtosis(x, bias=False),
        lambda x: spectral_entropy(x)
    ]).reset_index()
    vm_agg.columns = ['unique_id', 'VM_mean', 'VM_std', 'VM_min', 'VM_max', 'VM_median', 'VM_range', 'VM_skew', 'VM_kurtosis', 'VM_spec_entropy']

    combined = pd.merge(combined, vm_agg, on='unique_id')

    # 補上分類用欄位（如 cut_point、mode）
    cat_cols = [col for col in ['cut_point', 'mode'] if col in df.columns]
    if cat_cols:
        cat_df = df.groupby('unique_id')[cat_cols].agg('first').reset_index()
        combined = pd.merge(combined, cat_df, on='unique_id')

    return combined

X_train = aggregate_features(train_data)
X_test = aggregate_features(test_data)

# === 4. 標籤合併設定 ===
y_targets = {
    'gender': 'binary',
    'hold racket handed': 'binary',
    'play years': 'multi',
    'level': 'multi'
}
target_df = train_data.groupby('unique_id')[list(y_targets.keys())].first().reset_index()
X_train = pd.merge(X_train, target_df, on='unique_id')

# === 5. 預測與結果輸出 ===
output = pd.DataFrame()
output['unique_id'] = X_test['unique_id']

for target, t_type in y_targets.items():
    print(f"\n處理任務：{target}")

    X = X_train.drop(columns=['unique_id'] + list(y_targets.keys()))
    y = X_train[target]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMClassifier(
        objective='binary' if t_type == 'binary' else 'multiclass',
        num_class=y.nunique() if t_type == 'multi' else None,
        random_state=42,
        is_unbalance=(t_type == 'binary')
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)

    # === 效能輸出 ===
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:\n", cm)

    if t_type == 'binary':
        score = f1_score(y_val, y_pred)
        print(f"F1 Score: {score:.4f}")
    else:
        score = f1_score(y_val, y_pred, average='weighted')
        print(f"Weighted F1 Score: {score:.4f}")
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix for {target}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    # === 測試資料預測 ===
    X_test_input = X_test.drop(columns=['unique_id'])
    proba = model.predict_proba(X_test_input)

    if t_type == 'binary':
        idx_class_1 = list(model.classes_).index(1)
        output[target] = proba[:, idx_class_1]
    else:
        for i, cls in enumerate(model.classes_):
            output[f"{target}_{cls}"] = proba[:, i]

# === 6. 對齊格式與欄位順序 ===
template_cols = [
    'unique_id',
    'gender',
    'hold racket handed',
    'play years_0',
    'play years_1',
    'play years_2',
    'level_2',
    'level_3',
    'level_4',
    'level_5'
]
for col in template_cols:
    if col not in output.columns:
        output[col] = 0

output = output[template_cols].sort_values(by="unique_id").reset_index(drop=True)

# === 7. 輸出 CSV（避免科學記號，10 位小數） ===
for col in output.columns:
    if col != 'unique_id':
        output[col] = output[col].map(lambda x: f"{x:.10f}")

output.to_csv("Model_2_result/Submission_(no balance).csv", index=False)
print("\n輸出 Model_2_result/Submission_(no balance).csv")
