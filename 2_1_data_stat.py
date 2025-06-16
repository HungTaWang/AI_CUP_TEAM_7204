import pandas as pd
import matplotlib.pyplot as plt
import gdown

file_id = "1i0QbO72j-jHqiAE5KZFquvE2ppeHRxQH"
url = f"https://drive.google.com/uc?id={file_id}"
output = "train_tennis_resized.csv"

gdown.download(url, output, quiet=False)

file_id = "1PMhlLbWSIDBWXATdR3dngm4R-rJe0iTY"
url = f"https://drive.google.com/uc?id={file_id}"
output = "test_tennis_resized.csv"

gdown.download(url, output, quiet=False)

# Check Target Distribution in Training Data

train_df = pd.read_csv('train_tennis_resized.csv')
targets = ['gender', 'hold racket handed', 'play years', 'level']
features = ['player_id', 'mode'],

print("所有 player_id：", sorted(train_df['player_id'].unique().tolist()))
print("所有 mode：", sorted(train_df['mode'].unique().tolist()))

print("各分類目標的類別分布統計：")
for col in targets:
    print("-" * 40)
    print(train_df[col].value_counts().sort_index())

# Check the distribution of 'mode' in the training data
print("\nDistribution of 'mode' in the training data:")
id_mode = {}
for i in range(1, 11):
    id_mode[i] = []

for i in train_df["unique_id"].unique():
    result = train_df[train_df['unique_id'] == i]
    id_mode[result['mode'].unique()[0]].append(int(i))
sum = 0
for i in range(1,11):
    sum += len(id_mode[i])
for i in range(1,11):
    print(f"mode {i:2d}: {len(id_mode[i]):3d} ({(len(id_mode[i])/sum):.2%})")

print("Total: ", sum)

# Check the distribution of 'mode' in the test data

test_df = pd.read_csv('test_tennis_resized.csv')
print("\nDistribution of 'mode' in the test data:")
id_mode = {}
for i in range(1, 11):
    id_mode[i] = []

for i in test_df["unique_id"].unique():
    result = test_df[test_df['unique_id'] == i]
    id_mode[result['mode'].unique()[0]].append(int(i))

sum = 0
for i in range(1,11):
    sum += len(id_mode[i])
for i in range(1,11):
    print(f"mode {i:2d}: {len(id_mode[i]):3d} ({(len(id_mode[i])/sum):.2%})")
print("Total: ", sum)


# === mode 分布：訓練資料 ===
train_mode_counts = train_df.groupby('mode')['unique_id'].nunique().sort_index()
train_mode_pct = (train_mode_counts / train_mode_counts.sum()) * 100

plt.figure(figsize=(8,5))
plt.bar(train_mode_counts.index, train_mode_counts.values, color='skyblue')
for i, v in enumerate(train_mode_counts.values):
    plt.text(train_mode_counts.index[i], v + 1, f"{train_mode_pct.values[i]:.2f}%", ha='center', va='bottom')
plt.xlabel("Mode")
plt.ylabel("Unique ID Count")
plt.title("Train Data Mode Distribution")
plt.xticks(train_mode_counts.index)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === mode 分布：測試資料 ===
test_mode_counts = test_df.groupby('mode')['unique_id'].nunique().sort_index()
test_mode_pct = (test_mode_counts / test_mode_counts.sum()) * 100

plt.figure(figsize=(8,5))
plt.bar(test_mode_counts.index, test_mode_counts.values, color='orange')
for i, v in enumerate(test_mode_counts.values):
    plt.text(test_mode_counts.index[i], v + 1, f"{test_mode_pct.values[i]:.2f}%", ha='center', va='bottom')
plt.xlabel("Mode")
plt.ylabel("Unique ID Count")
plt.title("Test Data Mode Distribution")
plt.xticks(test_mode_counts.index)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === target 分布圖產生函數 ===
def plot_target_distribution(df, target_col, color):
    target_counts = df[target_col].value_counts().sort_index()
    target_pct = (target_counts / target_counts.sum()) * 100

    plt.figure(figsize=(7,4))
    plt.bar(target_counts.index.astype(str), target_counts.values, color=color)
    for i, v in enumerate(target_counts.values):
        plt.text(i, v + 1, f"{target_pct.values[i]:.2f}%", ha='center', va='bottom')
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.title(f"Train Data {target_col} Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# === 繪製四個 target ===
plot_target_distribution(train_df, 'gender', 'lightgreen')
plot_target_distribution(train_df, 'hold racket handed', 'lightcoral')
plot_target_distribution(train_df, 'play years', 'lightseagreen')
plot_target_distribution(train_df, 'level', 'plum')
