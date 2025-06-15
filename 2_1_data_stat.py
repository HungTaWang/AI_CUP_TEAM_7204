import pandas as pd
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
