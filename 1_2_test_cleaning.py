import pandas as pd
import numpy as np
import os
import re
import gdown
import zipfile

file_id = "129hzbJt_EGzCgQdgwO5ILpmy901Sl7C-"
url = f"https://drive.google.com/uc?id={file_id}"
output = "test_data.zip"
gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(".")

file_id = "1_IapPVMVgr1kK-e-DcwzH3hXIQlQ8fyl"
url = f"https://drive.google.com/uc?id={file_id}"
output = "test_info.csv"

gdown.download(url, output, quiet=False)
# test_stat.csv generation

folder_path = 'test_data'
data = []

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                line_count = sum(1 for line in lines if line.strip())
                data.append({'file': filename, 'row': line_count})
        except Exception as e:
            data.append({'file': filename, 'row': f'error: {str(e)}'})

df = pd.DataFrame(data)
df.to_csv("test_stat.csv", index=False)

# read test_info.csv and read corresponding text files

df = pd.read_csv("test_info.csv")

folder_path = "test_data"
data_list = []


for i in list(df["unique_id"]):
    file_path = os.path.join(folder_path, f"{i}.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        data_list.append(content)

# Create a DataFrame with the read data and save it to a CSV file

test_info = pd.read_csv("test_info.csv")

def parse_cut_points(cut_str):
    return list(map(int, re.findall(r'\d+', cut_str)))

test_info["cut_point"] = test_info["cut_point"].apply(parse_cut_points)

records = []

for _, row in test_info.iterrows():
    uid = row["unique_id"]
    cut_points = row["cut_point"]
    file_path = f"test_data/{uid}.txt"

    if not os.path.exists(file_path):
        print(f"檔案不存在：{file_path}")
        continue

    with open(file_path, "r") as f:
        data = [list(map(int, line.strip().split())) for line in f.readlines()]
        data = np.array(data)

    for i in range(27):
        start = cut_points[i]
        end = cut_points[i+1] if i < 26 else len(data)
        swing_data = data[start:end]

        for t, sensor in enumerate(swing_data):
            records.append({
                "unique_id": uid,
                "cut_point": start,
                "swing_id": i + 1,
                "mode": row["mode"],
                "Ax": sensor[0],
                "Ay": sensor[1],
                "Az": sensor[2],
                "Gx": sensor[3],
                "Gy": sensor[4],
                "Gz": sensor[5],
            })

df = pd.DataFrame(records)

# Revise cut points and swing IDs

def parse_cut_points(cut_str):
    return list(map(int, re.findall(r'\d+', cut_str)))

test_info = pd.read_csv("test_info.csv")

test_info["cut_point"] = test_info["cut_point"].apply(parse_cut_points)
cut_dict = dict(zip(test_info["unique_id"], test_info["cut_point"]))

new_cut_point = []
new_swing_id = []

current_cut = 0
swing_counter = 0
prev_unique_id = df.loc[0, 'unique_id']

for i in range(len(df)):
    if swing_counter in cut_dict[prev_unique_id]:
        current_cut += 1
        swing_counter = 0

    swing_counter += 1

    if df["unique_id"][i] != prev_unique_id:
        current_cut = 1
        swing_counter = 1
    new_cut_point.append(current_cut)
    new_swing_id.append(swing_counter)
    prev_unique_id = df["unique_id"][i]    
    
df['cut_point'] = new_cut_point
df['swing_id'] = new_swing_id

df.to_csv("test_tennis.csv", index=False)

