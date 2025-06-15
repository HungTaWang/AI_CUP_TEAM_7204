import pandas as pd
import numpy as np
import os
import re

folder_path = 'train_data'
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
df.to_csv("train_stat.csv", index=False)

# read train_info.csv and read corresponding text files

df = pd.read_csv("train_info.csv")

folder_path = "train_data"
data_list = []


for i in list(df["unique_id"]):
    file_path = os.path.join(folder_path, f"{i}.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        data_list.append(content)

# Create a DataFrame with the read data and save it to a CSV file

train_info = pd.read_csv("train_info.csv")

def parse_cut_points(cut_str):
    return list(map(int, re.findall(r'\d+', cut_str)))

train_info["cut_point"] = train_info["cut_point"].apply(parse_cut_points)

records = []

for _, row in train_info.iterrows():
    uid = row["unique_id"]
    cut_points = row["cut_point"]
    file_path = f"train_data/{uid}.txt"

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
                "player_id": row["player_id"],
                "mode": row["mode"],
                "gender": row["gender"],
                "hold racket handed": row["hold racket handed"],
                "play years": row["play years"],
                "level": row["level"],
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

train_info = pd.read_csv("train_info.csv")

train_info["cut_point"] = train_info["cut_point"].apply(parse_cut_points)
cut_dict = dict(zip(train_info["unique_id"], train_info["cut_point"]))

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

df.to_csv("train_tennis.csv", index=False)
