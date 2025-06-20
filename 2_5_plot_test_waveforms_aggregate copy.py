import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gdown

file_id = "1PMhlLbWSIDBWXATdR3dngm4R-rJe0iTY"
url = f"https://drive.google.com/uc?id={file_id}"
output = "test_tennis_resized.csv"

gdown.download(url, output, quiet=False)
# Load the test dataset and plot the waveforms for each mode

test_df = pd.read_csv('test_tennis_resized.csv')

id_mode = {}
for i in range(1, 11):
    id_mode[i] = []

for i in test_df["unique_id"].unique():
    result = test_df[test_df['unique_id'] == i]
    id_mode[result['mode'].unique()[0]].append(int(i))

graph = []
for i in range(1, 11):
    graph.append(random.choice(id_mode[i]))

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(14, 20))
axs = axs.flatten()
sensor_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']


for idx, uid in enumerate(graph):
    sub = test_df[test_df['unique_id'] == uid]
    sub2 = sub[sub['cut_point'] < 30]

    # 計算六軸絕對值後相加的總波形
    total_wave = sub2[sensor_cols].abs().sum(axis=1).values

    ax = axs[idx]
    ax.plot(total_wave, label='Total Abs Sum')
    ax.set_title(f"mode = {sub2['mode'].iloc[0]}, unique_id = {uid}")
    ax.set_xlabel("index")
    ax.set_ylabel("summed abs value")
    ax.legend(fontsize='small')

plt.tight_layout()
plt.show()
