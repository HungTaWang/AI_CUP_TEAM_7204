import pandas as pd
import gdown

file_id = "1A7UHfKQs9nX26xX1o7xDfQ3W1OO7gSrH"
url = f"https://drive.google.com/uc?id={file_id}"
output = "train_tennis.csv"

gdown.download(url, output, quiet=False)

file_id = "18X-uh5egqy-YBICcKtNn9RrNRMSpdfJ7"
url = f"https://drive.google.com/uc?id={file_id}"
output = "test_tennis.csv"

gdown.download(url, output, quiet=False)

df = pd.read_csv("train_tennis.csv")
df['sum_AG'] = df[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']].abs().sum(axis=1)

df.to_csv("train_tennis_resized.csv", index=False)

df = pd.read_csv("test_tennis.csv")
df['sum_AG'] = df[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']].abs().sum(axis=1)

df.to_csv("test_tennis_resized.csv", index=False)
