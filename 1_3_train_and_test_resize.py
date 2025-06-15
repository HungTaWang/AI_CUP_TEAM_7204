import pandas as pd

import pandas as pd

df = pd.read_csv("train_tennis.csv")
df['sum_AG'] = df[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']].abs().sum(axis=1)

df.to_csv("train_tennis_resized.csv", index=False)

df = pd.read_csv("test_tennis.csv")
df['sum_AG'] = df[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']].abs().sum(axis=1)

df.to_csv("test_tennis_resized.csv", index=False)
