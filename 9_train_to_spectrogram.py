import os
import pandas as pd
import numpy as np

train_df = pd.read_csv('train_tennis_resized.csv')

unique_ids = train_df['unique_id'].unique()
output_dir = 'train_spectrogram_images'
os.makedirs(output_dir, exist_ok=True)
sensor_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz','sum_AG']

def generate_spectrogram(signal_data):
    """
    Dummy spectrogram generation.
    Replace this with your actual spectrogram generation code that returns a figure object.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.specgram(signal_data, NFFT=64, Fs=100, noverlap=32)
    ax.axis('off')
    plt.tight_layout()
    return fig

def save_spectrogram(fig, save_path):
    """
    Save the generated spectrogram to disk.
    """
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    import matplotlib.pyplot as plt
    plt.close(fig)

print("Generating spectrogram images ...")
for uid in unique_ids:
    print(f"Processing unique_id: {uid}")
    uid_df = train_df[train_df['unique_id'] == uid]

    # Create a folder for this unique_id to store sensor images.
    sensor_dir = os.path.join(output_dir, str(uid))
    os.makedirs(sensor_dir, exist_ok=True)

    for sensor in sensor_names:
        # Extract the sensor's time-series values; adjust to your signal extraction
        signal_data = uid_df[sensor].values.astype(np.float32)
        fig = generate_spectrogram(signal_data)
        save_path = os.path.join(sensor_dir, f"{sensor}_spectrogram.png")
        save_spectrogram(fig, save_path)

print("Spectrogram image generation completed.")