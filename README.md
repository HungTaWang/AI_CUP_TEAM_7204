# TEAM_7204

## Contributed by 林哲宇、謝元皓、許家語、王泓達、王正廷

---

### 檔案結構

``` markdown
Code
├── 1_train_cleaning.py
├── 2_test_cleaning.py
├── 3_train_and_test_resize.py
├── 4_data_stat.py
├── 5_plot_train_waveforms_separate.py
├── 6_plot_train_waveforms_aggregate.py
├── 7_plot_test_waveforms_separate.py
├── 8_plot_test_waveforms_aggregate.py
├── 9_train_to_spectrogram.py
├── 10_test_to_spectrogram.py

File
├── train_info.csv: train 資料的 info
├── train_stat.csv: train 資料的每個 ID 含有幾筆資料
├── train_data: 放置訓練資料 txt 檔案之資料集
├── train_tennis.csv: train_data 處理後之 DataFrame
├── train_tennis_resized.csv: train_tennis.csv 擴增六軸加總之特徵
├── train_spectrogram_images.zip: 轉換 DataFrame 為光譜圖
├── test_info.csv: test 資料的 info
├── test_data: 放置測試資料 txt 檔案之資料集
├── test_tennis.csv: test_data 處理後之 DataFrame
├── test_tennis_resized.csv: test_tennis.csv 擴增六軸加總之特徵
├── test_spectrogram_images.zip: 轉換 DataFrame 為光譜圖
├── t_stat.csv: train 資料的每個 ID 含有幾筆資料
├──
└── requirements.txt
```

---

### 執行程式

#### 下載檔案

Google Drive 檔案連結：https://drive.google.com/drive/folders/16fJ4hJzgNerXktbx26lTl_5FQeear5CE?usp=drive_link

```bash
git clone https://github.com/HungTaWang/AI_CUP_TEAM_7204.git
cd AI_CUP_TEAM_7204
```

#### 環境建置

- Python 3.10+

- 必要套件：

    - os
    - re
    - pandas
    - numpy
    - random
    - seaborn 
    - matplotlib
    - scikit-learn
    - torch
    - torchvision
    - PIL
    - opencv-python
    - sklearn.manifold.TSNE

#### 套件安裝：

``` bash
python3 -m venv venv
source venv/bin/activate
``` 

``` bash
pip install -r requirements.txt
```

---

### 程式流程

#### 1. 資料清理與特徵計算

```bash
python 1_1_train_cleaning.py
python 1_2_test_cleaning.py
python 1_3_train_and_test_resize.py
```

#### 2. 資料統計與視覺化

```bash
python 2_1_data_stat.py
python 2_2_plot_train_waveforms_separate.py
python 2_3_plot_train_waveforms_aggregate.py
python 2_4_plot_test_waveforms_separate.py
python 2_5_plot_test_waveforms_aggregate.py
```

#### 3. 光譜圖轉換

```bash
python 3_1_train_to_spectrogram.py
python 3_2_test_to_spectrogram.py
```

---

### 資料說明

- train_data: 放置訓練 txt 檔案
- test_data: 放置測試 txt 檔案
