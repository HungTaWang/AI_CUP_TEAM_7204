# TEAM_7204

## Contributed by 林哲宇、謝元皓、許家語、王泓達、王正廷

### 檔案結構

``` markdown
Code
├── main.py: 最終預測結果生成
├── 1_1_train_cleaning.py: train 資料轉為 DataFrame，並輸出 train_info.csv
├── 1_2_test_cleaning.py: test 資料轉為 DataFrame，並輸出 test_info.csv
├── 1_3_train_and_test_resize.py: 將 train 和 test 資料擴增六軸加總之特徵
├── 2_1_data_stat.py: 輸出 mode 及 target 之分布
├── 2_2_plot_train_waveforms_separate.py: 繪製 train 資料六軸個別之波形圖
├── 2_3_plot_train_waveforms_aggregate.py: 繪製 train 資料六軸加總之波形圖
├── 2_4_plot_test_waveforms_separate.py: 繪製 test 資料六軸個別之波形圖
├── 2_5_plot_test_waveforms_aggregate.py: 繪製 test 資料六軸加總之波形圖
├── 3_1_train_to_spectrogram.py: train 資料轉為光譜圖
├── 3_2_test_to_spectrogram.py: test 資料轉為光譜圖
├── 4_1_cnn.py: 用 CNN 預測 play years 和 level 
└── 4_2_lgbm.py: 用 LGBM 預測 gender 和 hold racket hand 並且整併 4_1_cnn.py 之預測結果，生成最終 Submission.csv

Download Packages
└── requirements.txt
```

---

### 執行程式

Google Drive 檔案連結：https://drive.google.com/drive/folders/16fJ4hJzgNerXktbx26lTl_5FQeear5CE?usp=drive_link

```markdown
Train File
├── train_info.csv: train 資料的 info
├── train_stat.csv: train 資料的每個 ID 含有幾筆資料
├── train_data: 放置訓練資料 txt 檔案之資料集
├── train_tennis.csv: train_data 處理後之 DataFrame
├── train_tennis_resized.csv: train_tennis.csv 擴增六軸加總之特徵
└── train_spectrogram_images.zip: 轉換 DataFrame 為光譜圖

Test File
├── test_info.csv: test 資料的 info
├── test_data: 放置測試資料 txt 檔案之資料集
├── test_stat.csv: test 資料的每個 ID 含有幾筆資料
├── test_tennis.csv: test_data 處理後之 DataFrame
├── test_tennis_resized.csv: test_tennis.csv 擴增六軸加總之特徵
└── test_spectrogram_images.zip: 轉換 DataFrame 為光譜圖
```

#### 下載檔案

```bash
git clone https://github.com/HungTaWang/AI_CUP_TEAM_7204.git
cd AI_CUP_TEAM_7204
```

#### 環境建置

- Microsoft Windows 11 家用版 - 10.0.26100（組建 26100）
- NVIDIA L4 GPU
- Google Colab Pro+
- Python 3.12.2
- 必要套件 (python 內建)：
    - os
    - re
    - random
    - zipfile
- 必要套件 (需額外用 requirements.txt 安裝)：
    - pandas
    - numpy
    - seaborn 
    - matplotlib
    - gdown
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

可直接執行 main.py 得出最終結果
```bash
python3 main.py
```

以下其他步驟自 Step 1 為我們的實作順序

-----
### 程式流程 (程式皆可直接執行，已內建 gdown 可自動下載檔案)

#### 1. 資料清理與特徵計算

```bash
python3 1_1_train_cleaning.py
python3 1_2_test_cleaning.py
python3 1_3_train_and_test_resize.py
```

#### 2. 資料統計與視覺化

```bash
python3 2_1_data_stat.py
python3 2_2_plot_train_waveforms_separate.py
python3 2_3_plot_train_waveforms_aggregate.py
python3 2_4_plot_test_waveforms_separate.py
python3 2_5_plot_test_waveforms_aggregate.py
```

#### 3. 光譜圖轉換

```bash
python3 3_1_train_to_spectrogram.py
python3 3_2_test_to_spectrogram.py
```

#### 4. 模型訓練

```bash
python3 4_1_cnn.py
python3 4_2_lgbm.py
```
