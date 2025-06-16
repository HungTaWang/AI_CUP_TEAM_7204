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
└── 4_2_lgbm.py: 用 LGBM 預測 gender 和 hold racket handed 並且整併 4_1_cnn.py 之預測結果，生成最終 Submission.csv

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

Final Output
└── Submission.csv: 最終輸出結果

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

預覽最終輸出之 Submission.csv

| unique_id | gender  | hold racket handed | play years_0 | play years_1 | play years_2 | level_2 | level_3 | level_4 | level_5 |
|------------|---------|-------------------|--------------|--------------|--------------|---------|---------|---------|---------|
| 1968       | 0.998777| 0.999992           | 0.228527     | 0.303520     | 0.467953     | 0.143124| 0.573750| 0.099849| 0.183276|
| 1969       | 0.999951| 0.999992           | 0.737725     | 0.224822     | 0.037453     | 0.015652| 0.303062| 0.164518| 0.516768|
| 1970       | 0.999992| 0.999992           | 0.319669     | 0.437621     | 0.242710     | 0.089735| 0.155136| 0.127473| 0.627655|
| 1971       | 0.999885| 0.999992           | 0.141455     | 0.839276     | 0.019269     | 0.004597| 0.114470| 0.034964| 0.845969|
| 1972       | 0.999439| 0.999993           | 0.136777     | 0.381795     | 0.481429     | 0.174316| 0.666457| 0.057010| 0.102218|
| ...        | ...     | ...                 | ...          | ...          | ...          | ...     | ...     | ...     | ...     |
| 3407       | 0.999888| 0.999992           | 0.071111     | 0.603095     | 0.325794     | 0.326433| 0.186050| 0.189556| 0.297961|
| 3408       | 0.999450| 0.999992           | 0.137086     | 0.686310     | 0.176604     | 0.062086| 0.205575| 0.212231| 0.520108|
| 3409       | 0.999824| 0.999992           | 0.159049     | 0.291332     | 0.549619     | 0.324539| 0.480655| 0.122159| 0.072647|
| 3410       | 0.999956| 0.999992           | 0.169168     | 0.803056     | 0.027776     | 0.004517| 0.089795| 0.050104| 0.855583|
| 3411       | 0.999045| 0.999992           | 0.097699     | 0.825039     | 0.077262     | 0.066046| 0.324590| 0.183415| 0.425948|

1430 rows × 10 columns

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
