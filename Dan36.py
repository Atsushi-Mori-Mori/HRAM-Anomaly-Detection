#　-*- coding: utf-8 -*-
import sys
import os
import re
import struct
import binascii
import numpy as np
# -------------------------------------------------------
# 1. ライブラリーのインポート, データ読込み
# ライブラリーのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import shap
# -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_fontja
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score
import pandas as pd
import plotly.graph_objects as go
from pyod.models.knn import KNN
# # -------------------------------------------------------
# # -------------------------------------------------------
# データ取得
df_train = pd.read_csv("./data/train.csv", parse_dates=['timestamp'], index_col='timestamp')
df_test = pd.read_csv("./data/test.csv", parse_dates=['timestamp'], index_col='timestamp')
# testを2025-11-01を軸にvalid,testに分割
df_valid = df_test[df_test.index < '2025-11-01']
df_test = df_test[df_test.index >= '2025-11-01']
# # # -------------------------------------------------------
# 異常ラベルのカラム名
anomaly_label = 'is_anomaly'
# display(df_train.head())
# display(df_valid.head())
# display(df_test.head())
# # -------------------------------------------------------
# # データ品質確認
# print("="*80)
# print("【データセット概要】")
# print("="*80)
# # 各データセットの基本情報
# datasets = {
#     'train': df_train,
#     'valid': df_valid,
#     'test': df_test
# }
# for name, df in datasets.items():
#     print(f"\n■ {name.upper()}データセット")
#     print(f"  - データ数: {len(df):,}行")
#     print(f"  - カラム数: {len(df.columns)}列")
#     # 異常率を計算
#     if 'is_anomaly' in df.columns:
#         anomaly_rate = df['is_anomaly'].sum() / len(df) * 100
#         print(f"  - 異常率: {anomaly_rate:.2f}%")
#     else:
#         print(f"  - 異常率: N/A (is_anomalyカラムなし)")
# print("\n" + "="*80)
# print("【カラム情報とデータ型】")
# print("="*80)
# # データ型の詳細確認
# print("\n■ TRAINデータセットのカラム情報:")
# dtype_info = pd.DataFrame({
#     'カラム名': df_train.columns,
#     'データ型': df_train.dtypes.values,
#     'ユニーク数': [df_train[col].nunique() for col in df_train.columns],
#     '欠損数': df_train.isnull().sum().values,
#     '欠損率(%)': (df_train.isnull().sum() / len(df_train) * 100).values
# })
# display(dtype_info)
# # # -------------------------------------------------------
# データ型の分類
object_cols = df_train.select_dtypes(include=['object']).columns.tolist()
# print("\n" + "="*80)
# print("【Object型カラムの詳細調査】")
# print("="*80)
if len(object_cols) > 0:
    # ユニーク数が大きい順にソートして上位3つを選択
    object_col_info = [(col, df_train[col].nunique()) for col in object_cols]
    object_col_info_sorted = sorted(object_col_info, key=lambda x: x[1], reverse=True)
    top_2_cols = object_col_info_sorted[:2]    
    # print(f"ユニーク数が多い上位2つを調査:")    
    for col, unique_count in top_2_cols:
        # print(f"\n■ カラム: '{col}' (ユニーク数: {unique_count})")
        
        # 数値変換不可の値（ゴミデータ）を抽出
        numeric_conversion = pd.to_numeric(df_train[col], errors='coerce')
        non_numeric_mask = numeric_conversion.isna() & df_train[col].notna()
        non_numeric_values = df_train[col][non_numeric_mask].unique()        
        conversion_success_rate = (1 - numeric_conversion.isna().sum() / len(df_train)) * 100
        # print(f"\n  【数値変換】")
        # print(f"    - 変換成功率: {conversion_success_rate:.2f}%")        
        # if len(non_numeric_values) > 0:
        #     print(f"    - 数値でない値（ゴミデータ）: {len(non_numeric_values)}種類")
        #     print(f"    - サンプル: {non_numeric_values[:10].tolist()}")
        # else:
        #     print(f"    - 全て数値として変換可能")
else:
    print("\nObject型のカラムはありません。")
# print("\n✓データ品質チェック完了")
# # # -------------------------------------------------------
# --- 0) 数値化（念のため） ---
df_train['ftr_bdc_ratio'] = pd.to_numeric(df_train['ftr_bdc_ratio'], errors='coerce')
df_train['ftr_asymmetry'] = pd.to_numeric(df_train['ftr_asymmetry'], errors='coerce')
df_train['ftr_vib_high_band'] = pd.to_numeric(df_train['ftr_vib_high_band'], errors='coerce')
df_train['peak_angle_deg'] = pd.to_numeric(df_train['peak_angle_deg'], errors='coerce')
df_train['peak_tonnage_kN'] = pd.to_numeric(df_train['peak_tonnage_kN'], errors='coerce')
df_train['vibration_rms_g'] = pd.to_numeric(df_train['vibration_rms_g'], errors='coerce')
df_train['lube_pressure_bar'] = pd.to_numeric(df_train['lube_pressure_bar'], errors='coerce')
df_train['is_anomaly'] = pd.to_numeric(df_train['is_anomaly'], errors='coerce')
df_train['ftr_halfwidth80_deg'] = pd.to_numeric(df_train['ftr_halfwidth80_deg'], errors='coerce')
df_train['spm'] = pd.to_numeric(df_train['spm'], errors='coerce')
df_train['motor_current_A'] = pd.to_numeric(df_train['motor_current_A'], errors='coerce')
# ==========================
# #  特徴量の追加
# ==========================
#3. 電流効率指標
df_train['vib_per_ton'] = df_train['vibration_rms_g'] / df_train['peak_tonnage_kN'].replace(0, np.nan)
#4. BDC集中度
df_train['bdc_sharpness'] = df_train['ftr_bdc_ratio'] / df_train['ftr_halfwidth80_deg'].replace(0, np.nan)
# 7. 速度補正荷重
df_train['tonnage_norm_spm'] = df_train['peak_tonnage_kN'] / df_train['spm'].replace(0, np.nan)
# 5. 速度補正電流
df_train['current_norm_spm'] = df_train['motor_current_A'] / df_train['spm'].replace(0, np.nan)
# ==========================
# 0) zスコア補正 df_train
# ==========================
colsZ = [
    'ftr_bdc_ratio',
    'ftr_asymmetry',
    'ftr_vib_high_band',
    'peak_angle_deg',
    'peak_tonnage_kN',
    'vibration_rms_g',
    'lube_pressure_bar',
    'vib_per_ton',
    'bdc_sharpness',
    'tonnage_norm_spm',
    'current_norm_spm'
]
for col in colsZ:
    df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
df_train['is_anomaly'] = pd.to_numeric(df_train['is_anomaly'], errors='coerce')
# ==========================
# 1) 正常データのみ抽出
# ==========================
df_normal = df_train[df_train['is_anomaly'] == 0]
# ==========================
# 2) 各列ごとにZ補正
# ==========================
for col in colsZ:
    # 製品種別ごとの正常平均・標準偏差を算出
    stats = (
        df_normal
        .groupby('product_type')[col]
        .agg(['mean', 'std'])
        .rename(columns={
            'mean': f'{col}_mean0',
            'std': f'{col}_std0'
        })
        .reset_index()
    )
    # 元データへマージ
    df_train = df_train.merge(stats, on='product_type', how='left')    
    # 0除算回避
    denom = df_train[f'{col}_std0'].replace(0, np.nan)   
    # Zスコア算出（正常・異常関係なく）
    df_train[f'z_{col}'] = (
        (df_train[col] - df_train[f'{col}_mean0']) / denom
    )
# ==========================
# 確認
# ==========================
# print(df_train[
#     ['product_type', 'is_anomaly'] +
#     [f'z_{c}' for c in colsZ]
# ].head())
# # # -------------------------------------------------------
# --- 0) 数値化（念のため） ---
df_valid['ftr_bdc_ratio'] = pd.to_numeric(df_valid['ftr_bdc_ratio'], errors='coerce')
df_valid['ftr_asymmetry'] = pd.to_numeric(df_valid['ftr_asymmetry'], errors='coerce')
df_valid['ftr_vib_high_band'] = pd.to_numeric(df_valid['ftr_vib_high_band'], errors='coerce')
df_valid['peak_angle_deg'] = pd.to_numeric(df_valid['peak_angle_deg'], errors='coerce')
df_valid['peak_tonnage_kN'] = pd.to_numeric(df_valid['peak_tonnage_kN'], errors='coerce')
df_valid['vibration_rms_g'] = pd.to_numeric(df_valid['vibration_rms_g'], errors='coerce')
df_valid['lube_pressure_bar'] = pd.to_numeric(df_valid['lube_pressure_bar'], errors='coerce')
df_valid['is_anomaly'] = pd.to_numeric(df_valid['is_anomaly'], errors='coerce')
df_valid['ftr_halfwidth80_deg'] = pd.to_numeric(df_valid['ftr_halfwidth80_deg'], errors='coerce')
df_valid['spm'] = pd.to_numeric(df_valid['spm'], errors='coerce')
df_valid['motor_current_A'] = pd.to_numeric(df_valid['motor_current_A'], errors='coerce')
# ==========================
# #  特徴量の追加
# ==========================
#3. 電流効率指標
df_valid['vib_per_ton'] = df_valid['vibration_rms_g'] / df_valid['peak_tonnage_kN'].replace(0, np.nan)
#4. BDC集中度
df_valid['bdc_sharpness'] = df_valid['ftr_bdc_ratio'] / df_valid['ftr_halfwidth80_deg'].replace(0, np.nan)
# 7. 速度補正荷重
df_valid['tonnage_norm_spm'] = df_valid['peak_tonnage_kN'] / df_valid['spm'].replace(0, np.nan)
# 5. 速度補正電流
df_valid['current_norm_spm'] = df_valid['motor_current_A'] / df_valid['spm'].replace(0, np.nan)
# ==========================
# 0) zスコア補正 df_valid
# ==========================
for col in colsZ:
    df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce')
df_valid['is_anomaly'] = pd.to_numeric(df_valid['is_anomaly'], errors='coerce')
# ==========================
# 1) 正常データのみ抽出
# ==========================
df_normal = df_valid[df_valid['is_anomaly'] == 0]
# ==========================
# 2) 各列ごとにZ補正
# ==========================
for col in colsZ:
    # 製品種別ごとの正常平均・標準偏差を算出
    stats = (
        df_normal
        .groupby('product_type')[col]
        .agg(['mean', 'std'])
        .rename(columns={
            'mean': f'{col}_mean0',
            'std': f'{col}_std0'
        })
        .reset_index()
    )
    # 元データへマージ
    df_valid = df_valid.merge(stats, on='product_type', how='left')    
    # 0除算回避
    denom = df_valid[f'{col}_std0'].replace(0, np.nan)   
    # Zスコア算出（正常・異常関係なく）
    df_valid[f'z_{col}'] = (
        (df_valid[col] - df_valid[f'{col}_mean0']) / denom
    )
# # # -------------------------------------------------------
# --- 0) 数値化（念のため） ---
df_test['ftr_bdc_ratio'] = pd.to_numeric(df_test['ftr_bdc_ratio'], errors='coerce')
df_test['ftr_asymmetry'] = pd.to_numeric(df_test['ftr_asymmetry'], errors='coerce')
df_test['ftr_vib_high_band'] = pd.to_numeric(df_test['ftr_vib_high_band'], errors='coerce')
df_test['peak_angle_deg'] = pd.to_numeric(df_test['peak_angle_deg'], errors='coerce')
df_test['peak_tonnage_kN'] = pd.to_numeric(df_test['peak_tonnage_kN'], errors='coerce')
df_test['vibration_rms_g'] = pd.to_numeric(df_test['vibration_rms_g'], errors='coerce')
df_test['lube_pressure_bar'] = pd.to_numeric(df_test['lube_pressure_bar'], errors='coerce')
df_test['is_anomaly'] = pd.to_numeric(df_test['is_anomaly'], errors='coerce')
df_test['ftr_halfwidth80_deg'] = pd.to_numeric(df_test['ftr_halfwidth80_deg'], errors='coerce')
df_test['spm'] = pd.to_numeric(df_test['spm'], errors='coerce')
df_test['motor_current_A'] = pd.to_numeric(df_test['motor_current_A'], errors='coerce')
# ==========================
# #  特徴量の追加
# ==========================
#3. 電流効率指標
df_test['vib_per_ton'] = df_test['vibration_rms_g'] / df_test['peak_tonnage_kN'].replace(0, np.nan)
#4. BDC集中度
df_test['bdc_sharpness'] = df_test['ftr_bdc_ratio'] / df_test['ftr_halfwidth80_deg'].replace(0, np.nan)
# 7. 速度補正荷重
df_test['tonnage_norm_spm'] = df_test['peak_tonnage_kN'] / df_test['spm'].replace(0, np.nan)
# 5. 速度補正電流
df_test['current_norm_spm'] = df_test['motor_current_A'] / df_test['spm'].replace(0, np.nan)
# ==========================
# 0) zスコア補正 df_test
# ==========================
for col in colsZ:
    df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
df_test['is_anomaly'] = pd.to_numeric(df_test['is_anomaly'], errors='coerce')
# ==========================
# 1) 正常データのみ抽出
# ==========================
df_normal = df_test[df_test['is_anomaly'] == 0]
# ==========================
# 2) 各列ごとにZ補正
# ==========================
for col in colsZ:
    # 製品種別ごとの正常平均・標準偏差を算出
    stats = (
        df_normal
        .groupby('product_type')[col]
        .agg(['mean', 'std'])
        .rename(columns={
            'mean': f'{col}_mean0',
            'std': f'{col}_std0'
        })
        .reset_index()
    )
    # 元データへマージ
    df_test = df_test.merge(stats, on='product_type', how='left')    
    # 0除算回避
    denom = df_test[f'{col}_std0'].replace(0, np.nan)   
    # Zスコア算出（正常・異常関係なく）
    df_test[f'z_{col}'] = (
        (df_test[col] - df_test[f'{col}_mean0']) / denom
    )
# # # -------------------------------------------------------
# # -----------------------------------------------
# # IQR範囲外の異常値補正 df_train, df_vald, df_test
# # -----------------------------------------------
colIQR = [
    'z_ftr_bdc_ratio',
    'z_ftr_asymmetry',
    'z_ftr_vib_high_band',
    'z_peak_angle_deg',
    'z_peak_tonnage_kN',
    'z_vibration_rms_g',
    'z_lube_pressure_bar',
    'z_vib_per_ton',
    'z_bdc_sharpness',
    'z_tonnage_norm_spm',
    'z_current_norm_spm'
]
def iqr_correct_only_normal(df, cols, label_col='is_anomaly', k=1.5):
    df = df.copy()
    # 正常データのみ抽出
    df_normal = df[df[label_col] == 0]
    for col in cols:
        Q1 = df_normal[col].quantile(0.25)
        Q3 = df_normal[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        # print(f"{col}: lower={lower:.4f}, upper={upper:.4f}")
        # 正常データのみに補正を適用
        mask_normal = df[label_col] == 0
        df.loc[mask_normal, col] = df.loc[mask_normal, col].clip(lower, upper)
    return df
# 実行
df_train_iqr = iqr_correct_only_normal(df_train, colIQR)
df_train0 = df_train.copy()
df_train = df_train_iqr.copy()
# # # -------------------------------------------------------
df_valid_iqr = iqr_correct_only_normal(df_valid, colIQR)
df_valid0 = df_valid.copy()
df_valid = df_valid_iqr.copy()
# # # -------------------------------------------------------
df_test_iqr = iqr_correct_only_normal(df_test, colIQR)
df_test0 = df_test.copy()
df_test = df_test_iqr.copy()
# # # -------------------------------------------------------
# # -----------------------------
# # 移動平均
# # -----------------------------
df_train['r_ftr_bdc_ratio'] = df_train['z_ftr_bdc_ratio'].rolling(window=5).mean()
df_train['r_ftr_asymmetry'] = df_train['z_ftr_asymmetry'].rolling(window=5).mean()
df_train['r_ftr_vib_high_band'] = df_train['z_ftr_vib_high_band'].rolling(window=5).mean()
df_train['r_peak_angle_deg'] = df_train['z_peak_angle_deg'].rolling(window=5).mean()
df_train['r_peak_tonnage_kN'] = df_train['z_peak_tonnage_kN'].rolling(window=5).mean()
df_train['r_vibration_rms_g'] = df_train['z_vibration_rms_g'].rolling(window=5).mean()
df_train['r_lube_pressure_bar'] = df_train['z_lube_pressure_bar'].rolling(window=5).mean()
df_train['r_vib_per_ton'] = df_train['z_vib_per_ton'].rolling(window=5).mean()
df_train['r_bdc_sharpness'] = df_train['z_bdc_sharpness'].rolling(window=5).mean()
df_train['r_tonnage_norm_spm'] = df_train['z_tonnage_norm_spm'].rolling(window=5).mean()
df_train['r_current_norm_spm'] = df_train['z_current_norm_spm'].rolling(window=5).mean()
# # -----------------------------
df_valid['r_ftr_bdc_ratio'] = df_valid['z_ftr_bdc_ratio'].rolling(window=5).mean()
df_valid['r_ftr_asymmetry'] = df_valid['z_ftr_asymmetry'].rolling(window=5).mean()
df_valid['r_ftr_vib_high_band'] = df_valid['z_ftr_vib_high_band'].rolling(window=5).mean()
df_valid['r_peak_angle_deg'] = df_valid['z_peak_angle_deg'].rolling(window=5).mean()
df_valid['r_peak_tonnage_kN'] = df_valid['z_peak_tonnage_kN'].rolling(window=5).mean()
df_valid['r_vibration_rms_g'] = df_valid['z_vibration_rms_g'].rolling(window=5).mean()
df_valid['r_lube_pressure_bar'] = df_valid['z_lube_pressure_bar'].rolling(window=5).mean()
df_valid['r_vib_per_ton'] = df_valid['z_vib_per_ton'].rolling(window=5).mean()
df_valid['r_bdc_sharpness'] = df_valid['z_bdc_sharpness'].rolling(window=5).mean()
df_valid['r_tonnage_norm_spm'] = df_valid['z_tonnage_norm_spm'].rolling(window=5).mean()
df_valid['r_current_norm_spm'] = df_valid['z_current_norm_spm'].rolling(window=5).mean()
# # -----------------------------
df_test['r_ftr_bdc_ratio'] = df_test['z_ftr_bdc_ratio'].rolling(window=5).mean()
df_test['r_ftr_asymmetry'] = df_test['z_ftr_asymmetry'].rolling(window=5).mean()
df_test['r_ftr_vib_high_band'] = df_test['z_ftr_vib_high_band'].rolling(window=5).mean()
df_test['r_peak_angle_deg'] = df_test['z_peak_angle_deg'].rolling(window=5).mean()
df_test['r_peak_tonnage_kN'] = df_test['z_peak_tonnage_kN'].rolling(window=5).mean()
df_test['r_vibration_rms_g'] = df_test['z_vibration_rms_g'].rolling(window=5).mean()
df_test['r_lube_pressure_bar'] = df_test['z_lube_pressure_bar'].rolling(window=5).mean()
df_test['r_vib_per_ton'] = df_test['z_vib_per_ton'].rolling(window=5).mean()
df_test['r_bdc_sharpness'] = df_test['z_bdc_sharpness'].rolling(window=5).mean()
df_test['r_tonnage_norm_spm'] = df_test['z_tonnage_norm_spm'].rolling(window=5).mean()
df_test['r_current_norm_spm'] = df_test['z_current_norm_spm'].rolling(window=5).mean()
# # # -------------------------------------------------------
# object型でユニーク値が10以上のカラムを抽出（数値型に変換対象）
# # # -------------------------------------------------------
cols_to_convert_numeric = [col for col, unique_count in object_col_info_sorted if unique_count >= 10]
# # ------------------------
# 異常値（'-'を含む行）の削除
df_train_clean = df_train[~df_train['ftr_bdc_ratio'].astype(str).str.contains('-')].copy()
df_valid_clean = df_valid[~df_valid['ftr_bdc_ratio'].astype(str).str.contains('-')].copy()
df_test_clean = df_test[~df_test['ftr_bdc_ratio'].astype(str).str.contains('-')].copy()
# # ------------------------
# df_train_clean = df_train[~df_train['z_ftr_bdc_ratio'].astype(str).str.contains('-')].copy()
# df_valid_clean = df_valid[~df_valid['z_ftr_bdc_ratio'].astype(str).str.contains('-')].copy()
# df_test_clean = df_test[~df_test['z_ftr_bdc_ratio'].astype(str).str.contains('-')].copy()
# # ------------------------
# object型カラムを数値型に変換
for col in cols_to_convert_numeric:
    df_train_clean[col] = pd.to_numeric(df_train_clean[col], errors='coerce')
    df_valid_clean[col] = pd.to_numeric(df_valid_clean[col], errors='coerce')
    df_test_clean[col] = pd.to_numeric(df_test_clean[col], errors='coerce')
# 欠損値の前方補完
df_train_clean = df_train_clean.ffill()
df_valid_clean = df_valid_clean.ffill()
df_test_clean = df_test_clean.ffill()
# # # -------------------------------------------------------
# データ型の詳細確認（データクリーニング後）
# print("\n■ TRAINデータセットのカラム情報（データクリーニング後）:")
# dtype_info = pd.DataFrame({
#     'カラム名': df_train_clean.columns,
#     'データ型': df_train_clean.dtypes.values,
#     'ユニーク数': [df_train_clean[col].nunique() for col in df_train_clean.columns],
#     '欠損数': df_train_clean.isnull().sum().values,
#     '欠損率(%)': (df_train_clean.isnull().sum() / len(df_train_clean) * 100).values
# })
# display(dtype_info)
# # # -------------------------------------------------------
# 5_基本統計量と時系列可視化
# print("\n" + "="*80)
# print("【基本統計量】")
# print("="*80)

# # 数値型カラムの基本統計量を表示
# print("\n■ 訓練データの基本統計量:")
# display(df_train_clean.describe().T)

# """
# ### **Plotly可視化に関する重要な注意事項**

# **ノートブックの容量増加について:**
# - Plotlyのインタラクティブなグラフは、**ノートブックファイルのサイズを大幅に増加**させます
# - 大量のデータポイントや複数のグラフを保存すると、**数MB〜数十MBに膨れ上がる**可能性があります
# - ファイルサイズが大きくなると、**ノートブックの読み込みや保存が遅くなり、動作が重く**なります

# **対策:**
# - グラフ実行後は**出力をクリア**することを推奨
# - データポイント数を減らす（サンプリングやダウンサンプリング）
# - 必要に応じて静的な画像として保存（`fig.write_image()`）

# **Google Colabについて:**
# - **最近のColabでは設定不要で動作します**（そのまま使えます）
# - 表示に問題がある場合のみ、以下を試してください：

# ```python
# # グラフが表示されない場合のみ実行（通常は不要）
# import plotly.io as pio
# pio.renderers.default = 'colab'
# ```
# """
# # # -------------------------------------------------------
# 可視化する特徴量を指定（例）
# features_to_plot = ['ftr_bdc_ratio', 'ftr_asymmetry', 'ftr_vib_high_band']
# # 異常区間の抽出（整数インデックスを使用）
# anomaly_periods = []
# in_anomaly = False
# start_idx = None
# for i in range(len(df_train_clean)):
#     if df_train_clean.iloc[i][anomaly_label] and not in_anomaly:
#         start_idx = i
#         in_anomaly = True
#     elif not df_train_clean.iloc[i][anomaly_label] and in_anomaly:
#         anomaly_periods.append((start_idx, i - 1))
#         in_anomaly = False
# # 最後が異常で終わった場合
# if in_anomaly:
#     anomaly_periods.append((start_idx, len(df_train_clean) - 1))
# # # -------------------------------------------------------
# # 各特徴量をプロット
# for feature in features_to_plot:
#     if feature not in df_train_clean.columns:
#         print(f"特徴量 '{feature}' がデータに存在しません。スキップします。")
#         continue
    
#     fig = go.Figure()
    
#     # 異常区間の背景色を追加（整数インデックスをstroke_idに変換）
#     for start, end in anomaly_periods:
#         start_stroke_id = df_train_clean.iloc[start]['stroke_id']
#         end_stroke_id = df_train_clean.iloc[end]['stroke_id']
#         fig.add_vrect(
#             x0=start_stroke_id, x1=end_stroke_id,
#             fillcolor="red", opacity=0.2,
#             layer="below", line_width=0,
#         )
    
#     # データを線と点でプロット
#     fig.add_trace(go.Scatter(
#         x=df_train_clean['stroke_id'],
#         y=df_train_clean[feature],
#         mode='lines',
#         line=dict(width=1),
#         name=feature
#     ))
    
#     fig.update_layout(
#         title=f'訓練データ: {feature} の時系列推移',
#         xaxis_title='ストロークID',
#         yaxis_title=feature,
#         height=400,
#         showlegend=True
#     )
    
#     fig.show()
# # # -------------------------------------------------------
# # 異常検知用の特徴量を選定
selected_features = [
    'r_ftr_bdc_ratio',
    'r_ftr_asymmetry',
    'r_ftr_vib_high_band',
    'r_peak_angle_deg',
    'r_lube_pressure_bar',
    'r_vibration_rms_g',
    'r_peak_tonnage_kN',
    'r_vib_per_ton',
    'r_bdc_sharpness',
    'r_tonnage_norm_spm',
    'r_current_norm_spm'
]
anomaly_label = 'is_anomaly'

# 訓練/検証/テストデータを作成
df_train_selected = df_train_clean[selected_features + [anomaly_label]]
df_valid_selected = df_valid_clean[selected_features + [anomaly_label]]
df_test_selected = df_test_clean[selected_features + [anomaly_label]]

df_train_selected.head()
# # # -------------------------------------------------------
# 7_スライド窓変換（統計量版）
def sliding_window_transform_stats(
    df: pd.DataFrame,
    window_size: int,
    anomaly_label: str,
    add_raw_values: bool = False,   # Trueにすると従来通り t0..t29 も残せます
) -> pd.DataFrame:
    """
    時系列データにスライディングウィンドウ変換を適用し、
    各ウィンドウを1行に変換して統計量特徴を追加します。

    生成する統計量（各特徴量ごと）:
      - mean（平均）
      - var（分散: ddof=0）
      - max（最大）
      - min（最小）
      - delta_mean（この窓の平均 - 1行前(=前窓)の平均）
        ※最初の行は前がないため 0.0（必要ならnp.nanに変更可）

    異常ラベル:
      - ウィンドウ内に1つでも異常があれば is_window_anomaly=True

    引数:
        df: 入力データフレーム（時系列データ）
        window_size: スライディングウィンドウ幅
        anomaly_label: 異常ラベル列名
        add_raw_values: Trueなら "{col}_t{t}" の生値も出力に含める

    戻り値:
        各行が1つのウィンドウを表すデータフレーム
    """
    if anomaly_label not in df.columns:
        raise ValueError(f"anomaly_label '{anomaly_label}' がdfに存在しません。")

    feature_cols = [c for c in df.columns if c != anomaly_label]
    n = len(df)
    if n < window_size:
        raise ValueError(f"dfの行数({n})がwindow_size({window_size})より小さいため窓を作れません。")

    rows = []
    prev_mean = {col: None for col in feature_cols}  # 前窓の平均（列ごと）

    for i in range(n - window_size + 1):
        w = df.iloc[i:i + window_size]

        row = {}

        # 生値を残したい場合（従来互換）
        if add_raw_values:
            for col in feature_cols:
                vals = w[col].to_numpy()
                for t, v in enumerate(vals):
                    row[f"{col}_t{t}"] = v

        # 統計量を追加
        for col in feature_cols:
            vals = w[col].to_numpy(dtype=float)  # 数値化（object混在なら事前にto_numeric推奨）
            m = float(np.mean(vals))
            v = float(np.var(vals, ddof=0))
            mx = float(np.max(vals))
            mn = float(np.min(vals))

            row[f"{col}_mean"] = m
            row[f"{col}_var"]  = v
            row[f"{col}_max"]  = mx
            row[f"{col}_min"]  = mn

            # 前窓平均との差（最初の窓は前が無いので0.0。np.nanにしたければここを変更）
            if prev_mean[col] is None:
                row[f"{col}_delta_mean"] = 0.0
            else:
                row[f"{col}_delta_mean"] = m - prev_mean[col]

            prev_mean[col] = m

        # 窓内異常フラグ
        row["is_window_anomaly"] = bool(w[anomaly_label].any())
        rows.append(row)

    return pd.DataFrame(rows)

# スライド窓の窓幅
window_size = 30

# # 訓練/検証/テストデータのスライド窓変換
df_train_win = sliding_window_transform_stats(df_train_selected, window_size, anomaly_label, add_raw_values=False)
df_valid_win = sliding_window_transform_stats(df_valid_selected, window_size, anomaly_label, add_raw_values=False)
df_test_win  = sliding_window_transform_stats(df_test_selected,  window_size, anomaly_label, add_raw_values=False)

# df_train_win.head()
# # # # -------------------------------------------------------
# # 正解ラベル(is_window_anomaly)の削除
train = df_train_win.drop(columns=['is_window_anomaly'])
valid = df_valid_win.drop(columns=['is_window_anomaly'])
test = df_test_win.drop(columns=['is_window_anomaly'])
# 正解ラベルの抽出
valid_label = df_valid_win['is_window_anomaly'].astype(int)
test_label = df_test_win['is_window_anomaly'].astype(int)
df_train1 = pd.concat([df_train_win, df_valid_win], axis=0).reset_index(drop=True)

# train.head()
# # # # -------------------------------------------------------
# # (B)LightGBM
# # # # -------------------------------------------------------
# 特徴量とラベルに分割
X = df_train1.drop(columns=['is_window_anomaly'])
y = df_train1['is_window_anomaly'].astype(int)
# # -------------------------------------------------------
# LightGBMのハイパーパラメータの設定
params = {
    'objective': 'binary', # 多クラス分類
    'metric': 'auc', # 損失関数にmulti_loglossを使用
    # 'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    'learning_rate': 0.01,
    # 'scale_pos_weight': 10,  # label=1 に重み付けしたければいじる
}
# # -------------------------------------------------------
# # # クロスバリデーション
from sklearn.model_selection import KFold, StratifiedKFold
result_rocauc, result_prauc, result_f1 = [], [], []
train_idxA, val_idxA = [], []
kf = KFold(n_splits=3,shuffle=True,random_state=71)
# データの分割
for train_idx, val_idx in kf.split(X, y):
    train_idx2, val_idx2 = train_idx.tolist(), val_idx.tolist()
    X_train = X.iloc[train_idx2]
    y_train = y[train_idx2]
    X_test = X.iloc[val_idx2]
    y_test = y[val_idx2]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # LightGBM用のデータセットに変換
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    # LightGBMモデルの学習
    model = lgb.train(params, 
                  train_data, 
                  num_boost_round=1000,
                  valid_sets=[train_data, test_data])
                  # valid_sets=[train_data, test_data], 
                  # early_stopping_rounds=10)
    # テストデータでの予測
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred >= 0.5).astype(int)
    # 精度の評価
    # accuracy = accuracy_score(y_test, y_pred_class)
    roc_auc = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_pred_class)
    train_idxA.append(train_idx2)
    val_idxA.append(val_idx2)
    result_rocauc.append(roc_auc)
    result_prauc.append(pr_auc)
    result_f1.append(f1)
print('Average score for roc_auc: {}'.format(np.mean(result_rocauc)))
print('Average score for pr_auc: {}'.format(np.mean(result_prauc)))
print('Average score for f1: {}'.format(np.mean(result_f1)))
# # # # -------------------------
# 検証データでF1最大などのしきい値を探す
# from sklearn.metrics import precision_recall_curve
# y_pred = model.predict(X_test)
# p, r, th = precision_recall_curve(y_test, y_pred)
# f1 = 2*p*r/(p+r+1e-12)
# best_th = th[f1[1:].argmax()]   # thは1つ短いので注意
# y_pred_class = (y_pred >= best_th).astype(int)
# # # # -------------------------------------------------------
# # test推定
y_pred2 = model.predict(test)
# y_pred2_class = np.argmax(y_pred2, axis=1) # 予測結果のクラスの値を調整
y_pred2_class = (y_pred2 >= 0.5).astype(int)
# accuracy2 = accuracy_score(test_label, y_pred2_class)
# print('Accuracy2:', accuracy2)
roc_auc = roc_auc_score(test_label, y_pred2)
precision, recall, _ = precision_recall_curve(test_label, y_pred2)
pr_auc = auc(recall, precision)  # PR-AUC: 不均衡データに有効
f1 = f1_score(test_label, y_pred2_class)
print('Test ROC_AUC:', roc_auc)
print('Test PR_AUC:', pr_auc)
print('Test f1:', f1)
# # -------------------------------------------------------
#A(2) shap
explainer = shap.TreeExplainer(model=model)
# print(explainer.expected_value)
X_test_shap = test.copy().reset_index(drop=True)
shap_values = explainer.shap_values(X=X_test_shap)
shap.summary_plot(shap_values, X_test_shap) #左側の図
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # # # -------------------------------------------------------
# # (A)KNN
# # # # -------------------------------------------------------
# # # # kNNモデルの初期化（デフォルトパラメータ：n_neighbors=5, contamination=0.1）
# model_knn = KNN()
# # 訓練データでモデルを学習
# model_knn.fit(train)
# # 検証データに対して異常スコアと予測を取得
# # score_knn: 異常度スコア（値が大きいほど異常）
# # result_knn: 0=正常, 1=異常 のラベル
# score_knn = model_knn.decision_function(valid)
# result_knn = model_knn.predict(valid)
# # # # -------------------------------------------------------
# # # kNNモデルの評価指標を計算
# # # ROC-AUC: クラス分類性能の総合指標
# roc_auc = roc_auc_score(valid_label, score_knn)
# # Precision-Recall曲線からPR-AUCを計算
# precision, recall, _ = precision_recall_curve(valid_label, score_knn)
# pr_auc = auc(recall, precision)  # PR-AUC: 不均衡データに有効
# # F1スコア: 適合率と再現率の調和平均（バランスの良い指標）
# f1 = f1_score(valid_label, result_knn)
# # 評価結果を辞書形式で格納
# results = [{'Model': 'kNN', 'ROC_AUC': roc_auc, 'PR_AUC': pr_auc, 'F1': f1}]
# # 評価結果を表形式で表示
# df_results = pd.DataFrame(results)
# print("\n" + "="*60)
# print("  モデル評価指標（検証）")
# print("="*60)
# print(df_results.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
# print("="*60)
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------

