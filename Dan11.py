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

# スタイル設定の後に日本語フォントを設定（順序が重要）
matplotlib_fontja.japanize()

# 有効性検証の再現性のための乱数シード値
SEED = 42
# # -------------------------------------------------------
# ### 基本データ、加工データ、実績データの読込み
# # 基本データ
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
df_prod = pd.read_csv('./data/product_d_normal_stats.csv')
# test_merged = pd.merge(actual_test, processing_test)
# # -------------------------------------------------------
df_train0 = df_train.copy()
dstamp = []
for k in range(len(df_train0)):
    dstamp.append(df_train0['timestamp'][k].split()[0])
df_train0['datestamp'] = dstamp
# print(df_train0.groupby("datestamp").size().reset_index(name="count").sort_values("count", ascending=False))
# # -------------------------------------------------------
sstamp = []
for k in range(len(df_train0)):
    a0 = int(df_train0['timestamp'][k].split()[1].split(':')[0])
    a1 = int(df_train0['timestamp'][k].split()[1].split(':')[1])
    sstamp.append(a0*60+a1)
df_train0['secstamp'] = sstamp
# # -------------------------------------------------------
# df_train0.to_csv("./data/train_df.csv", index=False) # index=Falseで行番号を除外
# # -------------------------------------------------------
# 欠損値確認
# print(df_train0.isna().sum())
# print(df_train0.groupby("datestamp")["is_anomaly"].sum())
# print(df_train0[df_train0["run_type"] == "trial"].groupby("datestamp").size())
# print(df_train0[df_train0["product_type"] == "D"].groupby("datestamp").size())
# df_train0["spm"] = pd.to_numeric(df_train0["spm"], errors="coerce")
# print(df_train0.groupby("datestamp")["spm"].agg(["mean", "var"]))
# # -------------------------------------------------------
cols = [
    "spm",
    "stroke_time_ms",
    "peak_tonnage_kN",
    "peak_angle_deg",
    "energy_proxy_kNdeg",
    "motor_current_A",
    "vibration_rms_g",
    "air_pressure_bar",
    "lube_pressure_bar",
    "die_temp_C",
    "motor_temp_C",
    "ftr_asymmetry",
    "ftr_halfwidth80_deg",
    "ftr_bdc_ratio",
    "ftr_energy_irreg",
    "ftr_vib_high_band",
    "ftr_vib_sideband_idx",
    "ftr_c2c_peak_delta_abs",
    "ftr_c2c_peak_angle_delta_abs",
    "spm_cv_proxy",
]
# # 念のため数値化（"－" → NaN）
# df_train0[cols] = df_train0[cols].apply(pd.to_numeric, errors="coerce")
# # ---------------------------
# # ① 平均・分散
# # ---------------------------
# df_day_stats = (
#     df_train0.groupby("datestamp")[cols]
#     .agg(["mean", "var"])
# )
# df_day_stats.columns = [
#     f"{col}_{stat}" for col, stat in df_day_stats.columns
# ]
# df_day_stats = df_day_stats.reset_index()
# # ---------------------------
# # ② count（有効データ数）
# # ---------------------------
# df_day_count = (
#     df_train0.groupby("datestamp")[cols]
#     .count()
# )
# df_day_count.columns = [
#     f"{col}_count" for col in df_day_count.columns
# ]
# df_day_count = df_day_count.reset_index()
# # 確認
# df_day_stats.head()
# df_day_count.head()
# df_day_stats.to_csv("./data/train_stats_df.csv", index=False) # index=Falseで行番号を除外
# df_day_count.to_csv("./data/train_count_df.csv", index=False) # index=Falseで行番号を除外
# # -------------------------------------------------------
# # # 相関算出
# 数値化（"－"などはNaN）
# # 1) '-' を NaN に置換（必要なら '－' や空文字も追加）
# df_train0[cols] = df_train0[cols].replace(["-", "－", ""], np.nan)
# # 2) 数値化（変換できないものはNaN）
# df_train0[cols] = df_train0[cols].apply(pd.to_numeric, errors="coerce")
# # is_anomaly も数値化（True/FalseでもOK）
# df_train0["is_anomaly"] = df_train0["is_anomaly"].replace(["-", "－", ""], np.nan)
# df_train0["is_anomaly"] = pd.to_numeric(df_train0["is_anomaly"], errors="coerce")
# # 3) 相関（各特徴量 vs is_anomaly）
# rows = []
# y = df_train0["is_anomaly"]
# for c in cols:
#     x = df_train0[c]
#     m = x.notna() & y.notna()
#     n_valid = int(m.sum())
#     n_missing = int((~m).sum())
#     # is_anomaly が 0/1 の両方を含まない場合は相関は定義不可
#     if n_valid < 3 or y[m].nunique() < 2:
#         corr = np.nan
#     else:
#         corr = x[m].corr(y[m])  # Pearson（=点双列相関）
#     rows.append({
#         "feature": c,
#         "corr_with_is_anomaly": corr,
#         "n_valid": n_valid,
#         "n_missing": n_missing,
#     })
# df_corr = (
#     pd.DataFrame(rows)
#       .assign(abs_corr=lambda d: d["corr_with_is_anomaly"].abs())
#       .sort_values("abs_corr", ascending=False)
#       .drop(columns="abs_corr")
#       .reset_index(drop=True)
# )
# df_corr
# df_corr.to_csv("./data/train_corr.csv", index=False) # index=Falseで行番号を除外
# # -------------------------------------------------------
# # ----- 数値化（安全処理） -----
# df_train0[cols] = (
#     df_train0[cols]
#     .replace(["-", "－", ""], np.nan)
#     .apply(pd.to_numeric, errors="coerce")
# )
# df_train0["is_anomaly"] = pd.to_numeric(df_train0["is_anomaly"], errors="coerce")
# rows = []
# for c in cols:
#     sub = df_train0[[c, "is_anomaly"]].dropna()
#     if sub["is_anomaly"].nunique() < 2:
#         continue
#     normal = sub[sub["is_anomaly"] == 0][c]
#     anomaly = sub[sub["is_anomaly"] == 1][c]
#     if len(normal) < 2 or len(anomaly) < 2:
#         continue
#     mean_normal = normal.mean()
#     mean_anomaly = anomaly.mean()
#     mean_diff = mean_anomaly - mean_normal
#     # Cohen's d
#     pooled_std = np.sqrt(
#         ((normal.std()**2 + anomaly.std()**2) / 2)
#     )
#     effect_size = mean_diff / pooled_std if pooled_std != 0 else np.nan
#     # AUC（単特徴量）
#     try:
#         auc = roc_auc_score(sub["is_anomaly"], sub[c])
#     except:
#         auc = np.nan
#     rows.append({
#         "feature": c,
#         "mean_normal": mean_normal,
#         "mean_anomaly": mean_anomaly,
#         "mean_diff": mean_diff,
#         "effect_size_d": effect_size,
#         "AUC": auc,
#         "n_valid": len(sub)
#     })
# df_effect = (
#     pd.DataFrame(rows)
#     .sort_values("AUC", ascending=False)
#     .reset_index(drop=True)
# )
# df_effect
# df_effect.to_csv("./data/train_effect.csv", index=False) # index=Falseで行番号を除外
# # -------------------------------------------------------
# # 部品種別毎の相関
import pandas as pd
import numpy as np
# ----- 数値化 -----
df_tmp = df_train0.copy()
df_tmp[cols] = (
    df_tmp[cols]
    .replace(["-", "－", ""], np.nan)
    .apply(pd.to_numeric, errors="coerce")
)
# ----- product_type を One-Hot化 -----
df_dummies = pd.get_dummies(df_tmp["product_type"], prefix="product")
df_tmp = pd.concat([df_tmp, df_dummies], axis=1)
rows = []
for c in cols:
    row = {"feature": c}    
    for p in df_dummies.columns:
        sub = df_tmp[[c, p]].dropna()      
        if sub[p].nunique() < 2:
            corr = np.nan
        else:
            corr = sub[c].corr(sub[p])       
        row[f"corr_with_{p}"] = corr  
    rows.append(row)
df_product_corr = (
    pd.DataFrame(rows)
    .reset_index(drop=True)
)
df_product_corr
df_product_corr.to_csv("./data/train_product_corr.csv", index=False) # index=Falseで行番号を除外
# # -------------------------------------------------------
# # 箱ひげ図
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# # ---------------------------
# # 保存フォルダ作成
# # ---------------------------
# save_dir = "./11_png/train"
# os.makedirs(save_dir, exist_ok=True)
# # ---------------------------
# # 安全数値化
# # ---------------------------
# df_plot = df_train0.copy()
# df_plot[cols] = (
#     df_plot[cols]
#     .replace(["-", "－", ""], np.nan)
#     .apply(pd.to_numeric, errors="coerce")
# )
# df_plot["is_anomaly"] = pd.to_numeric(df_plot["is_anomaly"], errors="coerce")
# # ---------------------------
# # 特徴量ごとに保存
# # ---------------------------
# for c in cols:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(
#         data=df_plot,
#         x="is_anomaly",
#         y=c,
#         showfliers=False
#     )
#     plt.title(f"{c} (Normal vs Anomaly)")
#     plt.xlabel("is_anomaly (0=Normal, 1=Anomaly)")
#     plt.tight_layout()
#     save_path = os.path.join(save_dir, f"{c}_boxplot.png")
#     plt.savefig(save_path, dpi=300)
#     plt.close()   # メモリ節約
# print("保存完了:", save_dir)
# # -------------------------------------------------------
# # *******************************************************
# # -------------------------------------------------------
df_test0 = df_test.copy()
dstamp = []
for k in range(len(df_test0)):
    dstamp.append(df_test0['timestamp'][k].split()[0])
df_test0['datestamp'] = dstamp
# print(df_test0.groupby("datestamp").size().reset_index(name="count").sort_values("count", ascending=False))
# # -------------------------------------------------------
sstamp = []
for k in range(len(df_test0)):
    a0 = int(df_test0['timestamp'][k].split()[1].split(':')[0])
    a1 = int(df_test0['timestamp'][k].split()[1].split(':')[1])
    sstamp.append(a0*60+a1)
df_test0['secstamp'] = sstamp
# # -------------------------------------------------------
# df_test0.to_csv("./data/train_df.csv", index=False) # index=Falseで行番号を除外
# # -------------------------------------------------------
# 欠損値確認
# print(df_test0.isna().sum())
# print(df_test0.groupby("datestamp")["is_anomaly"].sum())
# print(df_test0[df_test0["run_type"] == "trial"].groupby("datestamp").size())
# print(df_test0[df_test0["product_type"] == "D"].groupby("datestamp").size())
# # -------------------------------------------------------
# # 念のため数値化（"－" → NaN）
# df_test0[cols] = df_test0[cols].apply(pd.to_numeric, errors="coerce")
# # ---------------------------
# # ① 平均・分散
# # ---------------------------
# df_day_stats = (
#     df_test0.groupby("datestamp")[cols]
#     .agg(["mean", "var"])
# )
# df_day_stats.columns = [
#     f"{col}_{stat}" for col, stat in df_day_stats.columns
# ]
# df_day_stats = df_day_stats.reset_index()
# # ---------------------------
# # ② count（有効データ数）
# # ---------------------------
# df_day_count = (
#     df_test0.groupby("datestamp")[cols]
#     .count()
# )
# df_day_count.columns = [
#     f"{col}_count" for col in df_day_count.columns
# ]
# df_day_count = df_day_count.reset_index()
# # 確認
# df_day_stats.head()
# df_day_count.head()
# df_day_stats.to_csv("./data/test_stats_df.csv", index=False) # index=Falseで行番号を除外
# df_day_count.to_csv("./data/test_count_df.csv", index=False) # index=Falseで行番号を除外
# # -------------------------------------------------------
# # 相関算出
# 数値化（"－"などはNaN）
# # 1) '-' を NaN に置換（必要なら '－' や空文字も追加）
# df_test0[cols] = df_test0[cols].replace(["-", "－", ""], np.nan)
# # 2) 数値化（変換できないものはNaN）
# df_test0[cols] = df_test0[cols].apply(pd.to_numeric, errors="coerce")
# # is_anomaly も数値化（True/FalseでもOK）
# df_test0["is_anomaly"] = df_test0["is_anomaly"].replace(["-", "－", ""], np.nan)
# df_test0["is_anomaly"] = pd.to_numeric(df_test0["is_anomaly"], errors="coerce")
# # 3) 相関（各特徴量 vs is_anomaly）
# rows = []
# y = df_test0["is_anomaly"]
# for c in cols:
#     x = df_test0[c]
#     m = x.notna() & y.notna()
#     n_valid = int(m.sum())
#     n_missing = int((~m).sum())
#     # is_anomaly が 0/1 の両方を含まない場合は相関は定義不可
#     if n_valid < 3 or y[m].nunique() < 2:
#         corr = np.nan
#     else:
#         corr = x[m].corr(y[m])  # Pearson（=点双列相関）
#     rows.append({
#         "feature": c,
#         "corr_with_is_anomaly": corr,
#         "n_valid": n_valid,
#         "n_missing": n_missing,
#     })
# df_corr = (
#     pd.DataFrame(rows)
#       .assign(abs_corr=lambda d: d["corr_with_is_anomaly"].abs())
#       .sort_values("abs_corr", ascending=False)
#       .drop(columns="abs_corr")
#       .reset_index(drop=True)
# )
# df_corr
# df_corr.to_csv("./data/test_corr.csv", index=False) # index=Falseで行番号を除外
# # -------------------------------------------------------
# ----- 数値化（安全処理） -----
# df_test0[cols] = (
#     df_test0[cols]
#     .replace(["-", "－", ""], np.nan)
#     .apply(pd.to_numeric, errors="coerce")
# )
# df_test0["is_anomaly"] = pd.to_numeric(df_test0["is_anomaly"], errors="coerce")
# rows = []
# for c in cols:
#     sub = df_test0[[c, "is_anomaly"]].dropna()
#     if sub["is_anomaly"].nunique() < 2:
#         continue
#     normal = sub[sub["is_anomaly"] == 0][c]
#     anomaly = sub[sub["is_anomaly"] == 1][c]
#     if len(normal) < 2 or len(anomaly) < 2:
#         continue
#     mean_normal = normal.mean()
#     mean_anomaly = anomaly.mean()
#     mean_diff = mean_anomaly - mean_normal
#     # Cohen's d
#     pooled_std = np.sqrt(
#         ((normal.std()**2 + anomaly.std()**2) / 2)
#     )
#     effect_size = mean_diff / pooled_std if pooled_std != 0 else np.nan
#     # AUC（単特徴量）
#     try:
#         auc = roc_auc_score(sub["is_anomaly"], sub[c])
#     except:
#         auc = np.nan
#     rows.append({
#         "feature": c,
#         "mean_normal": mean_normal,
#         "mean_anomaly": mean_anomaly,
#         "mean_diff": mean_diff,
#         "effect_size_d": effect_size,
#         "AUC": auc,
#         "n_valid": len(sub)
#     })
# df_effect = (
#     pd.DataFrame(rows)
#     .sort_values("AUC", ascending=False)
#     .reset_index(drop=True)
# )
# df_effect
# df_effect.to_csv("./data/test_effect.csv", index=False) # index=Falseで行番号を除外
# # -------------------------------------------------------
# # # 部品種別毎の相関
# import pandas as pd
# import numpy as np
# # ----- 数値化 -----
# df_tmp = df_test0.copy()
# df_tmp[cols] = (
#     df_tmp[cols]
#     .replace(["-", "－", ""], np.nan)
#     .apply(pd.to_numeric, errors="coerce")
# )
# # ----- product_type を One-Hot化 -----
# df_dummies = pd.get_dummies(df_tmp["product_type"], prefix="product")
# df_tmp = pd.concat([df_tmp, df_dummies], axis=1)
# rows = []
# for c in cols:
#     row = {"feature": c}    
#     for p in df_dummies.columns:
#         sub = df_tmp[[c, p]].dropna()      
#         if sub[p].nunique() < 2:
#             corr = np.nan
#         else:
#             corr = sub[c].corr(sub[p])       
#         row[f"corr_with_{p}"] = corr  
#     rows.append(row)
# df_product_corr = (
#     pd.DataFrame(rows)
#     .reset_index(drop=True)
# )
# df_product_corr
# df_product_corr.to_csv("./data/test_product_corr.csv", index=False) # index=Falseで行番号を除外
# # -------------------------------------------------------
# # 箱ひげ図
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# # ---------------------------
# # 保存フォルダ作成
# # ---------------------------
# save_dir = "./11_png/test"
# os.makedirs(save_dir, exist_ok=True)
# # ---------------------------
# # 安全数値化
# # ---------------------------
# df_plot = df_test0.copy()
# df_plot[cols] = (
#     df_plot[cols]
#     .replace(["-", "－", ""], np.nan)
#     .apply(pd.to_numeric, errors="coerce")
# )
# df_plot["is_anomaly"] = pd.to_numeric(df_plot["is_anomaly"], errors="coerce")
# # ---------------------------
# # 特徴量ごとに保存
# # ---------------------------
# for c in cols:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(
#         data=df_plot,
#         x="is_anomaly",
#         y=c,
#         showfliers=False
#     )
#     plt.title(f"{c} (Normal vs Anomaly)")
#     plt.xlabel("is_anomaly (0=Normal, 1=Anomaly)")
#     plt.tight_layout()
#     save_path = os.path.join(save_dir, f"{c}_boxplot.png")
#     plt.savefig(save_path, dpi=300)
#     plt.close()   # メモリ節約
# print("保存完了:", save_dir)
# # -------------------------------------------------------
# # 異常ラベルのカラム名
# anomaly_label = 'is_anomaly'
# # データ型の分類
# object_cols = df_train.select_dtypes(include=['object']).columns.tolist()
# print("\n" + "="*80)
# print("【Object型カラムの詳細調査】")
# print("="*80)
# if len(object_cols) > 0:
#     # ユニーク数が大きい順にソートして上位3つを選択
#     object_col_info = [(col, df_train[col].nunique()) for col in object_cols]
#     object_col_info_sorted = sorted(object_col_info, key=lambda x: x[1], reverse=True)
#     top_2_cols = object_col_info_sorted[:2]
#     print(f"ユニーク数が多い上位2つを調査:")
#     for col, unique_count in top_2_cols:
#         print(f"\n■ カラム: '{col}' (ユニーク数: {unique_count})")  
#         # 数値変換不可の値（ゴミデータ）を抽出
#         numeric_conversion = pd.to_numeric(df_train[col], errors='coerce')
#         non_numeric_mask = numeric_conversion.isna() & df_train[col].notna()
#         non_numeric_values = df_train[col][non_numeric_mask].unique()
#         conversion_success_rate = (1 - numeric_conversion.isna().sum() / len(df_train)) * 100
#         print(f"\n  【数値変換】")
#         print(f"    - 変換成功率: {conversion_success_rate:.2f}%") 
#         if len(non_numeric_values) > 0:
#             print(f"    - 数値でない値（ゴミデータ）: {len(non_numeric_values)}種類")
#             print(f"    - サンプル: {non_numeric_values[:10].tolist()}")
#         else:
#             print(f"    - 全て数値として変換可能")
# else:
#     print("\nObject型のカラムはありません。")
# print("\n✓データ品質チェック完了")
# # # 異常値（'-'を含む行）の削除
# cols_to_convert_numeric = [col for col, unique_count in object_col_info_sorted if unique_count >= 10]
# df_train_clean = df_train[~df_train['ftr_bdc_ratio'].astype(str).str.contains('-')].copy()
# df_test_clean = df_test[~df_test['ftr_bdc_ratio'].astype(str).str.contains('-')].copy()
# # object型カラムを数値型に変換
# for col in cols_to_convert_numeric:
#     df_train_clean[col] = pd.to_numeric(df_train_clean[col], errors='coerce')
#     df_test_clean[col] = pd.to_numeric(df_test_clean[col], errors='coerce')
# # 欠損値の前方補完
# df_train_clean = df_train_clean.ffill()
# df_test_clean = df_test_clean.ffill()
# # データ型の詳細確認（データクリーニング後）
# print("\n■ TRAINデータセットのカラム情報（データクリーニング後）:")
# dtype_info = pd.DataFrame({
#     'カラム名': df_train_clean.columns,
#     'データ型': df_train_clean.dtypes.values,
#     'ユニーク数': [df_train_clean[col].nunique() for col in df_train_clean.columns],
#     '欠損数': df_train_clean.isnull().sum().values,
#     '欠損率(%)': (df_train_clean.isnull().sum() / len(df_train_clean) * 100).values
# })
# display(dtype_info)
# # 可視化する特徴量を指定（例）
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
# # # -----------------------------
# import matplotlib.pyplot as plt
# # 可視化する特徴量を指定（例）
# features_to_plot = ['ftr_bdc_ratio', 'ftr_asymmetry', 'ftr_vib_high_band']
# # ---- 異常区間の抽出（ここはあなたのコードのままでOK）----
# # anomaly_periods = [(start_idx, end_idx), ...] ができている前提
# # ---- matplotlib で描画 ----
# # x軸（stroke_id）を numpy 配列にしておくと扱いやすい
# x = df_train_clean['stroke_id'].to_numpy()
# for feature in features_to_plot:
#     if feature not in df_train_clean.columns:
#         print(f"特徴量 '{feature}' がデータに存在しません。スキップします。")
#         continue
#     y = df_train_clean[feature].to_numpy()
#     fig, ax = plt.subplots(figsize=(12, 4))
#     # 異常区間を赤い背景で塗る（Plotlyのvrect相当）
#     for start, end in anomaly_periods:
#         x0 = df_train_clean.iloc[start]['stroke_id']
#         x1 = df_train_clean.iloc[end]['stroke_id']
#         ax.axvspan(x0, x1, alpha=0.2, color='red', linewidth=0)
#     # 時系列を描画（線＋必要なら点も）
#     ax.plot(x, y, linewidth=1, label=feature)
#     ax.set_title(f'訓練データ: {feature} の時系列推移')
#     ax.set_xlabel('ストロークID')
#     ax.set_ylabel(feature)
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # *******************************************************
# # -------------------------------------------------------
# testを2025-11-01を軸にvalid,testに分割
df_valid = df_test[df_test['timestamp'] < '2025-11-01']
df_test = df_test[df_test['timestamp'] >= '2025-11-01']
# 異常ラベルのカラム名
anomaly_label = 'is_anomaly'
# 各データセットの基本情報
datasets = {
    'train': df_train,
    'valid': df_valid,
    'test': df_test
}
# # -------------------------------------------------------
# 異常率確認
for name, df in datasets.items():
    print(f"\n■ {name.upper()}データセット")
    print(f"  - データ数: {len(df):,}行")
    print(f"  - カラム数: {len(df.columns)}列")
    # 異常率を計算
    if 'is_anomaly' in df.columns:
        anomaly_rate = df['is_anomaly'].sum() / len(df) * 100
        print(f"  - 異常率: {anomaly_rate:.2f}%")
    else:
        print(f"  - 異常率: N/A (is_anomalyカラムなし)")
# -------------------------------------------------------
# データ型の詳細確認
print("\n■ TRAINデータセットのカラム情報:")
dtype_info = pd.DataFrame({
    'カラム名': df_train.columns,
    'データ型': df_train.dtypes.values,
    'ユニーク数': [df_train[col].nunique() for col in df_train.columns],
    '欠損数': df_train.isnull().sum().values,
    '欠損率(%)': (df_train.isnull().sum() / len(df_train) * 100).values
})
display(dtype_info)
# -------------------------------------------------------
object_cols = df_train.select_dtypes(include=['object']).columns.tolist()
# -------------------------------------------------------
# データ型の詳細確認
print("\n■ TESTデータセットのカラム情報:")
dtype_info = pd.DataFrame({
    'カラム名': df_test.columns,
    'データ型': df_test.dtypes.values,
    'ユニーク数': [df_test[col].nunique() for col in df_train.columns],
    '欠損数': df_test.isnull().sum().values,
    '欠損率(%)': (df_test.isnull().sum() / len(df_train) * 100).values
})
display(dtype_info)
# -------------------------------------------------------
# データ型の詳細確認
print("\n■ TESTデータセットのカラム情報:")
dtype_info = pd.DataFrame({
    'カラム名': df_valid.columns,
    'データ型': df_valid.dtypes.values,
    'ユニーク数': [df_valid[col].nunique() for col in df_train.columns],
    '欠損数': df_valid.isnull().sum().values,
    '欠損率(%)': (df_valid.isnull().sum() / len(df_train) * 100).values
})
display(dtype_info)
# -------------------------------------------------------
object_cols = df_train.select_dtypes(include=['object']).columns.tolist()
if len(object_cols) > 0:
    # ユニーク数が大きい順にソートして上位3つを選択
    object_col_info = [(col, df_train[col].nunique()) for col in object_cols]
    object_col_info_sorted = sorted(object_col_info, key=lambda x: x[1], reverse=True)
    top_2_cols = object_col_info_sorted[:2]
    
    print(f"ユニーク数が多い上位2つを調査:")
    
    for col, unique_count in top_2_cols:
        print(f"\n■ カラム: '{col}' (ユニーク数: {unique_count})")
        
        # 数値変換不可の値（ゴミデータ）を抽出
        numeric_conversion = pd.to_numeric(df_train[col], errors='coerce')
        non_numeric_mask = numeric_conversion.isna() & df_train[col].notna()
        non_numeric_values = df_train[col][non_numeric_mask].unique()
        
        conversion_success_rate = (1 - numeric_conversion.isna().sum() / len(df_train)) * 100
        print(f"\n  【数値変換】")
        print(f"    - 変換成功率: {conversion_success_rate:.2f}%")
        
        if len(non_numeric_values) > 0:
            print(f"    - 数値でない値（ゴミデータ）: {len(non_numeric_values)}種類")
            print(f"    - サンプル: {non_numeric_values[:10].tolist()}")
        else:
            print(f"    - 全て数値として変換可能")

else:
    print("\nObject型のカラムはありません。")
# # -------------------------------------------------------
# # 異常検知用の特徴量を選定
selected_features = [
    'ftr_bdc_ratio',
    'ftr_asymmetry',
    'ftr_vib_high_band',
    'peak_angle_deg',
    'lube_pressure_bar',
    'vibration_rms_g',
    'peak_tonnage_kN'
]
anomaly_label = 'is_anomaly'
# 訓練/検証/テストデータを作成
df_train_selected = df_train_clean[selected_features + [anomaly_label]]
df_valid_selected = df_valid_clean[selected_features + [anomaly_label]]
df_test_selected = df_test_clean[selected_features + [anomaly_label]]
df_train_selected.head()
# # # -------------------------------------------------------
# 7_スライド窓変換
def sliding_window_transform(df: pd.DataFrame, window_size: int, anomaly_label: str) -> pd.DataFrame:
    """
    時系列データにスライディングウィンドウ変換を適用し、各ウィンドウを1行に変換します。  
    各特徴量のwindow_size個の連続した値を横方向にフラット化して1行とします。
    （例：window_size=30の場合、各特徴量のt0〜t29の値を"特徴量名_t0"〜"特徴量名_t29"として並べる）
    異常ラベル（anomaly_label）のカラムはスライド窓変換せず、
    ウィンドウ期間内に1回でも異常（True）があれば、その窓の"is_window_anomaly"をTrueとします。
    引数:
        df (pd.DataFrame): 入力データフレーム（時系列データ）
        window_size (int): スライディングウィンドウのサイズ（時間ステップ数）
        anomaly_label (str): 異常ラベルのカラム名
    戻り値:
        pd.DataFrame: 各行が1つのウィンドウを表すデータフレーム
                      特徴量カラム: "{元のカラム名}_t{時刻}" 形式
                      異常ラベル: "is_window_anomaly"
    """
    windows = []
    # 異常ラベル以外の特徴量カラムを取得
    feature_cols = [col for col in df.columns if col != anomaly_label]   
    # スライディングウィンドウでデータを走査
    for i in range(len(df) - window_size + 1):
        window_features = []
        # 各特徴量のwindow_size分の値を取得してフラット化
        for col in feature_cols:
            window_features.extend(df.iloc[i:i+window_size][col].values)
        # ウィンドウ内に1つでも異常があればTrueとする
        is_window_anomaly = df.iloc[i:i+window_size][anomaly_label].any()
        # 特徴量と異常ラベルを結合
        window = window_features + [is_window_anomaly]
        windows.append(window)
    # カラム名を生成（特徴量_t0, 特徴量_t1, ..., is_window_anomaly）
    columns = [f"{col}_t{t}" for col in feature_cols for t in range(window_size)] + ["is_window_anomaly"]
    return pd.DataFrame(windows, columns=columns)
# スライド窓の窓幅
window_size = 30
# 訓練/検証/テストデータのスライド窓変換
df_train_win = sliding_window_transform(df_train_selected, window_size=window_size, anomaly_label=anomaly_label)
df_valid_win = sliding_window_transform(df_valid_selected, window_size=window_size, anomaly_label=anomaly_label)
df_test_win = sliding_window_transform(df_test_selected, window_size=window_size, anomaly_label=anomaly_label)
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
# # データ取得
# df_train = pd.read_csv("./data/train.csv", parse_dates=['timestamp'], index_col='timestamp')
# df_test = pd.read_csv("./data/test.csv", parse_dates=['timestamp'], index_col='timestamp')

# # testを2025-11-01を軸にvalid,testに分割
# df_valid = df_test[df_test.index < '2025-11-01']
# df_test = df_test[df_test.index >= '2025-11-01']

# # 異常ラベルのカラム名
# anomaly_label = 'is_anomaly'

# display(df_train.head())
# display(df_valid.head())
# display(df_test.head())
# # # -------------------------------------------------------
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

# # データ型の分類
# object_cols = df_train.select_dtypes(include=['object']).columns.tolist()

# print("\n" + "="*80)
# print("【Object型カラムの詳細調査】")
# print("="*80)

# if len(object_cols) > 0:
#     # ユニーク数が大きい順にソートして上位3つを選択
#     object_col_info = [(col, df_train[col].nunique()) for col in object_cols]
#     object_col_info_sorted = sorted(object_col_info, key=lambda x: x[1], reverse=True)
#     top_2_cols = object_col_info_sorted[:2]
    
#     print(f"ユニーク数が多い上位2つを調査:")
    
#     for col, unique_count in top_2_cols:
#         print(f"\n■ カラム: '{col}' (ユニーク数: {unique_count})")
        
#         # 数値変換不可の値（ゴミデータ）を抽出
#         numeric_conversion = pd.to_numeric(df_train[col], errors='coerce')
#         non_numeric_mask = numeric_conversion.isna() & df_train[col].notna()
#         non_numeric_values = df_train[col][non_numeric_mask].unique()
        
#         conversion_success_rate = (1 - numeric_conversion.isna().sum() / len(df_train)) * 100
#         print(f"\n  【数値変換】")
#         print(f"    - 変換成功率: {conversion_success_rate:.2f}%")
        
#         if len(non_numeric_values) > 0:
#             print(f"    - 数値でない値（ゴミデータ）: {len(non_numeric_values)}種類")
#             print(f"    - サンプル: {non_numeric_values[:10].tolist()}")
#         else:
#             print(f"    - 全て数値として変換可能")

# else:
#     print("\nObject型のカラムはありません。")

# print("\n✓データ品質チェック完了")
# # # # -------------------------------------------------------
# # object型でユニーク値が10以上のカラムを抽出（数値型に変換対象）
# cols_to_convert_numeric = [col for col, unique_count in object_col_info_sorted if unique_count >= 10]

# # 異常値（'-'を含む行）の削除
# df_train_clean = df_train[~df_train['ftr_bdc_ratio'].astype(str).str.contains('-')].copy()
# df_valid_clean = df_valid[~df_valid['ftr_bdc_ratio'].astype(str).str.contains('-')].copy()
# df_test_clean = df_test[~df_test['ftr_bdc_ratio'].astype(str).str.contains('-')].copy()

# # object型カラムを数値型に変換
# for col in cols_to_convert_numeric:
#     df_train_clean[col] = pd.to_numeric(df_train_clean[col], errors='coerce')
#     df_valid_clean[col] = pd.to_numeric(df_valid_clean[col], errors='coerce')
#     df_test_clean[col] = pd.to_numeric(df_test_clean[col], errors='coerce')

# # 欠損値の前方補完
# df_train_clean = df_train_clean.ffill()
# df_valid_clean = df_valid_clean.ffill()
# df_test_clean = df_test_clean.ffill()

# # データ型の詳細確認（データクリーニング後）
# print("\n■ TRAINデータセットのカラム情報（データクリーニング後）:")
# dtype_info = pd.DataFrame({
#     'カラム名': df_train_clean.columns,
#     'データ型': df_train_clean.dtypes.values,
#     'ユニーク数': [df_train_clean[col].nunique() for col in df_train_clean.columns],
#     '欠損数': df_train_clean.isnull().sum().values,
#     '欠損率(%)': (df_train_clean.isnull().sum() / len(df_train_clean) * 100).values
# })
# display(dtype_info)

# # # # -------------------------------------------------------
# # 5_基本統計量と時系列可視化
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

# # 可視化する特徴量を指定（例）
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
# # # # -------------------------------------------------------
# # # 異常検知用の特徴量を選定
# selected_features = [
#     'ftr_bdc_ratio',
#     'ftr_asymmetry',
#     'ftr_vib_high_band',
#     'peak_angle_deg',
#     'lube_pressure_bar',
#     'vibration_rms_g',
#     'peak_tonnage_kN'
# ]
# anomaly_label = 'is_anomaly'

# # 訓練/検証/テストデータを作成
# df_train_selected = df_train_clean[selected_features + [anomaly_label]]
# df_valid_selected = df_valid_clean[selected_features + [anomaly_label]]
# df_test_selected = df_test_clean[selected_features + [anomaly_label]]

# df_train_selected.head()
# # # # -------------------------------------------------------
# # 7_スライド窓変換
# def sliding_window_transform(df: pd.DataFrame, window_size: int, anomaly_label: str) -> pd.DataFrame:
#     """
#     時系列データにスライディングウィンドウ変換を適用し、各ウィンドウを1行に変換します。
    
#     各特徴量のwindow_size個の連続した値を横方向にフラット化して1行とします。
#     （例：window_size=30の場合、各特徴量のt0〜t29の値を"特徴量名_t0"〜"特徴量名_t29"として並べる）
    
#     異常ラベル（anomaly_label）のカラムはスライド窓変換せず、
#     ウィンドウ期間内に1回でも異常（True）があれば、その窓の"is_window_anomaly"をTrueとします。

#     引数:
#         df (pd.DataFrame): 入力データフレーム（時系列データ）
#         window_size (int): スライディングウィンドウのサイズ（時間ステップ数）
#         anomaly_label (str): 異常ラベルのカラム名

#     戻り値:
#         pd.DataFrame: 各行が1つのウィンドウを表すデータフレーム
#                       特徴量カラム: "{元のカラム名}_t{時刻}" 形式
#                       異常ラベル: "is_window_anomaly"
#     """
#     windows = []
#     # 異常ラベル以外の特徴量カラムを取得
#     feature_cols = [col for col in df.columns if col != anomaly_label]
    
#     # スライディングウィンドウでデータを走査
#     for i in range(len(df) - window_size + 1):
#         window_features = []
#         # 各特徴量のwindow_size分の値を取得してフラット化
#         for col in feature_cols:
#             window_features.extend(df.iloc[i:i+window_size][col].values)
#         # ウィンドウ内に1つでも異常があればTrueとする
#         is_window_anomaly = df.iloc[i:i+window_size][anomaly_label].any()
#         # 特徴量と異常ラベルを結合
#         window = window_features + [is_window_anomaly]
#         windows.append(window)
    
#     # カラム名を生成（特徴量_t0, 特徴量_t1, ..., is_window_anomaly）
#     columns = [f"{col}_t{t}" for col in feature_cols for t in range(window_size)] + ["is_window_anomaly"]
#     return pd.DataFrame(windows, columns=columns)

# # スライド窓の窓幅
# window_size = 30

# # 訓練/検証/テストデータのスライド窓変換
# df_train_win = sliding_window_transform(df_train_selected, window_size=window_size, anomaly_label=anomaly_label)
# df_valid_win = sliding_window_transform(df_valid_selected, window_size=window_size, anomaly_label=anomaly_label)
# df_test_win = sliding_window_transform(df_test_selected, window_size=window_size, anomaly_label=anomaly_label)

# df_train_win.head()
# # # # -------------------------------------------------------
# # # 正解ラベル(is_window_anomaly)の削除
# train = df_train_win.drop(columns=['is_window_anomaly'])
# valid = df_valid_win.drop(columns=['is_window_anomaly'])
# test = df_test_win.drop(columns=['is_window_anomaly'])

# # 正解ラベルの抽出
# valid_label = df_valid_win['is_window_anomaly'].astype(int)
# test_label = df_test_win['is_window_anomaly'].astype(int)

# train.head()
# # # # -------------------------------------------------------
# # # kNNモデルの初期化（デフォルトパラメータ：n_neighbors=5, contamination=0.1）
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

