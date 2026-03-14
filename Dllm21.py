#　-*- coding: utf-8 -*-
import sys
import os
import re
import struct
import binascii
import numpy as np
# -------------------------------------------------------
# # -------------------------------------------------------
import re
import random
from datasets import load_dataset, Dataset, concatenate_datasets

SEED = 3407
random.seed(SEED)

SRC_DATASET = "u-10bei/structured_data_with_cot_dataset_512_v2"
SPLIT = "train"

# ====== ここを調整 ======
# 例：Text->TOML を 3倍、Text->YAML を 0.3倍（=70%削減）
MULTIPLIERS = {
    "Text to TOML": 3.0,
    c: 0.3,
    # 必要なら追加
    # "CSV to JSON": 1.5,
    # "XML to JSON": 1.2,
}

# 解析できなかったものはそのまま残す（True）/捨てる（False）
KEEP_UNKNOWN = True

OUT_JSONL = "/content/sft_mix_adjusted.jsonl"
# =======================


# ---------- 1) 変換タイプ推定（user contentから "A to B" を抽出） ----------
def infer_task_type_from_user_content(text: str) -> str:
    if not isinstance(text, str):
        return "Unknown"

    t = text.strip()

    # 0) まず「出力フォーマット」を強く推定（Return ONLY TOML / output TOML / into TOML format 等）
    def detect_output(tt: str):
        if re.search(r"\b(return|output)\s+only\s+toml\b", tt, re.I): return "TOML"
        if re.search(r"\b(return|output)\s+only\s+json\b", tt, re.I): return "JSON"
        if re.search(r"\b(return|output)\s+only\s+yaml\b", tt, re.I): return "YAML"
        if re.search(r"\b(return|output)\s+only\s+xml\b",  tt, re.I): return "XML"
        if re.search(r"\b(return|output)\s+only\s+csv\b",  tt, re.I): return "CSV"

        if re.search(r"\boutput\s+toml\b", tt, re.I): return "TOML"
        if re.search(r"\boutput\s+json\b", tt, re.I): return "JSON"
        if re.search(r"\boutput\s+yaml\b", tt, re.I): return "YAML"
        if re.search(r"\boutput\s+xml\b",  tt, re.I): return "XML"
        if re.search(r"\boutput\s+csv\b",  tt, re.I): return "CSV"

        if re.search(r"\binto\s+toml\b", tt, re.I): return "TOML"
        if re.search(r"\binto\s+json\b", tt, re.I): return "JSON"
        if re.search(r"\binto\s+yaml\b", tt, re.I): return "YAML"
        if re.search(r"\binto\s+xml\b",  tt, re.I): return "XML"
        if re.search(r"\binto\s+csv\b",  tt, re.I): return "CSV"

        if re.search(r"\btoml\s+format\b", tt, re.I): return "TOML"
        if re.search(r"\bjson\s+format\b", tt, re.I): return "JSON"
        if re.search(r"\byaml\s+format\b", tt, re.I): return "YAML"
        if re.search(r"\bxml\s+format\b",  tt, re.I): return "XML"
        if re.search(r"\bcsv\s+format\b",  tt, re.I): return "CSV"

        return None

    # 1) パターン1: "from X to Y:"
    m = re.search(r"from\s+([A-Za-z0-9\-\_ ]+?)\s+to\s+([A-Za-z0-9\-\_ ]+?)\s*:", t, flags=re.IGNORECASE)
    if m:
        src = m.group(1).strip().upper()
        dst = m.group(2).strip().upper()
        return f"{src.title()} to {dst.title()}".replace("Json", "JSON").replace("Xml", "XML").replace("Yaml", "YAML").replace("Toml", "TOML").replace("Csv", "CSV").replace("Text", "Text")

    # 2) パターン2: "CSV data into TOML format:"
    m = re.search(r"Transform\s+this\s+([A-Za-z0-9\-\_ ]+?)\s+data\s+into\s+([A-Za-z0-9\-\_ ]+?)\s+format\s*:", t, flags=re.IGNORECASE)
    if m:
        src = m.group(1).strip().upper()
        dst = m.group(2).strip().upper()
        return f"{src.title()} to {dst.title()}".replace("Json", "JSON").replace("Xml", "XML").replace("Yaml", "YAML").replace("Toml", "TOML").replace("Csv", "CSV").replace("Text", "Text")

    # 3) ここからが強化部分：from/to が無い場合に「入力」と「出力」を推定する
    out_fmt = detect_output(t)

    # 入力側：ユーザー文の中に "CSV:" "YAML:" "XML:" "JSON:" "TOML:" や "Transform this YAML" 等があるか
    # 見つからなければ Text 扱い
    in_fmt = None

    # "Transform this YAML ..." みたいな明示
    m = re.search(r"\bTransform\s+this\s+(CSV|JSON|YAML|XML|TOML)\b", t, re.I)
    if m:
        in_fmt = m.group(1).upper()

    # "from YAML" がないケースでも "YAML:" "JSON:" のようにTEXT中にタグがあるパターンを拾う
    if in_fmt is None:
        for cand in ["CSV", "JSON", "YAML", "XML", "TOML"]:
            # 例: "TEXT:\n..." は input=Text だが、"YAML:\n..." なら input=YAML
            if re.search(rf"\b{cand}\s*:\s*\n", t, re.I) or re.search(rf"\b{cand}\s*:\s*", t, re.I):
                in_fmt = cand
                break

    # "TEXT:" を強く Text とみなす（上で JSON: と誤検出しないように最後に）
    if re.search(r"\bTEXT\s*:\s*", t, re.I):
        in_fmt = "Text"

    if out_fmt is not None:
        if in_fmt is None:
            in_fmt = "Text"
        # 表記をそろえる
        in_disp = in_fmt if in_fmt == "Text" else in_fmt.title()
        out_disp = out_fmt.title()
        # title() すると Json になるので補正
        out_disp = out_disp.replace("Json", "JSON").replace("Xml", "XML").replace("Yaml", "YAML").replace("Toml", "TOML").replace("Csv", "CSV")
        in_disp  = in_disp.replace("Json", "JSON").replace("Xml", "XML").replace("Yaml", "YAML").replace("Toml", "TOML").replace("Csv", "CSV")
        return f"{in_disp} to {out_disp}"

    return "Unknown"


def add_task_type(example):
    # 多くのデータは system,user,assistant の順だが安全に探す
    msgs = example.get("messages", [])
    user_text = ""
    for m in msgs:
        if m.get("role") == "user":
            user_text = m.get("content", "")
            break
    example["task_type"] = infer_task_type_from_user_content(user_text)
    return example


# ---------- 2) まずロードして task_type を付与 ----------
ds = load_dataset(SRC_DATASET, split=SPLIT)
ds = ds.map(add_task_type, desc="infer task_type")

# 集計表示
from collections import Counter
cnt = Counter(ds["task_type"])
print("=== task_type counts (before) ===")
for k, v in cnt.most_common(30):
    print(f"{k:16s} : {v}")


# ---------- 3) タイプごとに up/down サンプリングして結合 ----------
def apply_multipliers(ds: Dataset, multipliers: dict, seed: int, keep_unknown: bool=True) -> Dataset:
    # グルーピング
    types = set(ds["task_type"])
    out_parts = []

    for tt in sorted(types):
        subset = ds.filter(lambda x, tt=tt: x["task_type"] == tt)

        if tt == "Unknown" and not keep_unknown:
            continue

        mult = multipliers.get(tt, 1.0)

        if mult == 1.0:
            out_parts.append(subset)
            continue

        n = len(subset)
        if n == 0:
            continue

        if mult > 1.0:
            # 複製（with replacement）
            new_n = int(round(n * mult))
            rng = random.Random(seed + hash(tt) % 1000000)
            idxs = [rng.randrange(0, n) for _ in range(new_n)]
            out_parts.append(subset.select(idxs))

        elif 0.0 < mult < 1.0:
            # 間引き（without replacement）
            new_n = max(1, int(round(n * mult)))
            rng = random.Random(seed + hash(tt) % 1000000)
            idxs = list(range(n))
            rng.shuffle(idxs)
            idxs = idxs[:new_n]
            out_parts.append(subset.select(idxs))

        else:
            # mult <= 0 は全部削除
            continue

    mixed = concatenate_datasets(out_parts)
    mixed = mixed.shuffle(seed=seed)
    return mixed


ds_adj = apply_multipliers(ds, MULTIPLIERS, seed=SEED, keep_unknown=KEEP_UNKNOWN)

cnt2 = Counter(ds_adj["task_type"])
print("\n=== task_type counts (after) ===")
for k, v in cnt2.most_common(30):
    print(f"{k:16s} : {v}")

print("\nTotal before:", len(ds), "Total after:", len(ds_adj))


# ---------- 4) jsonl に保存（SFTコードで load_dataset できる形） ----------
# messages列はそのまま、余計な列は残してもOKだが、念のため主要列だけ残す例
keep_cols = [c for c in ds_adj.column_names]  # 全列保持
ds_adj = ds_adj.select_columns(keep_cols)

ds_adj.to_json(OUT_JSONL, orient="records", lines=True, force_ascii=False)
print("\n[OK] wrote:", OUT_JSONL)
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

