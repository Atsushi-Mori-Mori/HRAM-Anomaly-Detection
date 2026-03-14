#　-*- coding: utf-8 -*-
import sys
import os
import re
import struct
import binascii
import numpy as np
# -------------------------------------------------------
# ============================================================
# Build mixed SFT dataset:
#  - base: u-10bei/structured_data_with_cot_dataset_512_v5
#  - add from v2/v4: only ["Text to TOML", "CSV to JSON"]
#  - downsample in v5: ["Text to YAML", "Text to JSON"] to 1/3
# ============================================================

import re
import random
from datasets import load_dataset, concatenate_datasets

SEED = 3407
random.seed(SEED)

V2 = "u-10bei/structured_data_with_cot_dataset_512_v2"
V4 = "u-10bei/structured_data_with_cot_dataset_512_v4"
V5 = "u-10bei/structured_data_with_cot_dataset_512_v5"

OUT_JSONL = "/content/sft_mix_v5plus_v2v4_textTOML_csvJSON.jsonl"

# -----------------------------
# 1) Task type detector
# -----------------------------
def _get_user_text(ex):
    # messages: [{role, content}, ...]
    msgs = ex.get("messages", [])
    for m in msgs:
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""

def detect_task_type(ex):
    """
    Returns one of:
      - "Text to TOML"
      - "CSV to JSON"
      - "Text to YAML"
      - "Text to JSON"
      - or "Other"
    Heuristic based on user prompt text.
    """
    t = _get_user_text(ex).strip()
    tl = t.lower()

    # ---- CSV to JSON (most reliable) ----
    # Patterns:
    #  - "Transform this CSV ... to JSON"
    #  - "Convert this CSV ... into JSON"
    #  - includes both CSV and JSON keywords
    if ("csv" in tl) and ("json" in tl):
        # Ensure it's actually conversion and not e.g. "JSON to CSV"
        # Heuristic: look for "csv ... to json" direction
        # - If contains "to json" and mentions csv earlier -> CSV to JSON
        if re.search(r"csv.*to\s+json", tl, flags=re.DOTALL) or re.search(r"csv.*into\s+json", tl, flags=re.DOTALL):
            return "CSV to JSON"
        # Some prompts say "Transform this data from CSV to JSON"
        if re.search(r"from\s+csv\s+to\s+json", tl):
            return "CSV to JSON"
        # If it just mentions both, treat as CSV to JSON only when it's not clearly JSON->CSV
        if not re.search(r"from\s+json\s+to\s+csv", tl) and not re.search(r"json.*to\s+csv", tl, flags=re.DOTALL):
            # fallback: many CSV->JSON prompts contain "CSV" and "JSON" only
            if "to json" in tl or "into json" in tl:
                return "CSV to JSON"

    # ---- Text to TOML ----
    # Examples:
    #  - "Extract ... and output TOML"
    #  - "Return ONLY TOML"
    #  - "Text to TOML" tasks often don't specify a source format explicitly (it's free text)
    toml_out = ("output toml" in tl) or ("return only toml" in tl) or re.search(r"\binto\s+toml\b", tl) or re.search(r"\bto\s+toml\b", tl)
    # If it says "from X to TOML" where X is JSON/YAML/XML/CSV/TOML -> that's NOT "Text to TOML"
    if toml_out:
        if re.search(r"from\s+(json|yaml|yml|xml|csv|toml)\s+to\s+toml", tl):
            return "Other"
        if re.search(r"(json|yaml|yml|xml|csv|toml)\s+to\s+toml", tl) and "text" not in tl:
            # likely structured->TOML
            return "Other"
        # Otherwise treat as Text->TOML
        return "Text to TOML"

    # ---- Text to YAML ----
    yaml_out = ("output yaml" in tl) or ("return only yaml" in tl) or re.search(r"\binto\s+yaml\b", tl) or re.search(r"\bto\s+yaml\b", tl)
    if yaml_out:
        if re.search(r"from\s+(json|yaml|yml|xml|csv|toml)\s+to\s+yaml", tl):
            return "Other"
        if re.search(r"(json|yaml|yml|xml|csv|toml)\s+to\s+yaml", tl) and "text" not in tl:
            return "Other"
        return "Text to YAML"

    # ---- Text to JSON ----
    json_out = ("output json" in tl) or ("return only json" in tl) or re.search(r"\binto\s+json\b", tl) or re.search(r"\bto\s+json\b", tl)
    if json_out:
        # exclude "from YAML/CSV/XML/TOML to JSON" etc
        if re.search(r"from\s+(json|yaml|yml|xml|csv|toml)\s+to\s+json", tl):
            return "Other"
        if re.search(r"(yaml|yml|xml|csv|toml|json)\s+to\s+json", tl) and "text" not in tl:
            return "Other"
        return "Text to JSON"

    return "Other"

def add_task_type_column(ds):
    return ds.map(lambda ex: {"task_type": detect_task_type(ex)})

def print_counts(ds, title):
    from collections import Counter
    c = Counter(ds["task_type"])
    print(f"\n=== {title} ===")
    for k, v in c.most_common():
        print(f"{k:14s}: {v}")

# -----------------------------
# 2) Load datasets
# -----------------------------
print("[INFO] Loading datasets...")
ds_v2 = load_dataset(V2, split="train")
ds_v4 = load_dataset(V4, split="train")
ds_v5 = load_dataset(V5, split="train")

# Add task_type
print("[INFO] Detecting task types...")
ds_v2 = add_task_type_column(ds_v2)
ds_v4 = add_task_type_column(ds_v4)
ds_v5 = add_task_type_column(ds_v5)

print_counts(ds_v5, "v5 task_type counts (raw)")
print_counts(ds_v2, "v2 task_type counts (raw)")
print_counts(ds_v4, "v4 task_type counts (raw)")

# -----------------------------
# 3) Downsample v5: Text to YAML & Text to JSON -> 1/3
# -----------------------------
def downsample_types(ds, target_types, keep_ratio=1/3, seed=SEED):
    idx_keep = []
    rng = random.Random(seed)

    for i in range(len(ds)):
        tt = ds[i]["task_type"]
        if tt in target_types:
            if rng.random() < keep_ratio:
                idx_keep.append(i)
        else:
            idx_keep.append(i)
    return ds.select(idx_keep)

print("[INFO] Downsampling v5 Text to YAML / Text to JSON -> 1/3 ...")
ds_v5_adj = downsample_types(ds_v5, {"Text to YAML", "Text to JSON"}, keep_ratio=1/3, seed=SEED)
print_counts(ds_v5_adj, "v5 after downsample(1/3)")

# -----------------------------
# 4) Extract from v2/v4: only ["Text to TOML", "CSV to JSON"]
# -----------------------------
def filter_only(ds, allowed):
    return ds.filter(lambda ex: ex["task_type"] in allowed)

allowed_add = {"Text to TOML", "CSV to JSON"}

print("[INFO] Filtering v2 additions (Text to TOML, CSV to JSON)...")
ds_v2_add = filter_only(ds_v2, allowed_add)
print_counts(ds_v2_add, "v2 additions")

print("[INFO] Filtering v4 additions (Text to TOML, CSV to JSON)...")
ds_v4_add = filter_only(ds_v4, allowed_add)
print_counts(ds_v4_add, "v4 additions")

# -----------------------------
# 5) Combine + shuffle
# -----------------------------
print("[INFO] Concatenating datasets...")
ds_mix = concatenate_datasets([ds_v5_adj, ds_v2_add, ds_v4_add])

print_counts(ds_mix, "MIX before shuffle")

print("[INFO] Shuffling...")
ds_mix = ds_mix.shuffle(seed=SEED)

# -----------------------------
# 6) Save as jsonl (messages only recommended)
# -----------------------------
# Keep only the columns you need (messages + metadata). task_type is optional but useful for debugging.
keep_cols = []
for col in ["messages", "metadata", "task_type"]:
    if col in ds_mix.column_names:
        keep_cols.append(col)

ds_out = ds_mix.remove_columns([c for c in ds_mix.column_names if c not in keep_cols])

print(f"[INFO] Saving jsonl to: {OUT_JSONL}")
ds_out.to_json(OUT_JSONL, orient="records", lines=True, force_ascii=False)

print_counts(ds_out, "FINAL (saved) task_type counts")
print("[OK] Done.")
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

