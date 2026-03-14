#　-*- coding: utf-8 -*-
import sys
import os
import re
import struct
import binascii
import numpy as np
# -------------------------------------------------------
# ============================================================
# Improved SFT dataset builder (quality-first version)
#
# Base:
#   u-10bei/structured_data_with_cot_dataset_512_v5
#
# Add from v2/v4:
#   - Text to TOML
#   - CSV to JSON
#
# Downsample in v5:
#   - Text to YAML (1/3, keep ONLY-output prompts)
#   - Text to JSON (1/3, keep ONLY-output prompts)
#
# Critical fix:
#   - Remove explanation text from assistant outputs
#   - Keep ONLY structured outputs ({ [ < ...)
# ============================================================

import re
import random
from collections import Counter
from datasets import load_dataset, concatenate_datasets

SEED = 3407
random.seed(SEED)

V2 = "u-10bei/structured_data_with_cot_dataset_512_v2"
V4 = "u-10bei/structured_data_with_cot_dataset_512_v4"
V5 = "u-10bei/structured_data_with_cot_dataset_512_v5"

OUT_JSONL = "/content/sft_mix_v5plus_v2v4_clean.jsonl"

# ------------------------------------------------------------
# 1) Utility: get user text
# ------------------------------------------------------------
def _get_user_text(ex):
    for m in ex.get("messages", []):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""

# ------------------------------------------------------------
# 2) Task type detector (same logic as your working version)
# ------------------------------------------------------------
def detect_task_type(ex):
    t = _get_user_text(ex).lower()

    if "csv" in t and "json" in t:
        if re.search(r"csv.*(to|into)\s+json", t) or re.search(r"from\s+csv\s+to\s+json", t):
            return "CSV to JSON"

    toml_out = (
        "output toml" in t or
        "return only toml" in t or
        re.search(r"\bto\s+toml\b", t) or
        re.search(r"\binto\s+toml\b", t)
    )
    if toml_out:
        if re.search(r"(json|yaml|yml|xml|csv|toml)\s+to\s+toml", t):
            return "Other"
        return "Text to TOML"

    yaml_out = (
        "output yaml" in t or
        "return only yaml" in t or
        re.search(r"\bto\s+yaml\b", t) or
        re.search(r"\binto\s+yaml\b", t)
    )
    if yaml_out:
        if re.search(r"(json|yaml|yml|xml|csv|toml)\s+to\s+yaml", t):
            return "Other"
        return "Text to YAML"

    json_out = (
        "output json" in t or
        "return only json" in t or
        re.search(r"\bto\s+json\b", t) or
        re.search(r"\binto\s+json\b", t)
    )
    if json_out:
        if re.search(r"(json|yaml|yml|xml|csv|toml)\s+to\s+json", t):
            return "Other"
        return "Text to JSON"

    return "Other"

def add_task_type(ds):
    return ds.map(lambda ex: {"task_type": detect_task_type(ex)})

# ------------------------------------------------------------
# 3) Assistant output cleaner (MOST IMPORTANT)
# ------------------------------------------------------------
def clean_assistant_output(ex):
    new_msgs = []
    for m in ex.get("messages", []):
        if m.get("role") != "assistant":
            new_msgs.append(m)
            continue

        txt = str(m.get("content", "")).strip()

        # Keep only after "Output:"
        if re.search(r"\boutput\s*:\s*", txt, re.I):
            txt = re.split(r"\boutput\s*:\s*", txt, flags=re.I)[-1].strip()

        # Remove leading explanations
        txt = re.sub(
            r"^(approach|analysis|steps|reasoning|explanation)\s*:.*?\n",
            "",
            txt,
            flags=re.I | re.S
        ).strip()

        # HARD FILTER: must start with structured format
        if not txt or txt[0] not in "{[<":
            return None

        new_msgs.append({"role": "assistant", "content": txt})

    ex["messages"] = new_msgs
    return ex

def clean_dataset(ds):
    cleaned = []
    for ex in ds:
        ex2 = clean_assistant_output(ex)
        if ex2 is not None:
            cleaned.append(ex2)
    return ds.select(range(len(cleaned))).from_list(cleaned)

# ------------------------------------------------------------
# 4) ONLY-output detector (Text tasks)
# ------------------------------------------------------------
def is_only_output_prompt(ex):
    t = _get_user_text(ex).lower()
    return bool(re.search(r"\b(return|output)\s+only\b", t))

# ------------------------------------------------------------
# 5) Downsample Text tasks in v5 (keep ONLY-output)
# ------------------------------------------------------------
def downsample_text_tasks(ds, target_types, keep_ratio=1/3):
    keep = []
    rng = random.Random(SEED)
    for i in range(len(ds)):
        ex = ds[i]
        if ex["task_type"] in target_types:
            if is_only_output_prompt(ex):
                keep.append(i)
            elif rng.random() < keep_ratio:
                keep.append(i)
        else:
            keep.append(i)
    return ds.select(keep)

# ------------------------------------------------------------
# 6) Load & clean datasets
# ------------------------------------------------------------
print("[LOAD] datasets...")
ds_v2 = load_dataset(V2, split="train")
ds_v4 = load_dataset(V4, split="train")
ds_v5 = load_dataset(V5, split="train")

print("[CLEAN] assistant outputs...")
ds_v2 = clean_dataset(ds_v2)
ds_v4 = clean_dataset(ds_v4)
ds_v5 = clean_dataset(ds_v5)

print("[TASK TYPE] detecting...")
ds_v2 = add_task_type(ds_v2)
ds_v4 = add_task_type(ds_v4)
ds_v5 = add_task_type(ds_v5)

# ------------------------------------------------------------
# 7) Downsample v5 noisy text tasks
# ------------------------------------------------------------
print("[DOWNSAMPLE] v5 Text to YAML / JSON...")
ds_v5_adj = downsample_text_tasks(
    ds_v5,
    {"Text to YAML", "Text to JSON"},
    keep_ratio=1/3
)

# ------------------------------------------------------------
# 8) Add from v2 / v4 (only high-value types)
# ------------------------------------------------------------
ADD_TYPES = {"Text to TOML", "CSV to JSON"}

ds_v2_add = ds_v2.filter(lambda ex: ex["task_type"] in ADD_TYPES)
ds_v4_add = ds_v4.filter(lambda ex: ex["task_type"] in ADD_TYPES)

# ------------------------------------------------------------
# 9) Merge & shuffle
# ------------------------------------------------------------
print("[MERGE] concatenate & shuffle...")
ds_mix = concatenate_datasets([ds_v5_adj, ds_v2_add, ds_v4_add])
ds_mix = ds_mix.shuffle(seed=SEED)

# ------------------------------------------------------------
# 10) Save jsonl (messages + metadata + task_type)
# ------------------------------------------------------------
keep_cols = [c for c in ["messages", "metadata", "task_type"] if c in ds_mix.column_names]
ds_out = ds_mix.remove_columns([c for c in ds_mix.column_names if c not in keep_cols])

print(f"[SAVE] -> {OUT_JSONL}")
ds_out.to_json(OUT_JSONL, orient="records", lines=True, force_ascii=False)

print("[DONE] Final task_type counts:")
for k, v in Counter(ds_out["task_type"]).most_common():
    print(f"{k:14s}: {v}")
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------

