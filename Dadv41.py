#　-*- coding: utf-8 -*-
# import json

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATASET = "./data/sft_alfworld_v4.jsonl"  # 適宜変更

GOAL_RE = re.compile(r"Your task is to:\s*(.*?)(?:\n|$)", re.IGNORECASE)

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] JSON decode error at line {line_no}")
                continue

def extract_goal_from_messages(ex: dict) -> str | None:
    """messages の user content から 'Your task is to:' を探して抽出"""
    msgs = ex.get("messages") or []
    for m in msgs:
        if m.get("role") == "user":
            text = m.get("content", "")
            mo = GOAL_RE.search(text)
            if mo:
                return mo.group(1).strip()
    return None

# ここは必要なら拡張：description/goal から task_type を推定する簡易ルール
def infer_task_type_from_text(goal_or_desc: str) -> str:
    t = goal_or_desc.lower()
    # AlfWorldでよくあるパターン例（必要なら増やす）
    if "heat" in t:
        return "heat_and_place"
    if "cool" in t:
        return "cool_and_place"
    if "clean" in t:
        return "clean_and_place"
    if t.startswith("examine") or "examine " in t:
        return "examine"
    if t.startswith("look") or "look at" in t:
        return "look"
    if "put two" in t or "two " in t:
        return "multi_object_place"
    if "place" in t or "put " in t:
        return "pick_and_place"
    return "UNKNOWN"

cnt_task_type = Counter()
cnt_goal_text = Counter()
cnt_room = Counter()
cnt_difficulty = Counter()

total = 0
fallback_used = 0

for ex in iter_jsonl(DATASET):
    total += 1
    md = ex.get("metadata") or {}
    task_type = md.get("task_type")

    desc = md.get("description")
    goal = extract_goal_from_messages(ex)

    # task_type が無い場合のフォールバック
    if not task_type:
        fallback_used += 1
        base = desc or goal or ""
        task_type = infer_task_type_from_text(base) if base else "UNKNOWN"

    cnt_task_type[task_type] += 1

    if desc:
        cnt_goal_text[desc] += 1
    elif goal:
        cnt_goal_text[goal] += 1
    else:
        cnt_goal_text["(NO_GOAL_FOUND)"] += 1

    if md.get("room_type"):
        cnt_room[md["room_type"]] += 1
    if md.get("difficulty"):
        cnt_difficulty[md["difficulty"]] += 1

print(f"Total examples: {total}")
print(f"Fallback used (task_type missing): {fallback_used} ({fallback_used/total*100:.1f}%)")

print("\n==== task_type distribution ====")
for k, v in cnt_task_type.most_common():
    print(f"{k:22s} {v:7d}  ({v/total*100:5.1f}%)")

print("\n==== room_type distribution (if metadata exists) ====")
for k, v in cnt_room.most_common():
    print(f"{k:12s} {v:7d}  ({v/total*100:5.1f}%)")

print("\n==== difficulty distribution (if metadata exists) ====")
for k, v in cnt_difficulty.most_common():
    print(f"{k:12s} {v:7d}  ({v/total*100:5.1f}%)")

print("\n==== top-20 goals/descriptions ====")
for k, v in cnt_goal_text.most_common(20):
    print(f"{v:5d}  {k}")
# # -------------------------------------------------------
import csv

# 例：task_type を CSV に保存
with open("./data/task_type_distribution.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["task_type", "count"])
    for k, v in cnt_task_type.most_common():
        w.writerow([k, v])

print("Saved: task_type_distribution.csv")
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------

