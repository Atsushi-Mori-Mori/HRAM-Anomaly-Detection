#　-*- coding: utf-8 -*-
# import json

import json
import random
from collections import Counter, defaultdict

# =========================
# Config
# =========================
# INPUT_PATH = "./data/sft_alfworld.jsonl"                  # 元データ（フルデータに変更）
# OUTPUT_PATH = "./data/sft_alfworld_rebalanced_os2.jsonl"   # 出力
INPUT_PATH = "./data/sft_alfworld_v3.jsonl"                  # 元データ（フルデータに変更）
OUTPUT_PATH = "./data/sft_alfworld_v3_rebalanced_os2.jsonl"   # 出力
SEED = 3407
random.seed(SEED)

# 100%残す（下振れを防ぐ）
KEEP_FULL = {"pick_two", "heat_and_place", "cool_and_place"}

# それ以外は間引き率 0.5
DOWNSAMPLE_RATIO_OTHERS = 0.5

# オーバーサンプリング倍率（単純複製）
# 「最終的に」この倍率になるよう、追加分を複製して足します
OVERSAMPLE_MULTIPLIER = {
    "pick_two": 1.5,
    "heat_and_place": 1.2,
    "cool_and_place": 1.2,
}

# =========================
# Utils
# =========================
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

def get_task_type(ex: dict) -> str:
    md = ex.get("metadata") or {}
    return md.get("task_type", "unknown")

def calc_extra_count(n: int, multiplier: float) -> int:
    """
    n件を multiplier倍にしたい。
    追加すべき件数 = round(n*(multiplier-1)) だとブレるので
    floor + 確率で端数を1足す方式にして再現性も確保。
    """
    if multiplier <= 1.0:
        return 0
    target = n * multiplier
    base = int(target)  # floor
    extra = base - n
    # 端数分
    frac = target - base
    if random.random() < frac:
        extra += 1
    return max(extra, 0)

# =========================
# 1) Read & first-stage rebalance (keep full + downsample others)
# =========================
kept_by_type = defaultdict(list)
orig_cnt = Counter()
kept_cnt_stage1 = Counter()
dropped_cnt_stage1 = Counter()

for ex in iter_jsonl(INPUT_PATH):
    t = get_task_type(ex)
    orig_cnt[t] += 1

    if t in KEEP_FULL:
        kept_by_type[t].append(ex)
        kept_cnt_stage1[t] += 1
    else:
        if random.random() < DOWNSAMPLE_RATIO_OTHERS:
            kept_by_type[t].append(ex)
            kept_cnt_stage1[t] += 1
        else:
            dropped_cnt_stage1[t] += 1

# =========================
# 2) Oversample by simple duplication (sample-with-replacement)
# =========================
added_cnt_os = Counter()

for t, mult in OVERSAMPLE_MULTIPLIER.items():
    base_list = kept_by_type.get(t, [])
    n = len(base_list)
    if n == 0:
        continue

    extra = calc_extra_count(n, mult)
    if extra <= 0:
        continue

    # 単純複製：既存からランダムに選んで追加（with replacement）
    for _ in range(extra):
        kept_by_type[t].append(random.choice(base_list))
    added_cnt_os[t] += extra

# =========================
# 3) Write output (shuffle optional)
# =========================
all_kept = []
for t, lst in kept_by_type.items():
    all_kept.extend(lst)

random.shuffle(all_kept)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for ex in all_kept:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# =========================
# 4) Print summary
# =========================
final_cnt = Counter(get_task_type(ex) for ex in all_kept)

print("==== Summary ====")
print(f"Input total : {sum(orig_cnt.values())}")
print(f"Output total: {len(all_kept)}")
print("")
print("---- by task_type (input -> stage1 kept / dropped -> oversample added -> final) ----")
all_types = sorted(set(orig_cnt.keys()) | set(kept_cnt_stage1.keys()) | set(final_cnt.keys()))
for t in all_types:
    print(
        f"{t:18s} "
        f"in={orig_cnt[t]:6d}  "
        f"kept1={kept_cnt_stage1[t]:6d}  "
        f"drop1={dropped_cnt_stage1[t]:6d}  "
        f"os+={added_cnt_os[t]:6d}  "
        f"final={final_cnt[t]:6d}"
    )

print("\nSaved:", OUTPUT_PATH)
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------

