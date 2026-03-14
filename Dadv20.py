# -*- coding: utf-8 -*-
"""
Spider -> DBBench形式(jsonl) 変換スクリプト（SQLite実行・観測(Observation)付き）

出力（1行1サンプル）例：
{
  "messages": [
    {"role":"system","content":"...DBBenchルール..."},
    {"role":"user","content":"...schema...\n\nQuestion: ..."},
    {"role":"assistant","content":"Action: Operation\nSELECT ..."},
    {"role":"tool","content":"Observation:\n..."},
    {"role":"assistant","content":"Action: Answer\n..."}
  ],
  "meta": {...}
}

前提：
- Hugging Face datasets の "spider" を使用（SpiderはAgentBench由来ではない）
- Spider付属の SQLite DB をローカルで実行して観測(Observation)を生成
- 変換した jsonl を SFT に投入（あなたのDBBench向け学習コードで読む想定）

使い方例：
  python spider_to_dbbench_jsonl.py \
    --split train \
    --out /content/dbbench_spider_train.jsonl \
    --max_rows 20 \
    --max_chars_cell 200 \
    --seed 3407

  python spider_to_dbbench_jsonl.py \
    --split validation \
    --out /content/dbbench_spider_val.jsonl
"""

import os
import re
import json
import argparse
import random
import sqlite3
from typing import Dict, Any, List, Tuple, Optional

from datasets import load_dataset, load_dataset_builder


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)


def safe_str(x, max_chars: int = 200) -> str:
    s = "" if x is None else str(x)
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return s


def walk_find_first(root: str, filename: str) -> Optional[str]:
    for d, _, files in os.walk(root):
        if filename in files:
            return os.path.join(d, filename)
    return None


def walk_find_db(root: str, db_id: str) -> Optional[str]:
    """
    SpiderのDBは多くの場合:
      .../database/<db_id>/<db_id>.sqlite
    または拡張子違いがあり得るので探索。
    """
    patterns = [
        os.path.join("database", db_id, f"{db_id}.sqlite"),
        os.path.join("database", db_id, f"{db_id}.db"),
        os.path.join(db_id, f"{db_id}.sqlite"),
        os.path.join(db_id, f"{db_id}.db"),
    ]
    for p in patterns:
        cand = os.path.join(root, p)
        if os.path.exists(cand):
            return cand

    # フォールバック：名前で探索
    for d, _, files in os.walk(root):
        for fn in files:
            if fn.lower() in (f"{db_id}.sqlite".lower(), f"{db_id}.db".lower()):
                return os.path.join(d, fn)
    return None


def normalize_ws(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip())


# -------------------------
# Schema handling (tables.json)
# -------------------------
def load_tables_json(dataset_root: str) -> List[Dict[str, Any]]:
    """
    Spiderのtables.jsonをキャッシュ内から見つけてロード。
    """
    path = walk_find_first(dataset_root, "tables.json")
    if not path:
        raise FileNotFoundError(
            "tables.json が見つかりません。datasetsキャッシュにSpiderが正しく展開されているか確認してください。"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_schema_index(tables: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {t["db_id"]: t for t in tables}


def format_schema_text(schema_obj: Dict[str, Any]) -> str:
    """
    DBBench向けに読みやすいスキーマ文字列を作る（簡易版）
    - テーブル一覧
    - 各テーブルの列（型）
    - 外部キー
    """
    # spider tables.json の代表的キー
    # table_names_original: List[str]
    # column_names_original: List[Tuple[int, str]]  (table_id, column_name) ; table_id=-1 は '*'
    # column_types: List[str]  (各columnに対応)
    # foreign_keys: List[Tuple[int,int]] (col_id, col_id)
    # primary_keys: List[int] (col_id)
    table_names = schema_obj.get("table_names_original", [])
    col_names = schema_obj.get("column_names_original", [])
    col_types = schema_obj.get("column_types", [])
    fks = schema_obj.get("foreign_keys", [])
    pks = set(schema_obj.get("primary_keys", []))

    # col_id -> (table_id, col_name, col_type)
    col_map = {}
    for i, ((tid, cname), ctype) in enumerate(zip(col_names, col_types)):
        col_map[i] = (tid, cname, ctype)

    # table_id -> list of col_id
    table_cols: Dict[int, List[int]] = {i: [] for i in range(len(table_names))}
    for cid, (tid, cname) in enumerate(col_names):
        if tid == -1:
            continue
        table_cols[tid].append(cid)

    lines = []
    lines.append("### Database schema")
    lines.append("Tables:")
    for tid, tname in enumerate(table_names):
        lines.append(f"- {tname}")

    lines.append("\nColumns:")
    for tid, tname in enumerate(table_names):
        lines.append(f"- {tname}:")
        for cid in table_cols.get(tid, []):
            _, cname, ctype = col_map[cid]
            pk_mark = " [PK]" if cid in pks else ""
            lines.append(f"  - {cname} ({ctype}){pk_mark}")

    if fks:
        lines.append("\nForeign keys:")
        for a, b in fks:
            ta, ca, _ = col_map[a]
            tb, cb, _ = col_map[b]
            # ta/tb は table_id
            if ta >= 0 and tb >= 0:
                lines.append(f"- {table_names[ta]}.{ca} -> {table_names[tb]}.{cb}")

    return "\n".join(lines)


# -------------------------
# SQLite execution + rendering
# -------------------------
def execute_sql(
    db_path: str,
    sql: str,
    max_rows: int = 20,
) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    """
    SELECT系を想定。Spider query は基本SELECTなのでここで十分。
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(max_rows)
        colnames = [d[0] for d in cur.description] if cur.description else []
        # sqlite3.Row -> tuple
        out_rows = [tuple(r) for r in rows]
        return colnames, out_rows
    finally:
        conn.close()


def render_observation(
    colnames: List[str],
    rows: List[Tuple[Any, ...]],
    max_chars_cell: int = 200,
) -> str:
    """
    DBBenchっぽく Observation をテキスト化（簡易）。
    - 1列1行の単一値なら値だけに近い形
    - それ以外はヘッダ+TSV風
    """
    if not colnames:
        return "Observation:\n<no columns>"

    if len(colnames) == 1 and len(rows) == 1:
        v = safe_str(rows[0][0], max_chars=max_chars_cell)
        return f"Observation:\n{v}"

    # TSV風
    lines = []
    lines.append("Observation:")
    lines.append("\t".join(colnames))
    for r in rows:
        lines.append("\t".join(safe_str(x, max_chars=max_chars_cell) for x in r))
    return "\n".join(lines)


def render_answer(
    colnames: List[str],
    rows: List[Tuple[Any, ...]],
    max_chars_cell: int = 200,
) -> str:
    """
    最終回答(Action: Answer)を簡易生成。
    DBBenchの採点系に完全一致する保証はないので、
    まずは「観測結果をそのまま返す」方針にする。
    """
    if not colnames:
        return "Action: Answer\n<no result>"

    if len(colnames) == 1 and len(rows) == 1:
        v = safe_str(rows[0][0], max_chars=max_chars_cell)
        return f"Action: Answer\n{v}"

    # JSON配列（列名付き）にして返す（汎用）
    arr = []
    for r in rows:
        obj = {}
        for c, v in zip(colnames, r):
            obj[c] = safe_str(v, max_chars=max_chars_cell)
        arr.append(obj)
    return "Action: Answer\n" + json.dumps(arr, ensure_ascii=False)


# -------------------------
# System prompt (DBBench style)
# -------------------------
def make_system_prompt() -> str:
    return (
        "You are a database agent.\n"
        "Follow these rules strictly:\n"
        "1) When you need to query the database, output exactly:\n"
        "   Action: Operation\n"
        "   <ONE SQL query>\n"
        "2) After you receive the observation, output exactly:\n"
        "   Action: Answer\n"
        "   <final answer>\n"
        "3) Do not include any other text.\n"
        "4) In each Operation, output only one SQL statement.\n"
    )


# -------------------------
# Dataset root discovery
# -------------------------
def get_spider_dataset_root() -> str:
    """
    datasets のキャッシュ内に展開された Spider リポジトリ（データファイル群）の場所を推定する。

    方法：
    - load_dataset_builder("spider") で download_and_prepare() を実行し、
      builder の cache_dir 近辺を探索対象にする。
    """
    builder = load_dataset_builder("spider")
    builder.download_and_prepare()
    # builder.cache_dir はユーザ環境で None のことがあるので複数候補
    roots = []
    if getattr(builder, "cache_dir", None):
        roots.append(builder.cache_dir)
    if getattr(builder, "_cache_dir", None):
        roots.append(builder._cache_dir)

    # datasets の標準キャッシュ
    roots.append(os.path.expanduser("~/.cache/huggingface/datasets"))

    # 最後に全候補を走査し、tables.json が存在する root を採用
    for r in roots:
        if not r or not os.path.exists(r):
            continue
        found = walk_find_first(r, "tables.json")
        if found:
            # tables.json があるディレクトリ階層の “少し上” をrootにしたいので、発見場所の上位へ戻る
            return os.path.dirname(found)

    raise FileNotFoundError("Spiderのキャッシュから tables.json を見つけられませんでした。")


# -------------------------
# Main conversion
# -------------------------
def convert_split(
    split: str,
    out_path: str,
    max_rows: int,
    max_chars_cell: int,
    seed: int,
    limit: int,
):
    set_seed(seed)

    print(f"[INFO] loading spider split={split}")
    ds = load_dataset("spider", split=split)

    # dataset_root = get_spider_dataset_root()
    dataset_root = "hf_spider_repo"
    print(f"[INFO] spider dataset root ~ {dataset_root}")

    tables = load_tables_json(dataset_root)
    schema_index = build_schema_index(tables)

    sys_prompt = make_system_prompt()

    n = len(ds)
    if limit > 0:
        n = min(n, limit)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    written = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as wf:
        for i in range(n):
            ex = ds[i]
            db_id = ex["db_id"]
            question = ex["question"]
            sql = ex["query"]

            # schema
            schema_obj = schema_index.get(db_id)
            if not schema_obj:
                skipped += 1
                continue
            schema_text = format_schema_text(schema_obj)

            # db file
            db_path = walk_find_db(dataset_root, db_id)
            if not db_path or not os.path.exists(db_path):
                skipped += 1
                continue

            # execute
            try:
                colnames, rows = execute_sql(db_path, sql, max_rows=max_rows)
            except Exception as e:
                # 実行不可SQLが稀にあり得る（環境差・sqlite方言など）
                skipped += 1
                continue

            obs = render_observation(colnames, rows, max_chars_cell=max_chars_cell)
            ans = render_answer(colnames, rows, max_chars_cell=max_chars_cell)

            # DBBench messages
            user_text = (
                f"{schema_text}\n\n"
                f"Question: {question}\n"
            )

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": "Action: Operation\n" + normalize_ws(sql)},
                # tool role が学習コード側で扱いにくい場合は role="user" に変えてもOK
                {"role": "tool", "content": obs},
                {"role": "assistant", "content": ans},
            ]

            out = {
                "messages": messages,
                "meta": {
                    "source": "spider",
                    "split": split,
                    "db_id": db_id,
                    "idx": i,
                    "sql": normalize_ws(sql),
                },
            }

            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

            if written % 500 == 0:
                print(f"[PROGRESS] written={written} skipped={skipped} / {i+1}")

    print(f"[DONE] out={out_path} written={written} skipped={skipped} total_seen={n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", choices=["train", "validation"])
    ap.add_argument("--out", type=str, required=True, help="Output jsonl path")
    ap.add_argument("--max_rows", type=int, default=20, help="Max rows to include in Observation")
    ap.add_argument("--max_chars_cell", type=int, default=200, help="Truncate each cell string to this length")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--limit", type=int, default=-1, help="For quick debug, limit number of samples (>0)")
    args = ap.parse_args()

    convert_split(
        split=args.split,
        out_path=args.out,
        max_rows=args.max_rows,
        max_chars_cell=args.max_chars_cell,
        seed=args.seed,
        limit=args.limit,
    )

# # -------------------------------------------------------
if __name__ == "__main__":
    convert_split(
        split="train",
        out_path="data/spider.jsonl",
        max_rows=20,
        max_chars_cell=200,
        seed=3407,
        limit=-1,
    )
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------

