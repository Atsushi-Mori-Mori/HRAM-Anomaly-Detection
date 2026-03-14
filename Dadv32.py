#　-*- coding: utf-8 -*-
# import json

# データセット増強用Pythonコード
# # -------------------------------------------------------
# 強化のポイントと根拠
# 長期的な推論能力の付与: ソース では、エージェントの失敗原因として「長期的な推論能力の欠如」が
# 挙げられています。上記のコードで INSERT を多ターン化（SELECT確認ステップの追加）することで、
# モデルに**「現状を確認してから行動する」というエージェントらしい思考パターン**を学習させることができます。
# フォーマットエラーの防止: Databaseタスクでは「Invalid Format（形式エラー）」が頻発します。
# 増強データでは、Action: Operation と sql ...  の形式、および Final Answer: [...] の
# 出力を全サンプルで徹底させることで、推論時のフォーマット遵守率を高めます。
# 予約語と特殊文字への対応: ユーザー様の推論結果でも見られたように、Rank や Year といった
# MySQLの予約語や、改行を含むカラム名（Area\n(km²)）が原因で構文エラー（1064エラー）が発生します。
# 増強コードでは、これらをバックティック（`）で囲むパターンを学習させることで、エラーリカバリー能力を強化しています。
# GPT-4の特性を模倣: ソース の分析によると、GPT-4は特に INSERT 操作において高い性能を
# 示しています。この「慎重かつ正確な手順」をSFTデータに組み込むことで、4Bクラスの小型モデルでも
# エージェントとしての成功率を底上げすることが期待できます。
# この処理コードを実行して生成された dbbench_sft_augmented.txt を用いてSFT学習を継続することで、
# 特に正解率の低かったカテゴリの改善が見込めます。
# # -------------------------------------------------------
import json
import re

input_file = './data/dbbench_sft.jsonl'
output_file = './data/dbbench_sft_aug.jsonl'

# ----------------------------
# Robust extractors
# ----------------------------
def find_instruction_message(item):
    """
    DBBenchのmessagesから「テーブル名とheadersが書かれている user メッセージ」を探す。
    典型:
      "... The name of this table is X, and the headers of this table are A,B,C."
    """
    msgs = item.get("messages", [])
    if not isinstance(msgs, list):
        return None

    # userメッセージのうち、table/headers を含むものを優先
    candidates = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        c = m.get("content") or ""
        if "The name of this table is" in c and "headers of this table are" in c:
            candidates.append(c)

    # 最後の候補（質問文の方）を採用しがちなので最後を返す
    if candidates:
        return candidates[-1]

    # フォールバック：userの最後
    user_msgs = [m.get("content") for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    return user_msgs[-1] if user_msgs else None


def extract_headers(instruction: str):
    """
    headersの抽出を頑強にする。
    - 末尾の '.' が無いケース
    - 改行を跨ぐケース
    """
    if not instruction:
        return None

    # DOTALLで改行を跨ぐ。末尾は '.' があってもなくてもOKにする。
    m = re.search(r"headers of this table are\s*(.+?)(?:\.\s*$|\s*$)", instruction, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None

    headers = m.group(1).strip()
    # 余計な改行やスペースを潰す
    headers = re.sub(r"\s+", " ", headers)
    return headers


def mysql_like_observation(text: str) -> str:
    """
    Observationを user の MySQL生出力風 [(...,)] に寄せる。
    ここでは augmentation 用ダミーなので最小限。
    """
    t = (text or "").strip()
    if t.startswith("[(") and t.endswith(")]"):
        return t
    return "[('dummy_val',)]"


def find_observation_message(item):
    """
    既存messagesから user出力(Observation相当)を探す。
    - 例: "[(6246.9,)]"
    - 例: "Query OK, 1 row affected"
    """
    msgs = item.get("messages", [])
    if not isinstance(msgs, list):
        return None
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            c = (m.get("content") or "").strip()
            if c.startswith("[(") and c.endswith(")]"):
                return c
            if "Query OK" in c or "row affected" in c or "Empty set" in c:
                return c
    return None


# # -------------------------------------------------------
augmented_data = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue

        item = json.loads(line)
        m_type = item.get("metadata", {}).get("type")
        table_name = item.get("metadata", {}).get("table_name")
        original_sql = item.get("metadata", {}).get("sql")

        # --- 1. INSERTタスクの強化 ---
        if m_type == "INSERT":
            instruction = find_instruction_message(item)
            if not instruction:
                # 変換不能ならそのまま通す（落とさない）
                augmented_data.append(item)
                continue

            headers = extract_headers(instruction)
            # headersが取れなくても LIMIT 1 の確認はできるので継続
            select_sql = f"SELECT * FROM `{table_name}` LIMIT 1;" if table_name else "SELECT 1;"

            new_messages = [
                # 重要: ここは item["messages"] (list) をそのまま入れると入れ子になって壊れます
                # 元コードの `item["messages"], # System Prompt` は誤りです。
                # 既存の先頭user(ルール文)だけ流用するのが安全。
                {"role": "user", "content": item["messages"][0]["content"]} if item.get("messages") else {"role": "user", "content": ""},
                {"role": "assistant", "content": "Ok."},
                {"role": "user", "content": instruction},

                # ターン1: 既存データ確認
                {"role": "assistant", "content":
                    "Action: Operation\n"
                    f"sql {select_sql.rstrip(';')};"
                },
                {"role": "user", "content": mysql_like_observation("[('dummy_val',)]")},

                # ターン2: INSERT実行
                {"role": "assistant", "content":
                    "Action: Operation\n"
                    f"sql {original_sql.rstrip(';')};"
                },
                {"role": "user", "content": mysql_like_observation("Query OK, 1 row affected")},

                # ターン3: 最終回答（INSERT系は none）
                {"role": "assistant", "content": "Action: Answer\nFinal Answer: [\"none\"]"},
            ]

            item["messages"] = new_messages
            item["id"] = item.get("id", "unknown") + "_aug_insert"

        # --- 2. MAXタスクの強化 ---
        elif m_type == "aggregation-MAX":
            instruction = find_instruction_message(item)
            if not instruction:
                augmented_data.append(item)
                continue

            # 予約語っぽいものをバックティック
            if original_sql:
                for w in ["Rank", "Year", "Level", "Status"]:
                    # 単語境界で置換（Ranked を壊さない）
                    original_sql = re.sub(rf"\b{re.escape(w)}\b", f"`{w}`", original_sql)

            obs = find_observation_message(item) or "[(0,)]"  # なければダミー

            new_messages = [
                {"role": "user", "content": item["messages"][0]["content"]} if item.get("messages") else {"role": "user", "content": ""},
                {"role": "assistant", "content": "Ok."},
                {"role": "user", "content": instruction},

                {"role": "assistant", "content":
                    "Action: Operation\n"
                    f"sql {original_sql.rstrip(';')};"
                },
                {"role": "user", "content": mysql_like_observation(obs)},

                # 既存ラベルを使う（metadata.labelがあるのでそれでFinal Answerを作る方が堅い）
                {"role": "assistant", "content": "Action: Answer\nFinal Answer: " + json.dumps(item.get("metadata", {}).get("label", [""]), ensure_ascii=False)},
            ]

            item["messages"] = new_messages
            item["id"] = item.get("id", "unknown") + "_aug_max"

        augmented_data.append(item)

# 結果の書き出し
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in augmented_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("saved:", output_file)
print("rows:", len(augmented_data))
# # -------------------------------------------------------