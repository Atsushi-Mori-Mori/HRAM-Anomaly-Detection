# 🚀【HRAM】実践で学ぶ! 機械学習を活用したデータサイエンス -異常検知偏-
<img src="docs/images/AN01.jpg" alt="AI agents" width="640">

〜プレス機械故障検知システムの開発検証〜<br>
LLMが言語処理能力を活用しインタラクティブに環境を改善する指示を正確に
出力できるかを競うコンペ。AI Agentsの基礎となる技術を関わるコンペである。<br>
目次<br>
1. 講義概要
2. 私のスコア
3. 異常検知の取組み<br>
3.1. 取組み概要<br>
3.2. 使用モデル<br>
3.3. データセット<br>
3.4. 作成プログラム<br>
3.5. スコア改善<br>
3.6. 所感<br>
4. 全体を通して<br>

## 1. コンペ概要
松尾研大規模言語モデル講座応用編の最終課題のコンペであり、メインと
アドバンスコンペの2トラックから構成される。<br>
- メインコンペ
Struct Evalというベンチマークを用いて、LLMが正確な構造化された
出力を生成できるかを競うコンペ。例えば、「これをjson形式で出力」とか
「json形式をyaml形式に変換」という指示に対応。<br>
- アドバンスコンペ
AgentBenchというベンチマーク(8つの多様なインタラクティブタスク)のうち
DB BenchとALFWorldの2タスクを用いて、LLMが与えられた指示に対して行動し、
環境とインタラクションすることで行動を改善し、最終的に期待される応えを
得ることができるかというエージェント能力を測るコンペ。
DB Benchはテーブル情報と要求を受け取り適切なクエリを組立ててDBを操作するタスク。
ALFWorldは環境描写+タスク目標を受取り目標を達成する行動を出力するタスク。<br>

**評価指標**：
評価指標は事務局から明確に開示されない(以下は推測)。<br>
- メインコンペ：F1スコア
- アドバンスコンペ：DB BenchとALFWorldの正解率から10点満点換算。

## 2. 私の順位・スコア
- **メインコンペ・スコア**：0.8093
- **メインコンペ予選通過順位**：17位 / 1000人程度
- **アドバンスコンペ・スコア**：4.4867
- **アドバンスコンペ最終順位**：48位 / 180人

ハイパーパラメータの調整とデータセット選択及び改善が
スコア向上の鍵であったと考える<br>

## 3. メインコンペ取組み
### 3.1 取組み概要
全体の流れは以下です。<br>
- 未学習データでモデルの実力を確認
- ハイパーパラメータの調整
- SFT+DPO学習
- 分布の異なるデータセットの組合せ
- タスク変換形式のデータ割合の調整

<事務局提供の標準コード><br>
- (1)2026最終課題メインコンペ_標準コード1(SFT).ipynb
- (2)2026最終課題予選_標準コード3(DPO).ipynb
- (3)2026最終課題メインコンペ_標準コード2(提出JSON生成).ipynb

### 3.2 使用モデル
事務局が指定したコンペで使用可能なLLMモデルは以下の通り。<br>
Qwen/Qwen3-4B-Instruct-2507<br>
unsloth/Qwen3-4B-Instruct-2507<br>

### 3.3 データセット
事務局が指定したデータセットは以下の通り。これらデータセットは
Hugging Faceからロード可能。<br>
<11_SFT向けデータセット><br>
u-10bei/structured_data_with_cot_dataset<br>
u-10bei/structured_data_with_cot_dataset_v2<br>
u-10bei/structured_data_with_cot_dataset_512<br>
u-10bei/structured_data_with_cot_dataset_512_v2<br>
u-10bei/structured_data_with_cot_dataset_512_v4<br>
u-10bei/structured_data_with_cot_dataset_512_v5<br>
daichira/structured-3k-mix-sft<br>
daichira/structured-5k-mix-sft<br>
<21_DPO向けデータセット><br>
u-10bei/dpo-dataset-qwen-cot<br>

### 3.4 解析プログラム
<11_データセットダウンロード><br>
Dllm10.py：SFTデータセットのダウンロード<br>
Dllm11.py：DPOデータセットのダウンロード<br>
<21_データセット改善><br>
Dllm21.py：u-10bei/structured_data_with_cot_dataset_512_v5のデータ解析し、"Text to TOML"を3倍、"Text to TOML"を1/3<br>
Dllm22.py：u-10bei/structured_data_with_cot_dataset_512_v2,v4のデータを、u-10bei/structured_data_with_cot_dataset_512_v5へ追加<br>
Dllm24.py：3データセット版<br>

### 3.5 スコア改善
未学習モデルでのスコアが既に0.6918であり、コンペのクライテリア(合格閾値)の
0.7にあとわずかという状況から開始。しかし、標準コード1(SFT)でデータセットv2
のみで学習するとスコアが0.6804へ劣化。一方、標準コード3(DPO)でそのまま
学習するとスコア0.7110で閾値クリアであった。<br>
これを踏まえてまずはSFTでスコアを上げれるところまで上げてその後DPOを実行
すればスコア向上が見込めるという仮説を設定。このSFT実行がなかなかスコアが
上がらないという長い道のりに突入した。すなわち、v2だけでなくv4,v5を追加したり、
ハイパーパラメータを調整したりと試行したがスコアが0.7を下回る状況が継続。
データセットとハイパーパラメータを同時に変更することで多少混乱が生じたかも。
またstructured_data_with_cot_dataset_512_v4とstructured-3k-mix-sfを混ぜた
データセットで試行するもスコアが0.62へ劇下がりとなる。学習分布が異なる
データセットの混在は悪影響を及ぼすようだ。<br>
評価データpublic_150.jsonを解析すると形式変換の割合が以下のように出た。<br>

```bash
 "Text to TOML", 25%
 "CSV to JSON",	14%
 "JSON to YAML", 14%
 "XML to JSON",	13%
  ･･･
 "CSV to XML",	3%
 "Text to JSON", 2%
 "Text to YAML", 1%
```

この解析結果に基づき、"Text to TOML", "CSV to JSON"のデータを4倍にし、
"Text to YAML"等のデータを1/3にする処置を適用。このタスク変換形式の
データ割合の調整によりスコアが0.8を超えて大きく向上。その後割合の
調整をいろいろと行ったが大きくなスコア向上は見られなかったが、
最終的に予選を17位で通過することができた。<br>
また今回最初はGoogle Colabo無償版のT4で実行していたがマシンパワーを
確保するためにPro契約を行いA100を使用開始した。しかし3日程度でProの
上限に達したためPro+契約へ更新した(Pro+契約がPro契約の開始日からとなり
Pro契約分は返金される)。予選終了まで1週間程度と短い時間ではあったが
マシンパワーは確保された。

👉 ハイパーパラメータの調整が意外とスコアアップに直結するようだ。<br>

### 3.6 メインコンペ所感
指定された出力形式(JSON,YAML,TOML,XML,CSV)への変換は正しい構造を安定して
生成する能力でありAgent機能の基本と考える。通常LLMはハルシネーションが
生じるリスクがあるが、それを抑制しつつ5種類の形式への変換を行わせる
という汎化性能を達成するために何をすべきか勉強になった。<br>

## 4. アドバンスコンペ取組み
### 4.1 取組み概要
全体の流れは以下です。<br>
- DBBench学習データセットの先行検討
- ハイパーパラメータの調整
- DBBench+ALFWorldの混合学習
- ALFWorldのデータ割合の調整

<事務局提供の標準コード><br>
- (1)2025最終課題アドバンスドコンペ標準コード_SFT_v_1_1.ipynb

### 4.2 使用モデル
事務局が指定したコンペで使用可能なLLMモデルは以下の通り。<br>
Qwen/Qwen2.5-7B-Instruct<br>
Qwen/Qwen3-4B-Instruct-2507<br>

合成データ作成用LLMホワイトモデル(事務局指定)<br>
LLMを使用して合成データ手法によりデータセット作成が許可されたが、
chatGPTやGemini等のクローズドモデル使用は禁止され、事務局指定の
ホワイトモデルのみが使用可能であった。

👉 マシンパワーと時間がなく今回合成データは作成していない。<br>

### 4.3 データセット
事務局が指定したデータセットは以下の通り。<br>
<11_DBBench向けデータセット><br>
u-10bei/dbbench_sft_dataset_react<br>
u-10bei/dbbench_sft_dataset_react_v2<br>
u-10bei/dbbench_sft_dataset_react_v3<br>
u-10bei/dbbench_sft_dataset_react_v4<br>
<21_ALFWorld向けデータセット><br>
u-10bei/sft_alfworld_trajectory_dataset<br>
u-10bei/sft_alfworld_trajectory_dataset_v2<br>
u-10bei/sft_alfworld_trajectory_dataset_v3<br>
u-10bei/sft_alfworld_trajectory_dataset_v4<br>
u-10bei/sft_alfworld_trajectory_dataset_v5<br>

### 4.4 解析プログラム
<11_データセットダウンロード><br>
Dadv11.py：spider.jsonlのダウンロード<br>
Dadv12.py：事務局データセットのダウンロード<br>
<21_データセット改善><br>
Dadv20.py：spider.jsonlをDBBench形式へ変換<br>
Dadv21.py：spider.jsonl内の"tool"を"user"へ変更<br>
Dadv32.py：DBBenchのINSERT強化用データセット生成<br>
Dadv41.py：AlfWorldデータセットsft_alfworld_v4.jsonlの処理毎のデータ数確認<br>
Dadv42.py：AlfWorldデータセットの正解率の高い処理のデータを削減<br>
Dadv42a.py：Dadv42.pyに加えて、正解率の低い処理のデータを増強<br>

### 4.5 スコア改善
未学習モデルでの投稿はルール上不可(投稿回数の制限)であったため事前に
モデルの実力確認ができない状況であった。<br>
事務局からの標準コードとデータセット開示がコンペ開始から1週間後であったため
先行して独自にデータセット調査とメインコンペの標準コード流用で進めた。
DBBenchは指示内容のSQLを作成しそれでDBを検索して解答を得るのが目的であり、
指示からSQLを作成するための学習データセットを調査するとKaggleサイトに
Yale University製作のtext-to-SQLのspider.jsonlがあることが判明。しかも
DBもあるのでこのデータセットからDBBench向けのデータセットをルールベースで
作成して実行したところスコアは3.6067であった。DBBenchの正解率は0.53から
0.57へ向上したがALFWorldの正解率が0.26から0.38とあまり向上しなかった。<br>
ハイパーパラメータは標準コードをベースにメインコンペの値を参考に決定。
それほど大きく変更して調整することはしなかった。<br>
最もスコアが向上したのはALFWorld向けデータセットv2,v3,v4,v5を使用した学習であり
スコアは4.4866であった。DBBenchの正解率が0.53から0.48へ低下したがALFWorld
の正解率が0.26から0.68へ大きく向上した。この結果に対して追加でDBBench向け
データセットによる追加学習を行っても少々スコア低下が確認された。<br>
ALFWorldの実行結果を解析し、cool/warm処理+対象物の移動等や、2物体処理の
ケースの正解率が低かった。<br>
メインコンペの途中契約であったGoogle Colabo Pro+でL4使用でアドバンスコンペは
最後まで完走することができた。但しL4では4Bモデルは学習可能であったが、
7Bモデル学習ではメモリ不足でフェイルするケースがあった。7Bモデルよりも
4Bモデルの方がスコアが良かったので4Bモデルでの学習を優先した。<br>
以下に4B/7Bのモデルの比較と選択のポイントを記載する。

|特徴	|Qwen/Qwen2.5-7B-Instruct	|Qwen/Qwen3-4B-Instruct-2507|
|:--------|:--------|:--------|
|パラメータ数	|7B (約70億)	|4B (約40億)|
|推論能力	|一般的にパラメータ数が多い方が複雑な推論（SQL生成や行動計画）に有利な傾向がある。	|最新世代のアーキテクチャだが、規模が小さいため、7Bに比べると複雑な指示への理解力が及ばない可能性がある。|
|推論速度	|4Bに比べると遅くなり、採点時間（2時間20分）の制限に注意が必要である。	|高速であり、試行錯誤（自己修正）を多く繰り返す設定にする場合、時間に余裕が持てる。|
|メモリ（VRAM）	|L4 GPU (24GB) で動作するが、4Bより多くのリソースを消費する。	|消費電力が少なく、リソースに余裕がある。量子化なしでも高速に動作し易い。|

👉 合成データによるデータセット増強がスコア改善の肝かも知れない。<br>

### 4.6 アドバンスコンペ所感
DBBenchとALFWorldという異なるタスク処理のLLMモデル開発であり、
分布の異なるデータセットによる学習で汎化性能を高めるのに苦労した。
メインコンペと比較してハイパーパラメータによる調整がスコア向上に
あまり寄与しなかった。やはり合成データによる学習を実施する必要が
あるのだろうと推測する。

### 5. 全体を通して
今回4B/7BクラスのSMLを使用してのAIエージェント機能モデル開発であったが、
ハイパーパラメータや入力データセットによりスコアが大きく変動し、
調整や再現性という観点でLLMモデル開発は難しい面があると感じた。


## 動作環境(Execution Environment)
- Windows / Python(Anaconda等)など

## 基本的な使い方(Basic Usage)

## 出力ファイルと保存先(Output Files and Storage)

## フォルダ構成(Folder Structure)

## ファイルサイズ(File Size)

## 関連リンク(Related Links)
AGENTBENCH: EVALUATING LLMS AS AGENTS<br>
https://arxiv.org/abs/2308.03688<br>
spider.jsonlのKaggleサイトからのダウンロード先<br>
https://www.kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset?resource=download<br>

## 注意事項(Notes)
None


