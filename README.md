# 多语种 SciEventKG + CL-TDR 复现指南

多语种科研事件时序知识图谱构建、跨语对齐与时序扩散推理的端到端可复现代码。流程覆盖 OpenAlex/arXiv 元数据获取、LLM 事件抽取、KG 构建、跨语对齐、CL-TDR 训练及前沿检测。所有脚本会打印 wall-clock 与峰值内存（RSS）以便记录运行开销。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 智普清言: export ZHIPUAI_API_KEY=sk-...
# OpenAI 可选: export OPENAI_API_KEY=sk-...
```

> 如果 pip 拉不到对应的 PyTorch 轮子，请先安装官方提供的版本（示例 CPU）：  
> `pip install torch==2.9.1` （或按照 https://pytorch.org 指南选择与你平台/加速器匹配的指令），再执行 `pip install -r requirements.txt`。
> 如 faiss-cpu 版本不可用，可改为 `pip install faiss-cpu==1.11.0`（已默认设置）；networkx 如 3.3 拉取失败，可使用 `networkx==3.2.1`。

执行顺序（若无真实数据，脚本会自动生成 dummy 数据跑通全链路）：

1) **下载元数据 / 生成样例**  
   ```bash
   python src/00_download_openalex.py --concept "AI for Science"
   ```
   按 `config.yaml` 设定的概念/时间/语言抓取；下载失败自动生成模拟数据到 `data/raw/*.jsonl`。

2) **语料标准化**  
   ```bash
   python src/01_build_corpus.py
   ```
   输出 `data/raw/corpus/corpus_{lang}.jsonl`（title/abstract/year/lang/authors/institutions）。

3) **LLM 事件抽取（默认智普清言）**  
   ```bash
   python src/02_llm_event_extract.py --provider zhipu   # 可选 openai / local
   ```
   产出 `data/extracted/events_{lang}.jsonl`。API 失败或缺 key 会自动回退启发式抽取，保证可运行。

4) **构建时序 KG**  
   ```bash
   python src/03_build_tkg.py
   ```
   生成 `data/kg/triples.tsv`（事件-方法/数据/发现/应用 + propose/use/support/apply），并合成 cite_diffuse/improve 边用于训练。

5) **跨语对齐 & 蒸馏对齐对**  
   ```bash
   python src/04_align_kd.py --model intfloat/multilingual-e5-base   # 可切换 e5-large / LaBSE
   ```
   写出 `data/kg/align_pairs.tsv`；可用 `--eval-file gold.tsv` 评估对齐 P/R/F1。

6) **训练 CL-TDR（支持蒸馏/对齐消融）**  
   ```bash
   python src/05_train_cltdr.py [--no-kd] [--no-align]
   ```
   HistoryEncoder(GAT+GRU) + cross-lingual diffusion head + bilinear scorer；损失含 TKGR、对齐、KD。

7) **前沿检测与趋势**  
   ```bash
   python src/06_frontier_detect.py --gold-frontier path/to/gold.tsv  # 可选
   ```
   计算 burst_z、结构新颖度、CBC、DFA，输出年度前沿列表；如提供 gold TSV(year\tevent) 还会给出 P/R/F1、LLI/CBC 统计。

## 配置说明
- 修改 `config.yaml` 可调概念、时间范围、语言、LLM 提供方（zhipu/openai/local）、嵌入模型与训练超参。
- `embedding.candidates` 给出推荐对齐模型候选。
- 训练设备由 `train.device` 决定（auto/cpu/cuda），无 CUDA 时自动退回 CPU。

## 数据目录
```
data/
  raw/            # OpenAlex/arXiv 原始或模拟 jsonl
  raw/corpus/     # 规范化 corpus_{lang}.jsonl
  extracted/      # LLM 事件抽取结果
  kg/             # triples.tsv, align_pairs.tsv, frontiers.json
  splits/         # 如需缓存数据划分可自定义放置
```

## 备注
- 智普清言需 `ZHIPUAI_API_KEY`，OpenAI 需 `OPENAI_API_KEY`；local 模式无需外网/显卡即可跑通。
- 若要真实实验，替换 `data/raw/openalex_{lang}.jsonl` 为下载数据后重复执行 01→06。
- 所有脚本末尾会输出 `[stats] {'seconds':..., 'max_rss_mb':...}` 便于记录性能。***
# mtscieventkg
