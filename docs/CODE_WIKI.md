# DMR-ABSA Code Wiki

> **项目全称**：DMR-ABSA: Dual-Perspective Multi-Agent Data Synthesis for Low-Resource Aspect-Based Sentiment Analysis
> （原名 DS²-ABSA，开源整理时更名；目录已重组为 `evaluation/`、`visualization/` 等规范结构，本文中的模块级讲解仍然准确，仓库级结构以根目录 [README](../README.md) 为准）
>
> **项目定位**：面向低资源（少样本）方面情感分析（Aspect-Based Sentiment Analysis, ABSA）的双视角数据合成与多智能体质量控制框架。通过 LLM（DeepSeek）从属性锚定与实例重建两个视角合成训练数据，并经由 A→B→C 三智能体评估流水线与 RAG 去重模块提升数据质量。

---

## 目录

- [1. 项目整体架构](#1-项目整体架构)
  - [1.1 项目背景](#11-项目背景)
  - [1.2 核心思想](#12-核心思想)
  - [1.3 系统架构图](#13-系统架构图)
  - [1.4 数据流总览](#14-数据流总览)
- [2. 目录结构](#2-目录结构)
- [3. 主要模块职责](#3-主要模块职责)
- [4. 关键类与函数说明](#4-关键类与函数说明)
  - [4.1 LLM API 封装层](#41-llm-api-封装层)
  - [4.2 提示词生成层](#42-提示词生成层)
  - [4.3 数据提取层](#43-数据提取层)
  - [4.4 三模型评估流水线](#44-三模型评估流水线)
  - [4.5 RAG 检索去重模块](#45-rag-检索去重模块)
- [5. 依赖关系](#5-依赖关系)
  - [5.1 Python 依赖](#51-python-依赖)
  - [5.2 模块间依赖](#52-模块间依赖)
  - [5.3 外部服务依赖](#53-外部服务依赖)
- [6. 项目运行方式](#6-项目运行方式)
  - [6.1 环境准备](#61-环境准备)
  - [6.2 完整流程运行](#62-完整流程运行)
  - [6.3 单步运行命令](#63-单步运行命令)
- [7. 数据格式规范](#7-数据格式规范)
- [8. 配置与扩展](#8-配置与扩展)

---

## 1. 项目整体架构

### 1.1 项目背景

少样本 ABSA 任务面临训练数据匮乏的问题。本项目通过 LLM 合成大规模训练数据，并引入多阶段质量提升机制，生成高质量合成数据集，覆盖四个 ABSA 域：

| 域名 | 含义 | 数据集来源 |
|------|------|------------|
| `lap` | 笔记本评论 | Laptop14 |
| `res` | 餐厅评论 | Restaurant14 |
| `res15` | 餐厅评论 | Restaurant15 |
| `res16` | 餐厅评论 | Restaurant16 |

### 1.2 核心思想

项目采用 **"双流合成 + 多阶段精炼"** 框架：

1. **双流合成（Dual-Stream Synthesis）**
   - **Key-Point-Driven（关键点驱动，即论文 AGS）**：基于 AGS 属性候选库（`attribute_candidates/`，由 LLM 预生成）中的属性候选值（对象、方面、类别、观点、情感）组合生成提示词，输出标准化合成数据，存放于 `property-driven`。
   - **Instance-Driven（实例驱动）**：基于真实样本进行改写、组合、掩码重建生成多样化合成数据，存放于 `seed-driven`。

2. **标签精炼（Label Refinement）**
   - 通过 **三模型顺序评估流水线（A→B→C，论文称 PRUS）** 对合成数据进行质量评估与提升：
     - 模型 A：方面词标准化
     - 模型 B：情感极性评估
     - 模型 C：句子质量提升与最终决策
   - 通过 **RAG 检索去重** 进一步消除重复数据。

### 1.3 系统架构图

```
┌────────────────────────────────────────────────────────────────────┐
│                       原始 ABSA 数据集 (data/)                      │
│                  lap / res / res15 / res16 各域                     │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
  ┌──────────────────────┐  ┌──────────────────────────┐
  │  Key-Point-Driven    │  │   Instance-Driven        │
  │  (AGS attribute lib) │  │   (paraphrase/mask/comb) │
  │  utils_prompt/       │  │   utils_prompt/          │
  │  get_keypoint_prompts│  │   get_instance_prompts   │
  └──────────┬───────────┘  └────────────┬─────────────┘
             │                           │
             │  LLM (DeepSeek via        │  LLM (DeepSeek via
             │   QNAIGC API)             │   QNAIGC API)
             ▼                           ▼
  ┌──────────────────────┐  ┌──────────────────────────┐
  │  prompts/            │  │  prompts/                │
  │  keypoint_prompts/   │  │  instance_prompts/       │
  └──────────┬───────────┘  └────────────┬─────────────┘
             │                           │
             ▼                           ▼
  ┌──────────────────────────────────────────────────────┐
  │     process_absa_data.py (ABSADataProcessor)        │
  │   解析 prompt_gen / prompt_aug，提取 Sentence+Label  │
  └──────────────────────────────────────────────────────┘
             │                           │
             ▼                           ▼
  ┌──────────────────┐  ┌───────────────┐
  │  property-driven │  │  seed-driven  │
  │  (标准化合成数据) │  │  (精炼合成数据) │
  └────────┬─────────┘  └───────┬───────┘
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │      evaluate/  (PRUS 评估流水线)          │
        │  ┌────────────────────────────────────┐  │
        │  │ Model A: 方面词标准化              │  │
        │  └─────────────┬──────────────────────┘  │
        │                ▼                         │
        │  ┌────────────────────────────────────┐  │
        │  │ Model B: 情感极性评估              │  │
        │  └─────────────┬──────────────────────┘  │
        │                ▼                         │
        │  ┌────────────────────────────────────┐  │
        │  │ Model C: 质量提升 + 最终决策        │  │
        │  └─────────────┬──────────────────────┘  │
        └────────────────┼─────────────────────────┘
                         │ Decision=="save"
                         ▼
        ┌──────────────────────────────────────────┐
        │   evaluate/outputs/  (*_final_dataset)  │
        └────────────────────┬─────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────────────┐
        │    dual-constrained-dedup/ (语义去重)     │
        │   方面词集合完全匹配 + 句子相似度≥90%     │
        └────────────────────┬─────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────────────────┐
        │  dual-constrained-dedup/rag_outputs/ (*_RAG) │
        │           最终高质量数据集               │
        └──────────────────────────────────────────────┘
```

### 1.4 数据流总览

| 阶段 | 输入位置 | 处理脚本 | 输出位置 | 输出文件命名规则 |
|------|----------|----------|----------|------------------|
| 1. 提示词生成 | `data/{domain}/{split}_all.json` | `utils_prompt/get_keypoint_prompts.py` | `prompts/keypoint_prompts/` | `{domain}_{split}{num_shots}.json` |
| 1. 提示词生成 | `data/{domain}/{split}_all.json` | `utils_prompt/get_instance_prompts.py` | `prompts/instance_prompts/` | `{domain}_{split}{num_shots}.json` |
| 2. 数据提取 | `prompts/keypoint_prompts/*.json` | `process_absa_data.py` | `property-driven/{domain}/` | `direct_{原文件名}.json` |
| 2. 数据提取 | `prompts/instance_prompts/*.json` | `process_absa_data.py` | `seed-driven/{domain}/` | `direct_{原文件名}.json` |
| 3. PRUS 评估 (A→B→C) | `property-driven` & `seed-driven` | `evaluate/run_evaluation.py` 或 `evaluate/optimized_streaming.py` | `evaluate/outputs/` | `{dataset_name}_final_dataset.json` |
| 4. RAG 去重 | `evaluate/outputs/*.json` | `dual-constrained-dedup/rag_retrieval.py` | `dual-constrained-dedup/rag_outputs/` | `{原文件名}_RAG.json` |

---

## 2. 目录结构

```
DMR-ABSA-open/
├── README.md                           # 项目说明文档（英文）
├── LICENSE / requirements.txt / .env.example / .gitignore
├── run.sh                              # 全流程流水线入口（阶段 1→4）
│
├── scripts/                            # 流水线脚本
│   ├── process_absa_data.py            # 提示词文件解析器（核心入口）
│   ├── process_all_datasets.py         # 批量处理所有数据集
│   └── fix_labels_from_processed.py    # 标签修复工具
│
├── docs/                               # 详细文档
│   ├── CODE_WIKI.md                    # 本 Code Wiki（模块级详解）
│   ├── pipeline.md                     # 五阶段复现指南
│   ├── data_format.md                  # 各数据目录 JSON schema
│   └── README_optimized_streaming.md   # 流式评估器说明
│
├── attribute_candidates/       # AGS：LLM 预生成的属性候选值 (Attribute-Grounded Synthesis)
│   ├── __init__.py
│   ├── categories.py                   # 方面类别 (aspect categories)
│   ├── aspects.py                      # 方面词 (aspect terms)
│   ├── opinions.py                     # 观点词 (opinion terms)
│   ├── sentiments.py                   # 情感类型
│   └── objects.py                      # 主体对象（餐厅/笔记本描述）
│
├── data/                               # 原始 ABSA 数据集
│   ├── lap/  res/  res15/  res16/      # 四个域
│       ├── train_all.json  dev_all.json  test_all.json
│       ├── sample2_all.json            # 2% 少样本采样
│       └── sample5_all.json            # 5% 少样本采样
│
├── prompts/                            # 所有提示词模板
│   ├── keypoint_driven.py              # 关键点驱动提示词模板
│   ├── instance_driven.py              # 实例驱动提示词模板
│   ├── templates.py                    # 基线方法提示词 + 工具函数
│   ├── keypoint_prompts/               # 论文使用的 keypoint 提示词（含 LLM 响应）
│   └── instance_prompts/               # 论文使用的 instance 提示词（含 LLM 响应）
│
├── utils_prompt/                       # 提示词填充与 LLM 调用
│   ├── qnaigc_api.py                   # LLM API 封装（环境变量读密钥）
│   ├── get_keypoint_prompts.py         # 关键点驱动提示词生成
│   └── get_instance_prompts.py         # 实例驱动提示词生成
│
├── property-driven/                # 关键点驱动合成数据（已提取）
│   └── lap/  res/  res15/  res16/  → direct_{domain}_{split}{num_shots}.json
│
├── seed-driven/             # 实例驱动合成数据（已提取）
│   └── lap/  res/  res15/  res16/  → direct_{domain}_{split}{num_shots}.json
│
├── evaluation/                         # PRUS：多智能体 A→B→C 评估流水线（仅保留核心）
│   ├── README.md                       # 评估系统说明
│   ├── run_evaluation.py               # 批量评估入口
│   ├── optimized_streaming.py          # 流式处理器（推荐主入口）
│   ├── model_a_aspect_standardization.py # 智能体 A：方面词标准化
│   ├── model_b_sentiment_evaluation.py   # 智能体 B：情感极性评估
│   ├── model_c_quality_decision.py       # 智能体 C：质量决策
│   ├── data_utils.py                    # 数据工具函数
│   ├── concurrency_protection.py        # 并发保护器
│   └── file_lock.py                     # 文件锁
│
├── dual-constrained-dedup/                         # RAG 检索去重模块（独立）
│   ├── README.md                       # 模块说明
│   ├── config.py                       # RAG 模块配置
│   ├── rag_retrieval.py                # RAG 检索主程序
│   ├── explain_aspect_matching.py      # 方面匹配解释
│   ├── requirements.txt               # 模块依赖
│   ├── test_structure.py              # 结构自检
│   └── rag_outputs/                   # 示例 RAG 去重输出
│
└── visualization/                      # INSTRUCTOR + t-SNE 论文级可视化
    ├── README.md                       # 使用说明
    ├── plot_tsne.py                    # 主脚本（两级缓存，PNG+PDF）
    └── all.py                          # 批量渲染 8 个数据集
```

---

## 3. 主要模块职责

| 模块 | 路径 | 职责 |
|------|------|------|
| **AGS 属性候选库 (Attribute-Grounded Synthesis)** | `attribute_candidates/` | 存放由 LLM 预先生成的属性候选值（对象、方面类别、方面词、观点词、情感类型），供关键点驱动（AGS）提示词随机组合使用 |
| **原始数据集** | `data/` | 四个 ABSA 域的原始数据，含 `train/dev/test` 及 `2%/5%` 少样本子集。每条样本含 `ID`、`sentence`、`aspects`（带字符偏移 `from/to/target/polarity`） |
| **提示词模板** | `prompts/` | 定义关键点驱动、实例驱动、基线方法的提示词模板字符串 |
| **提示词生成** | `utils_prompt/` | 填充模板、调用 LLM、增量写入生成提示词文件 |
| **数据提取** | `scripts/process_absa_data.py` | 不调用 LLM，仅通过正则与 `ast.literal_eval` 解析 `prompt_gen`/`prompt_aug`，提取 `Sentence` 与 `Label` |
| **合成数据存储** | `property-driven/`、`seed-driven/` | 分别存放关键点驱动（标准化）与实例驱动（精炼）合成数据 |
| **多智能体评估流水线（PRUS）** | `evaluation/` | A→B→C 顺序处理，对合成数据进行方面词标准化、情感极性评估、质量决策，仅保存决策为 `save` 的高质量数据。论文中该流水线称 PRUS，其"最大精炼迭代次数 = 3"对应 A/B/C 三个精炼阶段 |
| **RAG 检索去重** | `dual-constrained-dedup/` | 独立模块，基于"方面词集合完全匹配 + 句子余弦相似度≥90%"双条件精准去重 |
| **可视化** | `visualization/` | INSTRUCTOR 嵌入 + t-SNE 论文级图表（散点 + 密度，PNG/PDF 双输出，两级缓存） |

---

## 4. 关键类与函数说明

### 4.1 LLM API 封装层

#### `QNAIGCDeepSeek` ([utils_prompt/qnaigc_api.py](utils_prompt/qnaigc_api.py))

DeepSeek LLM 的 API 封装类，所有需要调用 LLM 的模块都通过此类或工厂函数创建。

| 方法 | 签名 | 说明 |
|------|------|------|
| `__init__` | `(self)` | 从 `.env` 读取 `OPENAI_API_KEY`、`OPENAI_API_BASE`、`OPENAI_Model`，初始化 `OpenAI` 客户端 |
| `chat_completion` | `(messages, max_tokens=4096, temperature=0.3) -> str` | 调用 LLM，最多重试 5 次，使用指数退避（`2^attempt + random`）|
| `generate_sentence_from_prompt` | `(keypoint_prompt, max_tokens=4096, temperature=0.6) -> dict` | 根据 prompt 生成句子和标签，返回 `{success, sentence, label, full_output}` |
| `batch_generate` | `(prompts, ...) -> list` | 批量生成（带延迟） |

**工厂函数**：
- `create_qnaigc_client() -> QNAIGCDeepSeek`：模型 A/B/C 通过此函数创建客户端

**System Prompt 关键约束**：
```
You are a helpful assistant that generates review sentences...
Strictly use the exact aspect terms provided in the prompt; do NOT paraphrase or alter them.
```

---

### 4.2 提示词生成层

#### `get_keypoint_prompts(...)` ([utils_prompt/get_keypoint_prompts.py](utils_prompt/get_keypoint_prompts.py))

关键点驱动提示词生成主函数。

| 参数 | 类型 | 说明 |
|------|------|------|
| `domain` | str | 域名：`res`/`lap`/`res15`/`res16` |
| `dataset` | str | 数据集类型：`sample`/`train`/`dev`/`test` |
| `seed` | int | 随机种子（默认 42） |
| `num_shots` | int | 少样本数（2 或 5；非 sample 时传 0） |
| `num_samples` | int | 生成样本数（默认 4000） |
| `num_examples` | int | 每条 prompt 含示例数（默认 4） |

**处理流程**：
1. 加载 `data/{domain}/{split}_all.json`，筛选含 `aspects` 的样本作为示例池
2. 根据域名选择 `res_objects/res_ac/res_at/res_opinions` 或 `lap_*`
3. 循环 `num_samples` 次，每次随机选择 `object/aspect/category/opinion/sentiment` 组合
4. 使用 `generate_prompt` 模板填充，附加随机示例
5. 调用 `llm.generate_sentence_from_prompt` 生成句子+标签
6. **增量写入**：每条结果立即追加到输出文件（先写 `[`，逐条 `,` 分隔，最后 `]`）

**输出文件**：`prompts/keypoint_prompts/{domain}_{dataset}{num_shots}.json`

#### `get_instance_prompts(...)` ([utils_prompt/get_instance_prompts.py](utils_prompt/get_instance_prompts.py))

实例驱动提示词生成主函数，包含 4 种策略：

| 策略类型 | ID 后缀 | 说明 |
|----------|---------|------|
| `paraphrase_one` | `_{type}` | 单样本改写：基于 `paraphrase_prompt` 生成 4 条改写句 |
| `paraphrase_comb` | `_{type}` | 双样本组合：随机组合两个样本，基于 `combination_prompt` |
| `aspect_mask` | `_{type}_{window}` | 方面词掩码重建：使用 `mask_aspect_terms` 掩码方面词及窗口 |
| `context_mask` | `_{type}_{idx}` | 上下文掩码重建：使用 `mask_context` 随机掩码 60% 上下文 |

**关键辅助函数**：
- `mask_aspect_terms(sentence, aspects, window_size=2)`：使用 spacy 分词，将方面词及周围 `window_size` 个 token 替换为 `<mask>`，合并连续 mask
- `mask_context(sentence, aspects, sample_times=2)`：在保留方面词的前提下，随机选择 60% 长度的区间掩码

**输出文件**：`prompts/instance_prompts/{domain}_{dataset}{num_shots}.json`

---

### 4.3 数据提取层

#### `ABSADataProcessor` ([process_absa_data.py](../scripts/process_absa_data.py))

不调用 LLM，仅通过代码解析提示词文件中的 `prompt_gen`/`prompt_aug` 字段。

| 方法 | 说明 |
|------|------|
| `parse_keypoint_prompt_gen(prompt_gen)` | 解析 keypoint 格式：按 `llm_output:` 分割取句子，从基础 prompt 用正则提取最后的 `Label:` |
| `parse_instance_prompt_aug(prompt_aug)` | 解析 instance 格式：用正则匹配 `4 Diverse ... Sentences with Labels:\n\n1. Sentence:` 模式，提取第一句与标签 |
| `process_single_item(item, prompt_type)` | 处理单条记录，返回 `{ID, Sentence, Label}` |
| `extract_domain_from_filename(filename)` | 从文件名提取域名（`_` 前部分） |
| `process_file(input_file, output_base_dir, prompt_type, source_filename)` | 处理单个文件，支持断点续传（跳过已处理 ID），每 10 条保存一次中间结果 |
| `save_results(results, output_file)` | 按 ID 排序后写入 JSON |

**命令行入口**：
```bash
python scripts/process_absa_data.py <input_file> <output_base_dir> <source_filename> --prompt_type <keypoint|instance>
```

**输出格式**（去除 `aspects` 字段，仅保留 ID/Sentence/Label）：
```json
{
    "ID": "0",
    "Sentence": "The ingredient quality was disappointing...",
    "Label": [["ingredient quality", "negative"], ["cleanliness", "negative"]]
}
```

---

### 4.4 三模型评估流水线（PRUS）

> **术语对应**：论文中的 **PRUS** 即本模块的 A→B→C 评估流水线。论文设置"最大精炼迭代次数 = 3"，对应数据依次经过智能体 A（方面词标准化）→ 智能体 B（情感极性评估）→ 智能体 C（质量决策）共 3 次精炼（每阶段一次）。

#### 模型 A：`AspectStandardizationModel` ([evaluate/model_a_aspect_standardization.py](evaluation/model_a_aspect_standardization.py))

**角色**：数据清洗专家，专注于方面词标准化与改写处理。

**核心方法**：
| 方法 | 说明 |
|------|------|
| `process_single_entry(data_entry)` | 处理单条数据，遍历 `Label` 中每个 `[aspect, polarity]`，调用 `_standardize_aspect` |
| `_standardize_aspect(sentence, aspect, polarity)` | 调用 LLM，返回 `(标准化方面词, 状态)`；若返回 `REMOVE_ASPECT` 则返回 `(None, "removed")` |
| `process_batch(data_entries, delay)` | 批量处理，带 API 延迟 |

**System Prompt 关键原则**：
- 方面词必须存在于句子中
- 禁止重复方面词
- 最小修改，不重写
- 若句子中无合适方面词，使用 `REMOVE_ASPECT`

**输出格式**：
```json
{
    "ID": "001",
    "Sentence": "原始句子",
    "ProcessedLabels": [
        {"aspect": "标准化方面词", "polarity": "情感极性", "standardization_status": "modified/unchanged"}
    ],
    "OriginalLabels": [[...]]
}
```

#### 模型 B：`SentimentEvaluationModel` ([evaluate/model_b_sentiment_evaluation.py](evaluation/model_b_sentiment_evaluation.py))

**角色**：情感分析专家，专注于评估和纠正情感极性。

**核心方法**：
| 方法 | 说明 |
|------|------|
| `process_single_entry(data_entry)` | 接收模型 A 输出，遍历 `ProcessedLabels` 调用 `_evaluate_sentiment` |
| `_evaluate_sentiment(sentence, aspect, polarity, standardization_status)` | 若 `standardization_status == "modified"`，强调需重新评估；返回 `(纠正极性, 状态)` |

**情感规则**：
- `expensive/costly` → 通常 negative
- `cheap/affordable` → 通常 positive
- `fast/quick` → 通常 positive
- `slow/laggy` → 通常 negative

**输出格式**：在模型 A 输出基础上增加 `sentiment_status` 字段。

#### 模型 C：`QualityDecisionModel` ([evaluate/model_c_quality_decision.py](evaluation/model_c_quality_decision.py))

**角色**：句子优化与质量评估专家，决定是否保存数据。

**核心方法**：
| 方法 | 说明 |
|------|------|
| `process_single_entry(data_entry)` | 接收模型 B 输出，调用 `_evaluate_quality_and_decide` |
| `_evaluate_quality_and_decide(sentence, processed_labels, has_modifications)` | 返回 `{optimized_sentence, sentence_optimized, decision, quality_score, reasoning}` |
| `save_single_entry_realtime(result, output_path)` | **实时保存**：决策为 save 时立即追加到文件 |
| `initialize_incremental_file(output_path)` | 初始化增量文件，记录状态到 `_incremental_file_state` |
| `process_batch_with_realtime_save(data_entries, output_path, delay)` | 实时保存批处理（推荐） |
| `save_final_dataset(results, output_path)` | 批量模式：仅保存 `Decision=="save"` 的数据 |

**决策逻辑**：
- `save`：方面词准确、极性正确、句子清晰、非重复
- `discard`：内容空洞、重复、方面词不相关、极性矛盾

**最终输出格式**：
```json
{
    "ID": "001",
    "Sentence": "优化后句子",
    "Label": [["aspect", "polarity"]]
}
```

#### 流水线编排

##### `DataEvaluationPipeline` ([evaluate/run_evaluation.py](evaluation/run_evaluation.py))

批量评估流水线，整合 A→B→C。

| 方法 | 说明 |
|------|------|
| `__init__(delay, realtime_save)` | 初始化三模型，创建 `evaluate/outputs/` |
| `run_evaluation(data, save_intermediate, dataset_name)` | 验证→模型A→模型B→模型C（实时/批量），返回统计 |

**命令行参数**：
| 参数 | 默认 | 说明 |
|------|------|------|
| `--base-dirs` | `[property-driven, seed-driven]` | 数据目录 |
| `--subdirs` | `[lap, res, res5, res16]` | 子目录 |
| `--dataset-name` | `combined_dataset` | 数据集名（影响输出文件名） |
| `--delay` | `1.0` | API 延迟秒数 |
| `--no-intermediate` | - | 不保存中间结果 |
| `--validate-only` | - | 仅验证数据结构 |
| `--no-realtime-save` | - | 禁用实时保存，使用批量模式 |
| `--force-clean` | - | 清理无效锁文件 |
| `--check-locks` | - | 检查运行中的进程 |

##### `OptimizedStreamingPipeline` ([evaluate/optimized_streaming.py](evaluation/optimized_streaming.py))

**推荐使用的优化流式处理器**，特性：
- 按文件逐个处理
- 单条数据 A→B→C 流水线（不保存 A/B 中间结果）
- 每个文件 ID 从 0 开始增量
- 处理完一条立即追加保存
- 内存友好 O(m)

**输出文件命名**：`{数据集类型}_{原文件名}.json`（如 `norm_direct_res_sample2.json`）

#### 数据工具 `data_utils.py`

| 函数 | 说明 |
|------|------|
| `load_json_file(file_path)` | 多编码尝试（utf-8-sig/utf-8/gbk/gb2312）加载 JSON |
| `load_data_from_directories(base_dirs, subdirs)` | 从多目录合并加载数据 |
| `discover_data_files(base_dirs, subdirs)` | 发现 JSON 文件 |
| `validate_data_structure(data)` | 验证字段完整性，支持大小写两种字段名 |
| `get_data_statistics(data)` | 统计条目数、标签数、极性分布、Top 方面词 |
| `create_data_splits(data, ratios, seed)` | 训练/验证/测试集分割 |
| `save_data(data, output_path, indent)` | 增量写入模式保存 |

#### 并发保护 `concurrency_protection.py`

| 类/函数 | 说明 |
|---------|------|
| `ConcurrencyProtector` | 基于 PID 文件 + 文件锁的并发保护器 |
| `acquire_process_lock(process_name)` | 获取进程锁，检测孤立 PID 文件 |
| `check_running_evaluation_processes()` | 检查运行中的评估进程 |
| `force_clean_stale_locks()` | 强制清理无效锁 |

---

### 4.5 RAG 检索去重模块

#### `RAGRetrievalModule` ([dual-constrained-dedup/rag_retrieval.py](dual-constrained-dedup/rag_retrieval.py))

**独立模块**，基于双条件精准去重。

**去重条件（必须同时满足）**：
1. **方面词集合完全匹配**（仅比较 aspect，不比较 polarity）
2. **句子余弦相似度 ≥ 90%**（基于 `all-MiniLM-L6-v2` 嵌入）

| 方法 | 说明 |
|------|------|
| `extract_aspect_set(label)` | 从 `Label` 提取方面词集合（`set`） |
| `calculate_sentence_embedding(sentence)` | 计算句子嵌入向量 |
| `is_duplicate(sentence, aspect_set)` | 判断是否重复，返回 `(是否重复, 最高相似度)` |
| `add_to_cache(sentence, aspect_set)` | 添加到去重缓存 |
| `validate_data_format(data)` | 验证 `ID/Sentence/Label` 格式 |
| `process_single_file(file_path)` | 处理单文件，ID 从 0 重新编号，输出 `{原名}_RAG.json` |
| `get_target_files()` | 通过 `glob` 获取 `INPUT_DIR` 下所有 `.json` |
| `run()` | 主运行方法 |
| `generate_summary_log()` | 生成汇总日志 |

**配置**（[dual-constrained-dedup/config.py](dual-constrained-dedup/config.py)）：
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `INPUT_DIR` | - | 输入目录（需修改为实际路径） |
| `OUTPUT_DIR` | - | 输出目录 |
| `LOG_DIR` | - | 日志目录 |
| `SIMILARITY_THRESHOLD` | `0.9` | 相似度阈值 |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 嵌入模型路径 |
| `RECURSIVE_TRAVERSAL` | `False` | 是否递归遍历 |
| `TARGET_FILE_SUFFIX` | `.json` | 目标文件后缀 |
| `BATCH_SIZE` | `1000` | 批处理大小 |

---

## 5. 依赖关系

### 5.1 Python 依赖

**核心依赖**（见 `.venv` 已安装）：

| 依赖 | 用途 |
|------|------|
| `openai` | DeepSeek LLM API 调用（兼容 OpenAI SDK） |
| `python-dotenv` | 从 `.env` 加载环境变量 |
| `spacy` + `en_core_web_sm` | 分词、词性标注、掩码操作 |
| `tqdm` | 进度条 |
| `sentence-transformers` | RAG 模块句子嵌入（`all-MiniLM-L6-v2`） |
| `scikit-learn` | 余弦相似度计算 |
| `numpy` | 数值计算 |
| `psutil` | 并发保护，检测进程运行状态 |

**可视化依赖**（`visualization/`）：
- `InstructorEmbedding`、`matplotlib`、`seaborn`

**安装方式**：
```bash
pip install openai python-dotenv spacy tqdm
pip install sentence-transformers scikit-learn numpy psutil
python -m spacy download en_core_web_sm
# 或本地安装 whl
pip install en_core_web_sm-3.7.1-py3-none-any.whl
```

### 5.2 模块间依赖

```
attribute_candidates (AGS 属性候选库)
        │
        ├──► utils_prompt/get_keypoint_prompts.py
        │         │
        │         ├──► prompts/keypoint_driven.py (模板)
        │         ├──► prompts/templates.py (工具函数)
        │         └──► utils_prompt/qnaigc_api.py (LLM)
        │                  │
        │                  └──► .env (API 配置)
        │
        └──► utils_prompt/get_instance_prompts.py
                  │
                  ├──► prompts/instance_driven.py (模板)
                  ├──► prompts/templates.py (工具函数)
                  ├──► spacy (en_core_web_sm)
                  └──► utils_prompt/qnaigc_api.py (LLM)

process_absa_data.py (独立，无 LLM 依赖)
        │
        ├──► prompts/keypoint_prompts/*.json
        └──► prompts/instance_prompts/*.json

evaluate/model_a_aspect_standardization.py
        └──► utils_prompt/qnaigc_api.py (create_qnaigc_client)

evaluate/model_b_sentiment_evaluation.py
        └──► utils_prompt/qnaigc_api.py (create_qnaigc_client)

evaluate/model_c_quality_decision.py
        └──► utils_prompt/qnaigc_api.py (create_qnaigc_client)

evaluate/run_evaluation.py
        ├──► model_a / model_b / model_c
        ├──► data_utils.py
        └──► concurrency_protection.py → file_lock.py

dual-constrained-dedup/rag_retrieval.py (完全独立)
        └──► dual-constrained-dedup/config.py
```

### 5.3 外部服务依赖

| 服务 | 用途 | 配置项 |
|------|------|--------|
| **DeepSeek LLM API** | 句子生成、方面词标准化、情感评估、质量决策 | `OPENAI_API_KEY`、`OPENAI_API_BASE`、`OPENAI_Model` |
| **DeepSeek-V3** | 默认模型（见 `.env.example`，与论文一致） | `OPENAI_Model=DeepSeek-V3` |
| **all-MiniLM-L6-v2** | RAG 句子嵌入 | `dual-constrained-dedup/config.py` 中 `EMBEDDING_MODEL` |

**`.env` 文件示例**（参考 [.env.example](../.env.example)）：
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://llmapi.paratera.com
OPENAI_Model=DeepSeek-V3
```

---

## 6. 项目运行方式

### 6.1 环境准备

1. **Python 环境**：建议 Python 3.8+，项目自带 `.venv`（Python 3.12）
2. **安装依赖**：
   ```bash
   pip install -r dual-constrained-dedup/requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. **配置 `.env`**：在项目根目录创建 `.env`，填入 DeepSeek API 信息
4. **下载嵌入模型**（RAG 模块）：将 `all-MiniLM-L6-v2` 模型放置到 `dual-constrained-dedup/config.py` 指定路径

### 6.2 完整流程运行

项目提供 `run.sh` 作为全流程运行脚本（详见 [run.sh](run.sh)），包含四个阶段：

```bash
# 阶段 1：生成提示词（keypoint + instance）
python utils_prompt/get_keypoint_prompts.py --domain res --dataset sample --seed 42 --num_shots 2 --num_samples 4000 --num_examples 4
python utils_prompt/get_instance_prompts.py --domain res --dataset sample --seed 42 --num_shots 2

# 阶段 2：提取合成数据（不调用 LLM）
python scripts/process_absa_data.py prompts/keypoint_prompts/res_sample2.json property-driven res_sample2 --prompt_type keypoint
python scripts/process_absa_data.py prompts/instance_prompts/res_sample2.json seed-driven res_sample2 --prompt_type instance

# 阶段 3：PRUS 评估（A→B→C，推荐使用优化流式处理器）
python evaluate/optimized_streaming.py property-driven/res/direct_res_sample2.json
# 或批量处理：
python evaluate/run_evaluation.py --base-dirs property-driven --subdirs res --dataset-name norm_res --delay 2.0

# 阶段 4：RAG 去重
cd dual-constrained-dedup
python rag_retrieval.py
```

**一键批量处理**（生成所有提示词并提取所有数据）：
```bash
python scripts/process_all_datasets.py
```

### 6.3 单步运行命令

#### 生成 keypoint 提示词
```bash
python utils_prompt/get_keypoint_prompts.py \
    --domain res --dataset sample --seed 42 \
    --num_shots 2 --num_samples 4000 --num_examples 4
```

#### 生成 instance 提示词
```bash
python utils_prompt/get_instance_prompts.py \
    --domain lap --dataset sample --seed 42 --num_shots 5
```

#### 提取合成数据
```bash
# keypoint → property-driven
python scripts/process_absa_data.py \
    prompts/keypoint_prompts/res_sample2.json \
    property-driven \
    res_sample2 \
    --prompt_type keypoint

# instance → seed-driven
python scripts/process_absa_data.py \
    prompts/instance_prompts/res_sample2.json \
    seed-driven \
    res_sample2 \
    --prompt_type instance
```

#### 运行三模型评估

**批量模式**：
```bash
python evaluate/run_evaluation.py \
    --base-dirs property-driven seed-driven \
    --subdirs lap res res15 res16 \
    --dataset-name combined_dataset \
    --delay 2.0
```

**流式模式（推荐）**：
```bash
python evaluate/optimized_streaming.py \
    property-driven/res/direct_res_sample2.json
```

**仅验证数据**：
```bash
python evaluate/run_evaluation.py --validate-only
```

#### 运行 RAG 去重
```bash
cd dual-constrained-dedup
# 修改 config.py 中的 INPUT_DIR 指向 evaluation/outputs
python rag_retrieval.py
```

#### 可视化（可选）
```bash
cd visualization
python plot_tsne.py --data ../data/lap/sample5_all.json --output picture/lap_sample5.png --gpus 0
```

---

## 7. 数据格式规范

### 原始数据格式（`data/`）

```json
[
    {
        "ID": "1701",
        "sentence": "Winnie and her staff are the best crew you can find serving you.",
        "aspects": [
            {"from": "15", "to": "20", "target": "staff", "polarity": "positive"},
            {"from": "34", "to": "38", "target": "crew", "polarity": "positive"}
        ]
    }
]
```

### 提示词文件格式（`prompts/`）

**Keypoint 格式**（`prompt_gen` 字段）：
```json
{
    "ID": 0,
    "object": "A modern gastropub...",
    "aspect": "ingredient quality",
    "category": "cleanliness",
    "opinion": ["dirty", "negative"],
    "sentiment": "a consistent sentiment (negative)",
    "examples": [...],
    "prompt_gen": "Write a review sentence for the restaurant: ...\n\nllm_output:The ingredient quality was disappointing...\nLabel: [['ingredient quality', 'negative']]"
}
```

**Instance 格式**（`prompt_aug` 字段）：
```json
{
    "ID": "1701_paraphrase_one",
    "prompt_aug": "Given a restaurant example review...\n\n4 Diverse Paraphrased Sentences with Labels:\n\n1. Sentence: [生成的句子]\nLabel: [[...]]"
}
```

### 合成数据格式（`property-driven/` & `seed-driven/`）

```json
[
    {
        "ID": "0",
        "Sentence": "The ingredient quality was disappointing and the cleanliness was shockingly dirty.",
        "Label": [
            ["ingredient quality", "negative"],
            ["cleanliness", "negative"]
        ]
    }
]
```

### 模型 A 输出格式
```json
{
    "ID": "001",
    "Sentence": "原始句子",
    "ProcessedLabels": [
        {"aspect": "标准化方面词", "polarity": "情感极性", "standardization_status": "modified/unchanged"}
    ],
    "OriginalLabels": [["原方面词", "原极性"]]
}
```

### 模型 B 输出格式
在模型 A 输出基础上，`ProcessedLabels` 每项增加 `sentiment_status` 字段。

### 模型 C 输出格式（详细）
```json
{
    "ID": "001",
    "OriginalSentence": "原始句子",
    "OptimizedSentence": "优化后句子",
    "SentenceOptimized": true,
    "FinalLabels": [["aspect", "polarity"]],
    "Decision": "save/discard",
    "QualityScore": 8,
    "Reasoning": "决策理由",
    "ProcessingHistory": {...}
}
```

### 最终高质量数据集格式（`*_final_dataset.json`）
仅包含 `Decision=="save"` 的数据：
```json
[
    {"ID": "0", "Sentence": "最终句子", "Label": [["aspect", "polarity"]]}
]
```

### RAG 输出格式（`*_RAG.json`）
ID 从 0 重新编号：
```json
[
    {"ID": 0, "Sentence": "去重后句子", "Label": [["aspect", "polarity"]]}
]
```

---

## 8. 配置与扩展

### 8.1 关键配置点

| 配置 | 文件位置 | 说明 |
|------|----------|------|
| LLM API | `.env`（参考 `.env.example`） | `OPENAI_API_KEY`、`OPENAI_API_BASE`、`OPENAI_Model` |
| RAG 参数 | `dual-constrained-dedup/config.py` | 输入输出路径、相似度阈值、嵌入模型 |
| 域配置 | `scripts/process_all_datasets.py` | `DOMAINS = ["res", "lap", "res15", "res16"]` |
| 数据集配置 | `scripts/process_all_datasets.py` | `DATASETS_CONFIG = [("sample", 2), ("sample", 5), ...]` |
| 默认参数 | `scripts/process_all_datasets.py` | `SEED=42, NUM_SAMPLES=4000, NUM_EXAMPLES=4` |
| 流式输出目录 | `evaluation/optimized_streaming.py` | `output_dir = "optimized_outputs"` |

### 8.2 扩展新域

1. 在 `attribute_candidates/` 中为新域添加 `objects/aspects/categories/opinions` 候选库
2. 在 `scripts/process_all_datasets.py` 的 `DOMAINS` 列表添加新域名
3. 在 `data/{new_domain}/` 放置原始数据
4. 运行 `scripts/process_all_datasets.py` 自动处理

### 8.3 切换 LLM 模型

修改 `.env` 中的 `OPENAI_Model` 即可切换 LLM 模型版本（论文默认使用 `DeepSeek-V3`）。所有模型 A/B/C 与提示词生成共享同一客户端配置。

### 8.4 注意事项

1. **路径配置**：开源版已将所有路径改为仓库相对路径（以仓库根目录为基准），无需修改即可使用；如需自定义，编辑 `dual-constrained-dedup/config.py` 或使用脚本命令行参数
2. **增量写入**：提示词生成与评估均采用增量写入，单条处理完立即落盘，支持断点续传
3. **并发保护**：评估运行时通过 `ConcurrencyProtector` 加锁，防止多进程同时写同一文件
4. **API 限流**：通过 `--delay` 参数控制请求间隔，避免触发 API 限流
5. **RAG 模块独立性**：`dual-constrained-dedup/` 为完全独立模块，不依赖项目其他代码，仅依赖 `sentence-transformers` 与 `scikit-learn`

---

> **文档生成时间**：2026-09-30
>
> **文档版本**：v1.0
>
> **基于代码状态**：当前仓库快照
