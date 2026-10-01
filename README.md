# DMR-ABSA: Dual-Perspective Multi-Agent Data Synthesis for Low-Resource Aspect-Based Sentiment Analysis

Official repository for the paper **DMR-ABSA**. It releases the **full pipeline code**, **all prompt templates**, and **the synthetic datasets** used in the paper, so that every step — from LLM-based data synthesis to multi-agent quality evaluation, RAG-based deduplication, and t-SNE visualization — can be inspected and reproduced.

---

## 1. Overview

DMR-ABSA builds high-quality synthetic training data for low-resource (few-shot) ABSA through two cooperating designs:

- **Dual-Perspective synthesis** — *attribute-grounded* generation (keypoint stream: domain → aspect → object → opinion → sentiment) and *instance reconstruction* generation (mask real aspect terms and let the LLM rebuild the sentence around them);
- **Multi-Agent quality control** — three LLM agents (aspect standardization → sentiment verification → quality decision; referred to as **PRUS** in the paper) plus an embedding-based **RAG deduplication** module.

```
                         ┌─────────────────────────────────────────────┐
                         │  Stage 1  Prompt Generation (utils_prompt/) │
                         │  keypoint-driven      instance-driven       │
                         │  (attribute prompting)(selective masking)   │
                         └──────────────┬──────────────────────────────┘
                                        │  prompts/*_{keypoint,instance}_prompts/
                                        ▼
        Original ABSA data ────►  LLM  (AI-compatible API)
                                        │  llm_output filled into prompt_gen
                                        ▼
                         ┌─────────────────────────────────────────────┐
                         │  Stage 2  Output Parsing (process_absa_data)│
                         │  extract Sentence + [[aspect, polarity]]    │
                         └──────────────┬──────────────────────────────┘
                                        ▼
                 property-driven/          seed-driven/
                 (label normalization)         (label refinement: norm + noisy self-training)
                                        │
                                        ▼
                         ┌─────────────────────────────────────────────┐
                         │  Stage 3  PRUS A→B→C Evaluation (evaluation/)│
                         │  A: aspect standardization                  │
                         │  B: sentiment polarity check                │
                         │  C: overall quality decision                │
                         └──────────────┬──────────────────────────────┘
                                        ▼
                         ┌─────────────────────────────────────────────┐
                         │  Stage 4  RAG Dedup (dual-constrained-dedup/)│
                         │  aspect-set match + semantic similarity     │
                         └──────────────┬──────────────────────────────┘
                                        ▼
                              Final synthetic data (*_RAG.json)
                                        │
                                        ▼
                         ┌─────────────────────────────────────────────┐
                         │  Stage 5  Visualization (visualization/)     │
                         │  INSTRUCTOR embeddings + t-SNE              │
                         └─────────────────────────────────────────────┘
```

**Four paradigms compared in the paper** (titles used in the figures):

| Paradigm | Title | Data directory |
|---|---|---|
| 5% few-shot training data (baseline) | *5%-shot Training Data* | `data/{lap,res}/sample5_all.json` |
| Keypoint-driven synthesis + label normalization | *Attribute-Grounded Synthesis* | `property-driven/` |
| Instance-driven synthesis + label refinement | *Seed-Driven Reconstruction* | `seed-driven/` |
| + RAG-based deduplication | *Dual-Constrained Deduplication* | `dual-constrained-dedup/rag_outputs/` |

---

## 2. Repository Structure

```
DMR-ABSA-open/
├── run.sh                     # One-command pipeline entry (Stages 1→4)
├── scripts/                   # Pipeline scripts
│   ├── process_absa_data.py       # Stage 2: parse LLM outputs into standard ABSA JSON
│   ├── process_all_datasets.py    # Batch driver for all domains/splits
│   └── fix_labels_from_processed.py  # Utility: patch labels back into prompt files
│
├── data/                      # Original & few-shot ABSA datasets
│   ├── lap/  res/  res15/  res16/     # train/val/test + sample2/sample5 few-shot subsets
├── attribute_candidates/  # AGS: LLM-generated attribute candidate values
│   ├── categories.py  aspects.py  opinions.py  sentiments.py  objects.py
├── prompts/                   # All prompt templates + released prompt files
│   ├── keypoint_driven.py  instance_driven.py  templates.py
│   ├── keypoint_prompts/      # Released keypoint-driven prompts (paper artifact)
│   └── instance_prompts/      # Released instance-driven prompts (paper artifact)
├── utils_prompt/              # Prompt population + LLM API client
│   ├── get_keypoint_prompts.py    # Attribute prompting (Stage 1a)
│   ├── get_instance_prompts.py    # Sample combination & masking (Stage 1b)
│   └── qnaigc_api.py              # OpenAI-compatible client (env-based keys)
│
├── property-driven/       # Synthetic data w/ label normalization (paper artifact)
├── seed-driven/    # Synthetic data w/ label refinement (paper artifact)
│
├── evaluation/                # Stage 3: PRUS — A→B→C streaming quality pipeline
│   ├── optimized_streaming.py     # Main entry (streaming, real-time saving)
│   ├── run_evaluation.py          # Batch directory runner
│   ├── model_a_aspect_standardization.py
│   ├── model_b_sentiment_evaluation.py
│   ├── model_c_quality_decision.py
│   └── data_utils.py  file_lock.py  concurrency_protection.py
│
├── dual-constrained-dedup/                # Stage 4: semantic deduplication
│   ├── config.py                  # Thresholds / paths (edit INPUT_DIR here)
│   └── rag_retrieval.py           # Main entry
│
├── visualization/             # Stage 5: t-SNE embedding visualization
│   ├── plot_tsne.py               # Paper-style scatter + density figures (cached)
│   └── all.py                     # Batch renderer for all 8 datasets
│
├── docs/                      # Detailed documentation
│   ├── CODE_WIKI.md               # Deep dive into the original codebase
│   ├── pipeline.md                # Stage-by-stage reproduction guide
│   ├── data_format.md             # JSON schemas of every data directory
│   └── README_optimized_streaming.md
├── requirements.txt
├── .env.example               # Copy to .env, fill in your API key
└── LICENSE
```

---

## 3. Quick Start

### 3.1 Environment

```bash
conda create -n dmrabsa python=3.10 -y
conda activate dmrabsa
pip install -r requirements.txt

# spaCy English model
python -m spacy download en_core_web_sm
```

### 3.2 Configure your LLM API

Any OpenAI-compatible endpoint works (OpenAI, DeepSeek, vLLM gateway, ...):

```bash
cp .env.example .env
# edit .env, then:
source .env
```

### 3.3 Run the full pipeline

```bash
bash run.sh res 5        # domain: res | lap | res15 | res16 ; shot: 2 | 5
```

Or run each stage manually — see [docs/pipeline.md](docs/pipeline.md).

### 3.4 Reproduce the paper figures

```bash
cd visualization
python all.py --gpus 0          # batch t-SNE figures for all datasets
# outputs -> visualization/picture/  (+ pdf/)
```

---

## 4. Data Format

All datasets are JSON arrays. The canonical sample format:

```json
{
  "ID": "0",
  "Sentence": "The ingredient quality is disappointing and the place is dirty.",
  "Label": [
    ["ingredient quality", "negative"],
    ["cleanliness", "negative"]
  ],
  "aspects": [
    {"from": 4,  "to": 22, "target": "ingredient quality", "polarity": "negative"},
    {"from": 47, "to": 56, "target": "cleanliness",        "polarity": "negative"}
  ]
}
```

See [docs/data_format.md](docs/data_format.md) for the exact schema of every directory.

---

## 5. Documentation

| Document | Content |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | Stage-by-stage commands, inputs/outputs, parameters |
| [docs/data_format.md](docs/data_format.md) | JSON schemas for data/, *_driven/, prompts/, rag outputs |
| [docs/CODE_WIKI.md](docs/CODE_WIKI.md) | Deep dive: module internals, call graphs, design notes |
| [docs/README_optimized_streaming.md](docs/README_optimized_streaming.md) | Details of the streaming A→B→C evaluator |

---

## 6. Notes

- **API keys**: never hardcode keys. Configure via environment variables (`.env`, see `.env.example`). All shipped code reads keys from `OPENAI_API_KEY`.
- **Prompt artifacts**: `prompts/keypoint_prompts/` and `prompts/instance_prompts/` (~100 MB) are the exact prompts fed to LLMs in the paper; you can regenerate them with `process_all_datasets.py`.
- **Reproducibility**: t-SNE figures use `random_state=42` and a two-level cache (embeddings + 2D coordinates), so restyling never recomputes.

## 7. License

Released under the [MIT License](LICENSE). The underlying SemEval ABSA datasets (lap/res15/res16/res) follow their original licenses — please cite the respective SemEval papers when using them.
