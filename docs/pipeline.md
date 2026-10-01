# Pipeline Reproduction Guide

Stage-by-stage commands with inputs and outputs. Run everything from the repository root.

## Prerequisites

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-xxx          # any OpenAI-compatible endpoint
export OPENAI_API_BASE=https://api.openai.com/v1
export OPENAI_Model=DeepSeek-V3
```

Notation: `DOMAIN ∈ {res, lap, res15, res16}`, `SHOT ∈ {2, 5}`.

---

## Stage 1 — Prompt Generation (`utils_prompt/`)

### 1a. Keypoint-driven (attribute prompting)

Populates templates from `prompts/keypoint_driven.py` with candidate values from
`attribute_candidates/` (AGS candidate library: categories → aspects → objects → opinions → sentiments).

```bash
# few-shot sample (SHOT = 2 or 5)
python utils_prompt/get_keypoint_prompts.py \
    --domain $DOMAIN --dataset sample \
    --seed 42 --num_shots $SHOT \
    --num_samples 4000 --num_examples 4

# full split (uses every example, num_shots ignored)
python utils_prompt/get_keypoint_prompts.py \
    --domain $DOMAIN --dataset train \
    --seed 42 --num_shots -1 \
    --num_samples 4000 --num_examples 4
```

Output: `prompts/keypoint_prompts/${DOMAIN}_sample${SHOT}.json`
(each record contains a `prompt_gen` field ending with an `llm_output:` slot)

### 1b. Instance-driven (selective reconstruction)

Samples real sentences, masks aspect terms, and asks the LLM to reconstruct:

```bash
python utils_prompt/get_instance_prompts.py \
    --domain $DOMAIN --dataset sample \
    --seed 42 --num_shots $SHOT
```

Output: `prompts/instance_prompts/${DOMAIN}_sample${SHOT}.json`

> Batch alternative for all domains/splits at once: `python scripts/process_all_datasets.py`

### 1c. Call your LLM

Iterate over the prompt files and fill `prompt_gen`'s `llm_output:` slot with the
model response (`utils_prompt/qnaigc_api.py` provides a ready client).
The released prompt files under `prompts/*_prompts/` already contain the
`llm_output` responses used in the paper.

---

## Stage 2 — Output Parsing (`process_absa_data.py`)

Extracts the generated `Sentence` and `Label` from each `llm_output` and writes
the standard ABSA JSON.

```bash
# keypoint stream -> label normalization dataset
python scripts/process_absa_data.py \
    prompts/keypoint_prompts/${DOMAIN}_sample${SHOT}.json \
    property-driven ${DOMAIN}_sample${SHOT} --prompt_type keypoint

# instance stream -> label refinement dataset
python scripts/process_absa_data.py \
    prompts/instance_prompts/${DOMAIN}_sample${SHOT}.json \
    seed-driven ${DOMAIN}_sample${SHOT} --prompt_type instance
```

Outputs: `property-driven/<domain>/*.json`, `seed-driven/<domain>/*.json`

Utility: `python scripts/fix_labels_from_processed.py <processed.json> <original_prompt.json>`
patches corrected labels back into a prompt file (auto-backup created).

---

## Stage 3 — A→B→C Quality Evaluation (`evaluation/`)

Three LLM judges run sequentially on every sample:

| Model | Role | Prompt source |
|---|---|---|
| **A** — Aspect Standardization | rewrite aspect terms to canonical surface forms | `prompts/templates.py` |
| **B** — Sentiment Evaluation | verify/fix the polarity of every aspect | `prompts/templates.py` |
| **C** — Quality Decision | accept / revise / reject the sample | `prompts/templates.py` |

```bash
# recommended: streaming processor (real-time append, crash-safe)
python evaluation/optimized_streaming.py \
    property-driven/${DOMAIN}/direct_${DOMAIN}_sample${SHOT}.json

# batch directory runner
python evaluation/run_evaluation.py \
    --base-dirs seed-driven \
    --subdirs ${DOMAIN} \
    --dataset-name refined_${DOMAIN} \
    --delay 2.0
```

Outputs: `evaluation/optimized_outputs/` / `evaluation/outputs/` —
`*_final_dataset.json` containing only samples that passed model C.

Details: [README_optimized_streaming.md](README_optimized_streaming.md)

---

## Stage 4 — RAG Deduplication (`dual-constrained-dedup/`)

Removes near-duplicates using a dual constraint:
**(1) identical aspect-term set** AND **(2) sentence-embedding cosine similarity ≥ threshold**.

```bash
# 1. edit dual-constrained-dedup/config.py: INPUT_DIR -> directory from Stage 3
# 2. run
python dual-constrained-dedup/rag_retrieval.py
```

Outputs: `dual-constrained-dedup/rag_outputs/*_RAG.json` + dedup statistics in `dual-constrained-dedup/logs/`.

Key parameters (`dual-constrained-dedup/config.py`): `SIMILARITY_THRESHOLD = 0.9`,
`EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`, local path supported).

---

## Stage 5 — Visualization (`visualization/`)

INSTRUCTOR embeddings (`hkunlp/instructor-large`) + t-SNE, paper-style figures.

```bash
cd visualization

# single dataset (scatter + density figure, PNG + PDF, two-level cache)
python plot_tsne.py --data ../data/lap/sample5_all.json \
                    --output picture/lap_sample5.png \
                    --title "5%-shot Traing Data" --gpus 0

# re-render with new styles without recomputing embeddings/t-SNE
python plot_tsne.py --viz-only

# batch: all 8 paper datasets
python all.py --gpus 0            # add --skip-existing to reuse cached t-SNE
```

Outputs: `visualization/picture/*.png` (+ `pdf/`), cache in `visualization/cache/`.

---

## End-to-end shortcut

```bash
bash run.sh res 5     # Stages 1→4 for one domain/shot
```
