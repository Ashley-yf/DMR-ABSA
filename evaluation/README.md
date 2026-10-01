# Evaluation — PRUS (A→B→C Quality Pipeline)

LLM-based three-judge pipeline that filters and repairs synthetic ABSA data. This pipeline is referred to as **PRUS** in the paper — its "maximum refinement iterations = 3" correspond to the three progressive agent stages A → B → C below (each sample is refined exactly once per stage).

## Pipeline

```
input sample ──► Model A ──► Model B ──► Model C ──► accepted sample
                aspect       sentiment     quality
                term         polarity      decision
                standardize  verify/fix    accept/revise/reject
```

| File | Role |
|---|---|
| `model_a_aspect_standardization.py` | Canonicalize aspect surface forms (LLM judge A) |
| `model_b_sentiment_evaluation.py` | Verify & fix aspect polarities (LLM judge B) |
| `model_c_quality_decision.py` | Final accept / revise / reject decision (LLM judge C) |
| `optimized_streaming.py` | **Main entry.** Streaming per-sample A→B→C, appends results to disk immediately (crash-safe, memory-friendly) |
| `run_evaluation.py` | Batch runner over dataset directories |
| `data_utils.py` | JSON I/O, validation, ID management |
| `file_lock.py` / `concurrency_protection.py` | Cross-process file safety for parallel runs |

## Usage

```bash
# streaming (recommended)
python evaluation/optimized_streaming.py property-driven/res/direct_res_sample5.json

# batch directories
python evaluation/run_evaluation.py --base-dirs seed-driven \
    --subdirs res --dataset-name refined_res --delay 2.0
```

Outputs land in `optimized_outputs/` / `outputs/` as `*_final_dataset.json`.

Configuration: API endpoint/model via environment variables (`OPENAI_API_KEY`,
`OPENAI_API_BASE`, `OPENAI_Model`) — see `../.env.example`.

Details: [../docs/README_optimized_streaming.md](../docs/README_optimized_streaming.md)
