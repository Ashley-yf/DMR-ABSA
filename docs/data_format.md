# Data Format Reference

All datasets are UTF-8 JSON arrays (`utf-8-sig` safe). Every directory below exists in this repository.

## 1. `data/` — Original & few-shot ABSA datasets

```
data/
├── lap/    res/        # SemEval 2014 laptop & restaurant
├── res15/  res16/      # SemEval 2015 / 2016 restaurant
```

Each domain contains `train / val / test` splits plus few-shot subsets:

| File | Meaning |
|---|---|
| `sample2_all.json` | 2%-shot randomly sampled training subset |
| `sample5_all.json` | 5%-shot randomly sampled training subset |
| `train_all.json` / `train.json` | full training split |

Record schema:

```json
{
  "ID": "0",
  "sentence": "The ingredient quality is disappointing and the place is dirty.",
  "aspects": [
    {"from": 4,  "to": 22, "target": "ingredient quality", "polarity": "negative"},
    {"from": 47, "to": 56, "target": "cleanliness",        "polarity": "negative"}
  ]
}
```

- `from` / `to`: character offsets of the aspect term in the sentence.
- `polarity ∈ {positive, negative, neutral, conflict}` (conflict filtered in the pipeline).

Some splits use the alternative label format `"Label": [["aspect", "polarity"], ...]`;
loaders in the pipeline accept both.

## 2. `attribute_candidates/` — AGS attribute candidate values

Python modules, one list per attribute category (keypoint-driven / AGS synthesis inputs):

`categories.py`, `aspects.py`, `opinions.py`, `sentiments.py`, `objects.py`

## 3. `prompts/` — Prompt templates & released prompts

| Path | Content |
|---|---|
| `keypoint_driven.py` / `instance_driven.py` | Template strings for the two synthesis streams |
| `templates.py` | Baseline & evaluation judge prompts + helpers |
| `keypoint_prompts/<domain>_<split>[_<shot>].json` | Released keypoint prompts **with paper LLM responses** |
| `instance_prompts/<domain>_<split>[_<shot>].json` | Released instance prompts with responses |

Prompt record schema (keypoint example):

```json
{
  "ID": "0",
  "keypoint_prompt": "Write a review sentence for the laptop: ... Label the sentence by ...",
  "prompt_gen": "Write a review sentence ...\nllm_output:\nThe laptop offers impressive lid rigidity...\nLabel: [['lid rigidity', 'positive']]"
}
```

`prompt_gen` = prompt + `llm_output:` + model response. Stage 2 parses the
`llm_output` sentence and `Label:` block.

## 4. `property-driven/` — Synthesis w/ label normalization

```
property-driven/<domain>/*.json
```

Standard ABSA records produced by Stage 2 (keypoint stream):

```json
{
  "ID": "0",
  "Sentence": "The laptop offers impressive lid rigidity and efficient software.",
  "Label": [["lid rigidity", "positive"], ["efficient software", "positive"]]
}
```

## 5. `seed-driven/` — Synthesis w/ label refinement

```
seed-driven/<domain>/direct_<domain>_sample<shot>.json
```

Same schema as §4, instance stream, labels cleaned by the refinement module
(normalization + noisy self-training).

## 6. `evaluation/outputs|optimized_outputs/` — Evaluated data (generated)

Same schema as §4 — only samples that passed Model C (quality decision), with
canonical aspect surface forms (Model A) and verified polarities (Model B).

## 7. `dual-constrained-dedup/rag_outputs/` — Final deduplicated data

```
*_RAG.json
```

Same schema as §4 with sequential local IDs. This is the final training data of
the *Dual-Constrained Deduplication* paradigm in the paper. Two example outputs
(`norm_res_final_dataset_RAG.json`, `refined_res_final_dataset_RAG.json`) are
shipped in the repository.

## Label formats at a glance

| Location | Format |
|---|---|
| `data/` raw | `aspects: [{target, polarity, from, to}]` or `Label: [[aspect, polarity]]` |
| `*_driven` | `Label: [[aspect, polarity]]` |
| Paper figures | sampled + RAG-deduplicated variants of the above |
