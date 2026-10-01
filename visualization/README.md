# Visualization — INSTRUCTOR + t-SNE

Reproduces the paper's embedding-space figures: sentiment scatter + point-density
maps for the four paradigms (*5%-shot Training Data*, *Attribute-Grounded
Synthesis*, *Seed-Driven Reconstruction*, *Dual-Constrained Deduplication*).

## Files

| File | Role |
|---|---|
| `plot_tsne.py` | **Main script.** Paper-style figures with two-level caching |
| `all.py` | Batch renderer for all 8 paper datasets (figure titles preset) |

## Features

- INSTRUCTOR (`hkunlp/instructor-large`) embeddings → t-SNE (perplexity 50, PCA init, 3000 iters)
- Two-level cache (`cache/`): embeddings and 2D coordinates are persisted, so
  restyling with `--viz-only` takes seconds and never recomputes
- Color-blind-safe palette (Wong 2011), 300 dpi PNG **and vector PDF** output
- Canvas-level centered title; multi-tick axes; legend with clean frame

## Usage

```bash
# single dataset
python plot_tsne.py --data ../data/lap/sample5_all.json \
                    --output picture/lap_sample5.png \
                    --title "5%-shot Traing Data" --gpus 0

# restyle only (uses cache, seconds)
python plot_tsne.py --viz-only

# all 8 paper datasets
python all.py --gpus 0              # --skip-existing reuses cached t-SNE
```

Outputs: `picture/*.png`, `picture/pdf/*.pdf`; cache in `cache/`.

## Dependencies

`matplotlib`, `seaborn`, `scikit-learn`, `torch`, `InstructorEmbedding`,
`sentence-transformers` — all in the top-level `requirements.txt`.
