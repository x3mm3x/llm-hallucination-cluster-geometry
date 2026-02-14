# Detecting LLM Hallucinations via Embedding Cluster Geometry

Code and data for the paper *"Detecting LLM Hallucinations via Embedding Cluster Geometry: A Three-Type Taxonomy with Measurable Signatures"*.

We propose a geometric taxonomy of LLM hallucinations based on three measurable signatures in token embedding cluster structure:

- **Type 1 (Center-drift)** — generation collapses toward high-frequency generic tokens under weak context
- **Type 2 (Wrong-well convergence)** — confident commitment to a locally coherent but contextually wrong cluster
- **Type 3 (Coverage gap)** — query lands in regions where no cluster structure exists

Three statistics validate the geometric prerequisites across 11 transformer models:

| Statistic | What it measures | Result |
|-----------|-----------------|--------|
| λ_r (radial information gradient) | Nonlinear norm–information relationship | 9/11 significant |
| β_diff (cluster cohesion) | Clusters tighter than background | 11/11 significant |
| α (polarity coupling) | Antonym axes within clusters | 11/11 present |

## Repository Structure

```
├── geometric_survey.py        # Main 11-model survey (Table 1, cross-model figures)
├── bert_baseline.py           # Single-model deep analysis on BERT-base
├── diagnostic_anomalies.py    # Anomaly diagnostics (ALBERT, MiniLM, GPT-2)
├── paper/
│   ├── main.tex               # Paper source
│   └── references.bib         # Bibliography
├── figures/                   # Generated figures (after running scripts)
├── requirements.txt
└── README.md
```

## Requirements

- Python ≥ 3.10
- CPU only, 16 GB RAM
- ~8 GB disk for model downloads (first run)

```bash
pip install -r requirements.txt
```

## Running

The scripts are independent. Run order matters only if you want consistent outputs across all tables and figures.

```bash
# 1. Full 11-model survey — produces Table 1 data, cross-model figures
python geometric_survey.py
# Output: ./results_geometric_survey/

# 2. BERT-base deep analysis — produces single-model figures
python bert_baseline.py
# Output: ./results_bert_baseline/

# 3. Anomaly diagnostics — ALBERT, MiniLM, GPT-2 deep dive
python diagnostic_anomalies.py
# Output: ./results_anomaly_diagnostics/
```

Total runtime is approximately 90 minutes on a 4-core CPU.

## Models Analyzed

| Model | Dim | Type | HuggingFace ID |
|-------|-----|------|----------------|
| BERT-base | 768 | Encoder | `bert-base-uncased` |
| BERT-large | 1024 | Encoder | `bert-large-uncased` |
| DistilBERT | 768 | Encoder | `distilbert-base-uncased` |
| RoBERTa-base | 768 | Encoder | `roberta-base` |
| RoBERTa-large | 1024 | Encoder | `roberta-large` |
| ALBERT-base | 128 | Encoder | `albert-base-v2` |
| ELECTRA-base | 768 | Encoder | `google/electra-base-discriminator` |
| DeBERTa-base | 768 | Encoder | `microsoft/deberta-base` |
| MiniLM-L6 | 384 | Encoder | `nreimers/MiniLM-L6-H384-uncased` |
| GPT-2 small | 768 | Decoder | `gpt2` |
| GPT-2 medium | 1024 | Decoder | `gpt2-medium` |

All models are downloaded automatically via HuggingFace `transformers` on first run.

## Key Outputs

**`geometric_survey.py`** produces:
- `full_results.csv` — all metrics for all 11 models (Table 1 source)
- `full_results.json` — detailed JSON results
- `cross_model_comparison.txt` — human-readable summary
- `fig_lambda_comparison.png` — λ_r across models
- `fig_cross_model_summary.png` — α, β, λ_r comparison

**`bert_baseline.py`** produces:
- `fig_lambda_s.png` — radial information gradient (BERT)
- `fig_alpha_beta.png` — α and β distributions
- `fig_type_signatures.png` — hallucination type zones
- `summary_report_v2.txt` — full BERT analysis report

**`diagnostic_anomalies.py`** produces:
- `fig_anomaly_deep_dive.png` — radial profiles and PCA for anomaly models
- `diagnostic_report.txt` — diagnostic report

## Reproducibility

All experiments use `random_state=42`. The pipeline is deterministic given the same model weights and wordfreq data. No GPU required.

## License

MIT
