# Code Release

Analysis code for an interpretable multimodal neuroimaging framework for
Alzheimer's disease diagnosis (HC / MCI / AD classification) with
evidence-grounded personalized clinical reporting.

The raw cohort data are **not redistributed** in this release. All input paths
and credentials are passed on the command line or via environment variables;
nothing is hard-coded.

## Layout

```
code_release/
├── scripts/
│   ├── experiments_proto.py          # ProtoNet architecture, data loading, base trainer
│   ├── proto_vif_clean.py            # VIF-based collinearity filtering (threshold 10)
│   ├── final_compare.py              # 15-run (5-fold x 3 seeds) main evaluation
│   ├── feature_importance_retrain.py # drop-column retraining feature importance
│   ├── centroid_model.py             # centroid-initialized display model
│   ├── external_validation.py        # external-cohort validation + site-level LOSO
│   └── confusion_matrix.py           # pooled out-of-fold confusion matrix
├── combat_harmonizer.py              # leak-free empirical-Bayes ComBat
├── combat_3datasets_leakfree.py      # harmonization orchestration (argparse inputs)
├── prototype_rag.py                  # RAG report generator (argparse inputs + API key)
├── example_reports_output/           # created at runtime by prototype_rag.py
├── requirements.txt                  # pinned dependencies
└── run_all.sh                        # one-shot pipeline (argparse inputs)
```

## Protocol

Leak-free, no test leakage:

- **Preprocessing.** A global ComBat is fit on the two training cohorts (A and
  B, age/sex covariates) and applied as a fixed transform to the held-out
  external cohort. Within each CV fold, a second per-fold ComBat is fit on the
  fold's training partition using the site batch label (`site` 0 = cohort A,
  1..7 = the seven sub-centres of cohort B) and applied to the fold's
  validation and test partitions. Then `SelectKBest(150)` -> `VIF(10)` ->
  `StandardScaler`, all fit only on the inner-training subset.
- **Split.** Outer 5-fold x 3 seeds (42/123/2024) = 15 runs; exact 60/20/20
  train/validation/test per fold; validation-based early stopping.
- **Model.** ProtoNet, 64-d latent encoder, prototypes `[1,2,1]` (HC, MCI, AD),
  focal loss (gamma=2.0) with inverse-frequency class weights, prototype
  classification + diversity losses (alpha=0.3). Tuned: Adam lr 5e-4, wd 1e-4,
  dropout 0.3, batch 32, up to 300 epochs, patience 40.
- **Tasks.** Binary (HC vs AD, prototypes `[1,1]`, Youden threshold on the
  validation split) and three-class (HC/MCI/AD); mean +/- SD over 15 runs.

## Files

| File | Purpose |
|------|---------|
| `scripts/final_compare.py` | Main evaluation of the proposed ProtoNet (3-class and independent 2-class), 15 runs. (Baselines are NOT part of this release.) |
| `scripts/external_validation.py` | Held-out external-cohort validation (AD-vs-nonAD and MCI-vs-AD derived from `mci_subtype` / `label`) + site-level LOSO. |
| `scripts/feature_importance_retrain.py` | Drop-column retraining feature importance (MMSE dominant). |
| `scripts/centroid_model.py` | Centroid-initialized display model. |
| `scripts/confusion_matrix.py` | Pooled out-of-fold three-class confusion matrix. |
| `scripts/experiments_proto.py` | Shared architecture / data loading / training utilities. |
| `scripts/proto_vif_clean.py` | VIF filtering + main-config evaluation (3-class matches `final_compare.py`; its 2-class is the P(AD) marginalization of the 3-class model). |
| `combat_3datasets_leakfree.py`, `combat_harmonizer.py` | Leak-free ComBat (fit training cohorts with covariates -> transform external cohort). |
| `prototype_rag.py` | RAG report generator: embedding + FAISS retrieval, LLM report synthesis with inline PubMed citations (metadata resolved from PMIDs via NCBI E-utilities `esummary`). |

## Dependencies

```bash
pip install -r requirements.txt
```

Pinned, mutually compatible; CPU-only torch is sufficient. The RAG stack
targets the langchain 0.3-era API — do not upgrade to langchain >= 1.0.

## Data

Provide your own cohort tables under `DATA_ROOT` (default `../..` relative to
this folder) or pass absolute paths. Required files:

- `cohort_a.csv` — training cohort A (single-site). Columns: `subject`,
  `label` (0/1/2), `age`, `male`, `MMSE`, `site` (=0), imaging features.
- `cohort_b.csv` — training cohort B (multi-site). Same layout, `site` in {1..7}.
- `external.csv` — held-out external cohort. `subject`, `age`, `male`, same
  imaging feature column names as the training cohorts.
- `external_labels.csv` — external cohort clinical labels (`subject`, `label`,
  `mci_subtype`). `mci_subtype` marks AD (strings containing AD/Alzheimer) and
  non-AD cases; rows starting with "uncertain"/"unknown" are excluded from the
  AD-vs-nonAD evaluation.

Imaging feature column names must match exactly across the three cohort files.
Clinical columns (`age`/`male`/`MMSE`/`site`/`label`/`mci_subtype`) are never
harmonized. `site` is mandatory in the training cohorts (per-fold ComBat batch
label); `age`/`male` are the ComBat covariates.

Step 1 of the pipeline writes the harmonized inputs consumed by the remaining
steps to `${DATA_ROOT}/experiments/`:

- `train_features.csv` — harmonized training pool (`site` preserved).
- `external_features.csv` — harmonized external cohort (`label`/`mci_subtype`
  merged in).

All outputs are written to `${DATA_ROOT}/experiments/`.

## Usage

Each script takes its inputs via `argparse` (run `python <script> --help` for
the full option list). The harmonizer is the entry point that produces the
harmonized tables for the rest of the pipeline:

```bash
# 1. Harmonization (your data in):
python combat_3datasets_leakfree.py \
    --train-a cohort_a.csv --train-b cohort_b.csv \
    --external external.csv --external-labels external_labels.csv

# 2. Main evaluation + analyses:
python scripts/final_compare.py
python scripts/feature_importance_retrain.py
python scripts/external_validation.py
python scripts/confusion_matrix.py
```

The RAG report step needs an LLM API key and the model-output / feature tables
produced by your own run (paths are explicit, nothing is assumed):

```bash
python prototype_rag.py \
    --api-key sk-... \
    --test-csv experiments/external_features.csv \
    --test-json <inference results JSON> \
    --prototype-json <prototype explanations JSON> \
    --importance-json <feature importance JSON> \
    --features-range-csv <per-feature normal-range CSV>
```

Or run the whole pipeline in one command (same options as the harmonizer, plus
`--api-key`):

```bash
bash run_all.sh --train-a cohort_a.csv --train-b cohort_b.csv \
    --external external.csv --external-labels external_labels.csv \
    --api-key sk-...           # omit --api-key to skip the RAG step
```

## Environment variables

Environment variables are used as defaults when the corresponding flag is not
passed.

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATA_ROOT` | Root holding input CSVs / result JSONs | `../..` |
| `OPENAI_API_KEY` | LLM API key for the RAG step (or `--api-key`) | *unset* |
| `OPENAI_API_BASE` | Optional custom API endpoint (or `--api-base`) | *unset* |
| `PUBMED_FAISS_INDEX` | PubMed FAISS index for retrieval | `pubmed_faiss_index_new` |
