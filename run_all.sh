#!/usr/bin/env bash
# ============================================================================
# run_all.sh — reproduce the full leak-free protocol pipeline.
#
# All inputs can be passed on the command line; every option also has an
# environment-variable equivalent (set the env var before calling instead).
#
# Usage:
#   bash run_all.sh [--data-root DIR] [--train-a FILE] [--train-b FILE]
#                   [--external FILE] [--external-labels FILE]
#                   [--api-key KEY] [--api-base URL] [--pubmed-index DIR]
#
# Steps:
#   1. Leak-free ComBat harmonization (fit training cohorts -> transform external)
#   2. Main evaluation (15-run nested CV, proposed ProtoNet)
#   3. Feature importance (drop-column retraining)
#   4. External validation + LOSO + confusion matrix
#   5. RAG report generation (needs an API key; skipped when absent)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-../..}"
TRAIN_A="${TRAIN_A:-cohort_a.csv}"
TRAIN_B="${TRAIN_B:-cohort_b.csv}"
EXTERNAL="${EXTERNAL:-external.csv}"
EXTERNAL_LABELS="${EXTERNAL_LABELS:-external_labels.csv}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
OPENAI_API_BASE="${OPENAI_API_BASE:-}"
PUBMED_FAISS_INDEX="${PUBMED_FAISS_INDEX:-pubmed_faiss_index_new}"

usage() { sed -n '2,20p' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)       DATA_ROOT="$2"; shift 2 ;;
    --train-a)         TRAIN_A="$2"; shift 2 ;;
    --train-b)         TRAIN_B="$2"; shift 2 ;;
    --external)        EXTERNAL="$2"; shift 2 ;;
    --external-labels) EXTERNAL_LABELS="$2"; shift 2 ;;
    --api-key)         OPENAI_API_KEY="$2"; shift 2 ;;
    --api-base)        OPENAI_API_BASE="$2"; shift 2 ;;
    --pubmed-index)    PUBMED_FAISS_INDEX="$2"; shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "unknown option: $1"; usage; exit 1 ;;
  esac
done

export DATA_ROOT OPENAI_API_KEY OPENAI_API_BASE PUBMED_FAISS_INDEX

echo ">> [1/5] Leak-free ComBat harmonization (fit training cohorts -> transform external)"
python combat_3datasets_leakfree.py --train-a "$TRAIN_A" --train-b "$TRAIN_B" \
    --external "$EXTERNAL" --external-labels "$EXTERNAL_LABELS"

echo ">> [2/5] Main evaluation (ProtoNet, 15 runs)"
python scripts/final_compare.py

echo ">> [3/5] Drop-column feature importance"
python scripts/feature_importance_retrain.py

echo ">> [4/5] External validation + LOSO + confusion matrix"
python scripts/external_validation.py
python scripts/confusion_matrix.py

if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo ">> RAG report generation SKIPPED (no API key)."
  echo "   Re-run with --api-key <KEY> (or export OPENAI_API_KEY), then:"
  echo "   python prototype_rag.py --api-key <KEY> [--api-base <URL>]"
else
  echo ">> RAG evidence-grounded clinical report generation"
  python prototype_rag.py --api-key "$OPENAI_API_KEY" --api-base "$OPENAI_API_BASE"
fi

echo ">> Done. Outputs are written under experiments/ (JSON/CSV artifacts)."
