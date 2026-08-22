"""
combat_3datasets_leakfree.py
============================
Leak-free multi-cohort ComBat harmonization for the AD / MCI framework.

Design (no leakage):
  * ComBat is FIT only on the two training cohorts (A: single-site,
    B: multi-site).
  * Biological covariates (age, sex) are preserved during harmonization.
  * The fitted (reference) parameters are applied to the external cohort
    via `transform`, which never participates in parameter estimation ->
    the external cohort stays fully held out.
  * ComBat is applied ONLY to imaging / DTI features. Clinical / demographic
    variables (MMSE, age, sex) are NOT harmonized because they are not
    site-affected scanner measurements.

The empirical-Bayes ComBat implementation lives in `combat_harmonizer.py`
(ComBatHarmonizer). This script only orchestrates the data loading and the
fit-on-training / transform-external flow.

Usage:
    python combat_3datasets_leakfree.py --root <DATA_DIR> \
        --train-a cohort_a.csv --train-b cohort_b.csv \
        --external external.csv --external-labels external_labels.csv
"""
import argparse
import os
import numpy as np
import pandas as pd

from combat_harmonizer import ComBatHarmonizer


# Columns that are metadata / clinical / demographic, never harmonized as scans.
META_COLS = ["subject", "age", "male", "MMSE", "label", "mci_subtype", "site", "dataset"]
# Drop these from the set of *modeling* (harmonized) features.
NON_IMAGING = {"subject", "label", "mci_subtype", "male", "age", "site", "MMSE", "dataset"}


def load_and_harmonize(root, out, train_a, train_b, external, external_labels):
    cohort_a = pd.read_csv(os.path.join(root, train_a))
    cohort_b = pd.read_csv(os.path.join(root, train_b))
    ext = pd.read_csv(os.path.join(root, external))

    # Common imaging / DTI features shared across all three cohorts.
    common = sorted(
        (set(cohort_a.columns) & set(cohort_b.columns) & set(ext.columns)) - NON_IMAGING
    )

    def keep(df):
        return [c for c in META_COLS if c in df.columns]

    cohort_a_c = cohort_a[common + keep(cohort_a)].copy()
    cohort_b_c = cohort_b[common + keep(cohort_b)].copy()
    ext_c = ext[common + keep(ext)].copy()

    # ---- ComBat FIT on the training cohorts (A + B) with covariates age, sex ----
    train_pool = pd.concat([cohort_a_c, cohort_b_c], ignore_index=True)
    harmonizer = ComBatHarmonizer()
    harmonizer.fit(
        train_pool[common],
        batch=np.concatenate([np.zeros(len(cohort_a_c)), np.ones(len(cohort_b_c))]).astype(int),
        covariates=train_pool[["age", "male"]],
    )

    # ---- Transform training + external cohort using the SAME fitted parameters ----
    train_h = harmonizer.transform(train_pool[common], covariates_ext=train_pool[["age", "male"]])
    ext_h = harmonizer.transform(ext_c[common], covariates_ext=ext_c[["age", "male"]])

    train_pool[common] = train_h.values
    ext_c[common] = ext_h.values

    # Attach external clinical labels from the labeled file (not used for
    # parameter estimation; purely for downstream evaluation).
    ext_lab = pd.read_csv(os.path.join(root, external_labels))
    ext_lab = ext_lab[["subject", "label", "mci_subtype"]].drop_duplicates("subject")
    ext_c = ext_c.merge(ext_lab, on="subject", how="left", suffixes=("", "_src"))
    for base in ["label", "mci_subtype"]:
        if base in ext_c.columns:
            src = base + "_src"
            if src in ext_c.columns:
                ext_c[base] = ext_c[base].fillna(ext_c[src])
                ext_c = ext_c.drop(columns=[src])

    # Modeling feature set: harmonized imaging/DTI + clinical MMSE + demographics.
    model_feats = [c for c in (list(common) + ["MMSE", "age", "male"])
                   if c in train_pool.columns]

    os.makedirs(out, exist_ok=True)
    # Downstream scripts read these exact names:
    #   * experiments_proto.load_data  -> train_features.csv
    #   * external_validation.py       -> train_features.csv + external_features.csv
    # The `site` column (0 = cohort A, 1..7 = cohort B sub-centres) is preserved
    # in train_features.csv so the nested-CV can apply the per-fold ComBat.
    train_path = os.path.join(out, "train_features.csv")
    ext_path = os.path.join(out, "external_features.csv")
    train_pool.to_csv(train_path, index=False)
    ext_c.to_csv(ext_path, index=False)

    print(f"[combat] common imaging/DTI features : {len(common)}")
    print(f"[combat] model features (incl. MMSE/age/sex) : {len(model_feats)}")
    print(f"[combat] train_pool shape : {train_pool.shape}")
    print(f"[combat] external shape  : {ext_c.shape}")
    print(f"[combat] wrote {train_path}")
    print(f"[combat] wrote {ext_path}")
    return train_pool, ext_c, model_feats


def main():
    ap = argparse.ArgumentParser(description="Leak-free multi-cohort ComBat harmonization.")
    ap.add_argument("--root", default=os.environ.get("DATA_ROOT", "../.."),
                    help="directory containing the raw cohort CSVs "
                         "(default: $DATA_ROOT, else ../.. relative to this script)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <root>/experiments)")
    ap.add_argument("--train-a", default="cohort_a.csv",
                    help="training cohort A (single-site; site column = 0)")
    ap.add_argument("--train-b", default="cohort_b.csv",
                    help="training cohort B (multi-site; site column in 1..7)")
    ap.add_argument("--external", default="external.csv",
                    help="held-out external cohort (same imaging column names)")
    ap.add_argument("--external-labels", default="external_labels.csv",
                    help="external cohort clinical labels (subject, label, mci_subtype)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    out = args.out or os.path.join(root, "experiments")
    load_and_harmonize(root, out, args.train_a, args.train_b,
                       args.external, args.external_labels)


if __name__ == "__main__":
    main()
