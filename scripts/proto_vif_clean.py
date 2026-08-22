"""
the leak-free protocol validation for the proposed ProtoNet:

  * Feature pipeline fit on inner-train subset ONLY (leak-free):
        SelectKBest(f_classif, k=150) -> iterative VIF removal (thr=10) -> StandardScaler
  * Split per outer fold: train 60% / validation 20% / test 20% (exact):
        outer StratifiedKFold(5, seed) -> 80% train fold; inside it
        train_test_split(25%, stratify, rs=seed*100+fold) =>
        60% inner-train / 20% validation / 20% test.
        Early stopping uses the validation subset (NOT the test fold).
  * 15 independent runs (5-fold x SEEDS=[42,123,2024]); metrics mean+/-SD
    (ddof=1) with empirical 2.5/97.5 percentile CI.
  * Same protocol for: main (img-only 381 / full 384 features).

Outputs under experiments/:
  vif_clean_reval.json          (main + iron-rule check)
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments_proto import (
    build_fold, predict_labels, predict_proba, eval_preds,
    load_data, SEEDS, N_FOLDS, K_FEATURES, DEFAULT_PPCL,
)
# NOTE: train_tuned is imported lazily inside train_one_fold to avoid a
# circular import (final_compare imports this module at top level).

VIF_THRESHOLD = 10.0
INNER_VAL_FRAC = 0.25          # -> exact 60/20/20
# Resolve the experiments output directory the same way experiments_proto does
# (DATA_ROOT environment variable, defaulting to three levels up from here).
_DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
EXP = os.path.join(os.environ.get("DATA_ROOT", _DEFAULT_ROOT), "experiments")


def vif_mask(X, threshold=VIF_THRESHOLD):
    n_feat = X.shape[1]
    kept = list(range(n_feat))
    while len(kept) > 1:
        C = np.corrcoef(X[:, kept].T)
        C = np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(C, 1.0)
        try:
            inv = np.linalg.inv(C + np.eye(len(kept)) * 1e-6)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(C)
        vifs = np.diag(inv)
        j = int(np.argmax(vifs))
        if vifs[j] <= threshold:
            break
        kept.pop(j)
    m = np.zeros(n_feat, dtype=bool)
    m[kept] = True
    return m, len(kept)


def train_one_fold(df, cols, X_raw, y_all, tr, te, ppcl, seed, fi):
    """Leak-free fold (shared pipeline): per-fold ComBat on imaging (fit on
    inner-train, applied to val + test) -> SelectKBest -> VIF -> Scaler ->
    ProtoNet. Returns pp (3-class proba on test), pr, y_te, n_kept, Xs/yt
    (inner train) and Xte_s (transformed test features)."""
    from final_compare import train_tuned   # lazy import (circular-import guard)
    d = build_fold(df, cols, X_raw, y_all, tr, te, seed, fi)
    pnet = train_tuned(d["Xs"], d["yt"], d["Xv_s"], d["yv"], ppcl, seed=seed)
    pp = predict_proba(pnet, d["Xte_s"])
    pr = predict_labels(pnet, d["Xte_s"])
    return {"pp": pp, "pr": pr, "y_te": d["y_te"], "n_kept": d["n_kept"],
            "Xs": d["Xs"], "yt": d["yt"], "Xte_s": d["Xte_s"]}


def agg(rows, key):
    v = np.array([r[key] for r in rows])
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5))}


def ece_3class(probas, y, n_bins=10):
    conf = probas.max(1); pred = probas.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return ece


def run_main_cfg(df, cols, name):
    """Main headline config for the proposed ProtoNet (leak-free protocol)."""
    print(f"\n########## MAIN: {name} ({len(cols)} feats) ##########")
    y_all = df["label"].values.astype(int)
    X_raw = df[cols].values
    rows_3, rows_2, vif_kept = [], [], []
    P = {"y3": [], "p3": [], "y2": [], "ad": [], "pred2": []}
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fi, (tr, te) in enumerate(skf.split(X_raw, y_all)):
            r = train_one_fold(df, cols, X_raw, y_all, tr, te, DEFAULT_PPCL, seed, fi)
            vif_kept.append(r["n_kept"])
            ev3 = eval_preds(r["y_te"], r["pr"], r["pp"])
            ev3.update({"seed": seed, "fold": fi, "model": "ProtoNet", "config": name})
            rows_3.append(ev3)
            m = r["y_te"] != 1
            y2 = r["y_te"][m]; ad = r["pp"][m][:, 2]
            p2 = (ad > 0.5).astype(int)
            ev2 = eval_preds(y2, p2, np.c_[1 - ad, ad])
            ev2["auc_ovr_macro"] = float(roc_auc_score(y2, ad))   # binary AUC
            ev2.update({"seed": seed, "fold": fi, "model": "ProtoNet", "config": name})
            rows_2.append(ev2)
            P["y3"].append(r["y_te"]); P["p3"].append(r["pp"])
            P["y2"].append(y2); P["ad"].append(ad); P["pred2"].append(p2)
            print(f"  [{name}] s={seed} f={fi} | 3cl AUC {ev3['auc_ovr_macro']:.4f} "
                  f"acc {ev3['accuracy']:.4f} | 2cl AUC {ev2['auc_ovr_macro']:.4f} | "
                  f"VIF kept {r['n_kept']}")
    y3 = np.concatenate(P["y3"]); p3 = np.concatenate(P["p3"])
    y2 = np.concatenate(P["y2"]); ad = np.concatenate(P["ad"])
    return {
        "three_class": {k: agg(rows_3, k) for k in
                        ["accuracy", "balanced_accuracy", "precision_macro",
                         "recall_macro", "f1_macro", "auc_ovr_macro"]},
        "two_class": {k: agg(rows_2, k) for k in
                      ["accuracy", "balanced_accuracy", "precision_macro",
                       "recall_macro", "f1_macro", "auc_ovr_macro"]},
        "vif": {"mean_kept": float(np.mean(vif_kept)),
                "sd_kept": float(np.std(vif_kept, ddof=1))},
        "ece_3class_pooled": {"ece": float(ece_3class(p3, y3)), "n": int(len(y3))},
    }



def summarize(rows, group_col):
    df = pd.DataFrame([{"group": r[group_col], "auc": r["auc_ovr_macro"],
                        "acc": r["accuracy"], "seed": r["seed"], "fold": r["fold"]}
                       for r in rows])
    g = df.groupby("group")
    return pd.DataFrame({
        "acc_mean": g["acc"].mean(), "acc_std": g["acc"].std(),   # std -> ddof=1
        "auc_mean": g["auc"].mean(), "auc_std": g["auc"].std(),
    }).round(4).reset_index()


def main():
    df, feats, imaging, clinical = load_data()
    full_feats = imaging + clinical
    out = {"config": {
        "proto_ppcl": DEFAULT_PPCL, "seeds": SEEDS, "outer_folds": N_FOLDS,
        "n_runs": len(SEEDS) * N_FOLDS, "selectk": K_FEATURES,
        "vif_threshold": VIF_THRESHOLD, "vif_order": "SelectKBest->VIF->StandardScaler",
        "inner_val_split": "train_test_split 25% stratify (rs=seed*100+fold) inside outer train fold",
        "effective_split_per_fold": "train 60% / val 20% / test 20% (exact)"}}

    out["main"] = {"img_only": run_main_cfg(df, imaging, "img_only"),
                   "full_features": run_main_cfg(df, full_feats, "full_features")}

    json.dump(out, open(os.path.join(EXP, "vif_clean_reval.json"), "w"), indent=1)
    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- SD over 15 runs)")
    for cfg in ["img_only", "full_features"]:
        t = out["main"][cfg]["three_class"]; b = out["main"][cfg]["two_class"]
        print(f"  {cfg}: 3cl Acc {t['accuracy']['mean']:.4f}+/-{t['accuracy']['sd']:.4f} | "
              f"3cl AUC {t['auc_ovr_macro']['mean']:.4f}+/-{t['auc_ovr_macro']['sd']:.4f} | "
              f"2cl AUC {b['auc_ovr_macro']['mean']:.4f}+/-{b['auc_ovr_macro']['sd']:.4f}")
    print("saved -> experiments/vif_clean_reval.json")


if __name__ == "__main__":
    main()
