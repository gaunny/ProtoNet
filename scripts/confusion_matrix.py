"""Recompute the pooled 3-class confusion matrix under the leak-free protocol
(tuned ProtoNet, full features, 15 runs pooled out-of-fold)."""
import os, sys, json, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments_proto import (load_data, predict_proba, SEEDS, N_FOLDS,
                                       K_FEATURES, DEFAULT_PPCL)
from proto_vif_clean import vif_mask, INNER_VAL_FRAC
from final_compare import prepare_fold, train_tuned

EXP = os.path.join(os.environ.get("DATA_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))), "experiments")


def main():
    df, feats, imaging, clinical = load_data()
    cols = imaging + clinical
    y_all = df["label"].values.astype(int)
    X_raw = df[cols].values.astype(float)
    y_pool, p_pool = [], []
    per_fold = []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fi, (tr, te) in enumerate(skf.split(X_raw, y_all)):
            f = prepare_fold(df, cols, X_raw, y_all, tr, te, seed, fi)
            model = train_tuned(f["Xs"], f["yt"], f["Xv_s"], f["yv"], DEFAULT_PPCL, seed)
            pp = predict_proba(model, f["Xte_s"])
            y_pool.append(f["y_te"]); p_pool.append(pp)
            acc = (pp.argmax(1) == f["y_te"]).mean()
            per_fold.append(acc)
            print(f"  s={seed} f={fi} acc={acc:.4f}")
    y_pool = np.concatenate(y_pool); p_pool = np.concatenate(p_pool)
    pred = p_pool.argmax(1)
    cm = confusion_matrix(y_pool, pred, labels=[0, 1, 2])
    acc = (pred == y_pool).mean()
    print("\n=== Pooled confusion matrix (15 folds, tuned full) ===")
    print("rows=truth (HC,MCI,AD), cols=pred")
    print(cm)
    print(f"pooled accuracy = {acc:.4f} | n = {len(y_pool)}")
    out = {"confusion": cm.tolist(), "accuracy": float(acc), "n": int(len(y_pool)),
           "labels": ["HC", "MCI", "AD"],
           "per_fold_acc": [round(a, 4) for a in per_fold]}
    json.dump(out, open(os.path.join(EXP, "confusion_matrix.json"), "w"), indent=1)
    print("saved -> experiments/confusion_matrix.json")


if __name__ == "__main__":
    main()
