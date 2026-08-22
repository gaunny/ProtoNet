"""Drop-column retraining feature importance under the leak-free protocol.

Representative fold (seed 42, fold 0, tuned ProtoNet full): fixed
SelectKBest(150) + VIF mask, EACH post-VIF feature column is REMOVED in turn,
the scaler is re-fit on the training subset, and the model is retrained under
the identical hyper-parameters. The drop in 3-class test accuracy vs the full
baseline is the importance.

ALL post-VIF features are evaluated (no selection bias from a permutation
ranking). Output: experiments/feature_importance_retrain.json
"""
import os, sys, json, warnings, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments_proto import (build_fold, load_data, predict_proba, SEEDS,
                                 N_FOLDS, DEFAULT_PPCL)
from final_compare import train_tuned

EXP = os.path.join(os.environ.get("DATA_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))), "experiments")


def main():
    df, feats, imaging, clinical = load_data()
    cols = imaging + clinical
    y_all = df["label"].values.astype(int)
    X_raw = df[cols].values.astype(float)
    seed, fi = SEEDS[0], 0
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    tr, te = list(skf.split(X_raw, y_all))[fi]
    # Leak-free per-fold fold construction (per-fold ComBat on imaging ->
    # SelectKBest -> VIF -> Scaler), shared with the main evaluation via
    # experiments_proto.build_fold.
    d = build_fold(df, cols, X_raw, y_all, tr, te, seed, fi)
    yt, yv, y_te = d["yt"], d["yv"], d["y_te"]
    Xs, Xv_s, Xte_s = d["Xs"], d["Xv_s"], d["Xte_s"]
    n_kept = d["n_kept"]
    feat_names = d["feat_names"]
    # Raw (unscaled) post-VIF blocks, re-scaled per dropped column below.
    Xtr_raw = d["Xtr_vif"]
    Xv_raw = d["Xv_vif"]
    Xte_raw = d["Xte_vif"]
    print(f"seed{seed} fold{fi}: test n={len(y_te)}, post-VIF features={n_kept}")

    model = train_tuned(Xs, yt, Xv_s, yv, DEFAULT_PPCL, seed)
    base_acc = (predict_proba(model, Xte_s).argmax(1) == y_te).mean()
    print(f"baseline acc = {base_acc:.4f}")

    # ALL post-VIF features (no preselection from the permutation ranking)
    targets = set(feat_names)

    items = []
    for j, fn in enumerate(feat_names):
        if fn not in targets:
            continue
        t0 = time.time()
        drop_idx = [k for k in range(n_kept) if k != j]
        sc2 = StandardScaler()
        Xs2 = sc2.fit_transform(Xtr_raw[:, drop_idx])
        Xv2 = sc2.transform(Xv_raw[:, drop_idx])
        Xte2 = sc2.transform(Xte_raw[:, drop_idx])
        m2 = train_tuned(Xs2, yt, Xv2, yv, DEFAULT_PPCL, seed)
        acc = (predict_proba(m2, Xte2).argmax(1) == y_te).mean()
        drop = base_acc - acc
        items.append({"feature": fn, "drop": round(drop, 4),
                      "retrained_acc": round(float(acc), 4)})
        print(f"  drop-col {fn[:46]:48s} acc={acc:.4f} drop={drop:+.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    items.sort(key=lambda r: -r["drop"])
    json.dump({"method": "drop-column retraining (fixed SelectKBest+VIF masks), "
                          "tuned ProtoNet full, seed42 fold0",
               "baseline_acc": round(float(base_acc), 4), "n_features": n_kept,
               "all": items, "top15": items[:15]},
              open(os.path.join(EXP, "feature_importance_retrain.json"), "w"), indent=1)
    print("\n=== TOP (retrained drop) ===")
    for r in items:
        print(f"  {r['feature'][:48]:50s} drop={r['drop']:+.4f}")


if __name__ == "__main__":
    main()
