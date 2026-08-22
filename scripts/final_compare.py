"""
Main evaluation of the PROPOSED method (ProtoNet) under the leak-free protocol:

  * Proposed ProtoNet, tuned on the validation subset
    (best hyperparameters from search_protonet_hp.py: lr=5e-4, dropout=0.3,
    alpha=0.3 for prototype losses, patience=40, wd=1e-4, epochs=300), run on
    the imaging-only (381) and full (384) feature sets.
  * Leak-free protocol everywhere: outer 5-fold x 3 seeds = 15 runs; inner
    60/20/20; per-fold ComBat (fit on the inner-training partition via the site
    0-7 batch label, applied to the validation and test partitions) ->
    SelectKBest(150) -> VIF(10) -> Scaler on the inner training subset only;
    early stopping on the validation subset; binary task uses independent [1,1]
    models with the Youden threshold from the validation subset.

NOTE: This release reproduces the proposed ProtoNet evaluation only.
The *baselines* (XGBoost / SVM / MLP / ProtoPNet / PIPNet / CBM /
GRLVQ / GMLVQ) are NOT part of this code release.

Outputs: experiments/final_compare.json + _3class.csv / _2class.csv
"""
import os, sys, json, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments_proto import (eval_preds, predict_proba, load_data,
                                       SEEDS, N_FOLDS, K_FEATURES, DEFAULT_PPCL,
                                       ProtoNet, prototype_classification_loss,
                                       prototype_diversity_loss, build_fold)
from proto_vif_clean import vif_mask, INNER_VAL_FRAC

# Resolve the experiments output directory the same way experiments_proto does
# (DATA_ROOT environment variable, defaulting to three levels up from here =
# the project data root). This keeps every artifact in one place.
_DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
EXP = os.path.join(os.environ.get("DATA_ROOT", _DEFAULT_ROOT), "experiments")

# tuned HP (search_protonet_hp.py top-1)
LR, DROPOUT, ALPHA, PATIENCE, WD, EPOCHS = 5e-4, 0.3, 0.3, 40, 1e-4, 300


def prepare_fold(df, cols, X_raw, y_all, tr, te, seed, fi):
    """Leak-free per-fold pipeline (delegates to experiments_proto.build_fold).

    `cols` is the ordered list of feature column names (imaging, plus optional
    clinical age/male/MMSE); `X_raw` is df[cols].values aligned to df rows.
    """
    return build_fold(df, cols, X_raw, y_all, tr, te, seed, fi)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__(); self.gamma = gamma; self.weight = weight
    def forward(self, logits, targets):
        # Per-sample focal loss:
        #   L = -(1/N) sum_i w_{y_i} (1 - p_{i,y_i})^gamma * log p_{i,y_i}
        log_probs = F.log_softmax(logits, dim=1)
        ce = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)   # per-sample CE
        pt = torch.exp(-ce)                                          # true-class probability
        focal = (1.0 - pt) ** self.gamma * ce
        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]
            focal = focal * w
        return focal.mean()


def train_tuned(Xs, yt, Xv_s, yv, ppcl, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ProtoNet(Xs.shape[1], len(ppcl), ppcl, 64).cpu()
    model.encoder = nn.Sequential(
        nn.Linear(Xs.shape[1], 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(DROPOUT),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Linear(128, 64))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    Xs_t = torch.tensor(Xs, dtype=torch.float32); yt_t = torch.tensor(yt, dtype=torch.long)
    Xv_t = torch.tensor(Xv_s, dtype=torch.float32); yv_t = torch.tensor(yv, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xs_t, yt_t), batch_size=32, shuffle=True)
    counts = yt_t.bincount(minlength=len(ppcl)).float().clamp(min=1)
    crit = FocalLoss(gamma=2.0, weight=1. / counts)
    best_acc, best_state, no_imp = 0.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits, z = model(xb)
            loss = crit(logits, yb) + ALPHA * prototype_classification_loss(
                model.prototypes, z, yb, model.prototype_labels) + \
                ALPHA * prototype_diversity_loss(model.prototypes)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va_acc = (model(Xv_t)[0].argmax(1) == yv_t).float().mean().item()
        sched.step(va_acc)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def youden_threshold(yv, pv):
    thrs = np.unique(pv)
    best_t, best_j = 0.5, -1.0
    for t in thrs:
        se = (pv[yv == 1] >= t).mean(); sp = (pv[yv == 0] < t).mean()
        j = se + sp - 1
        if j > best_j:
            best_t, best_j = t, j
    return float(best_t)


def run_3class(df, imaging, full_feats):
    print(f"\n########## 3-CLASS FINAL (the leak-free protocol, tuned ProtoNet) ##########")
    rows = []
    cfgs = [("ProtoNet(img)", imaging), ("ProtoNet(full)", full_feats)]
    for name, cols in cfgs:
        y_all = df["label"].values.astype(int)
        X_raw = df[cols].values.astype(float)
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fi, (tr, te) in enumerate(skf.split(X_raw, y_all)):
                f = prepare_fold(df, cols, X_raw, y_all, tr, te, seed, fi)
                model = train_tuned(f["Xs"], f["yt"], f["Xv_s"], f["yv"], DEFAULT_PPCL, seed)
                pp = predict_proba(model, f["Xte_s"]); pr = pp.argmax(1)
                ev = eval_preds(f["y_te"], pr, pp)
                ev.update({"model": name, "seed": seed, "fold": fi, "task": "3class"})
                rows.append(ev)
                print(f"  [{name}] s={seed} f={fi} AUC={ev['auc_ovr_macro']:.4f} acc={ev['accuracy']:.4f}")
    return rows


def run_2class(df, imaging, full_feats):
    print(f"\n########## 2-CLASS FINAL (independent [1,1], Youden) ##########")
    rows = []
    cfgs = [("ProtoNet(img)", imaging), ("ProtoNet(full)", full_feats)]
    for name, cols in cfgs:
        sub = df[df["label"].isin([0, 2])].copy()
        y2 = (sub["label"].values == 2).astype(int)
        X2 = sub[cols].values.astype(float)
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fi, (tr, te) in enumerate(skf.split(X2, y2)):
                f = prepare_fold(sub, cols, X2, y2, tr, te, seed, fi)
                model = train_tuned(f["Xs"], f["yt"], f["Xv_s"], f["yv"], [1, 1], seed)
                pp = predict_proba(model, f["Xte_s"])
                vp = predict_proba(model, f["Xv_s"])[:, 1]
                _push_2class(rows, f, pp, vp, name, seed, fi)
    return rows


def _push_2class(rows, f, pp, vp, name, seed, fi):
    p_pos = pp[:, 1]
    auc = float(roc_auc_score(f["y_te"], p_pos))
    thr = youden_threshold(f["yv"], vp)
    pred = (p_pos >= thr).astype(int)
    ev = eval_preds(f["y_te"], pred, np.c_[1 - p_pos, p_pos])
    ev["auc_ovr_macro"] = auc
    ev["threshold"] = thr
    ev.update({"model": name, "seed": seed, "fold": fi, "task": "2class"})
    rows.append(ev)
    print(f"  [{name}] s={seed} f={fi} AUC={auc:.4f} acc={ev['accuracy']:.4f} thr={thr:.3f}")


def summarize(rows):
    df = pd.DataFrame(rows)
    agg = {}
    for m in df["model"].unique():
        s = df[df["model"] == m]
        agg[m] = {k: {"mean": round(float(s[k].mean()), 4),
                      "sd": round(float(s[k].std(ddof=1)), 4)} for k in
                  ["accuracy", "balanced_accuracy", "precision_macro",
                   "recall_macro", "f1_macro", "auc_ovr_macro"]}
    return df, agg


def main():
    df, feats, imaging, clinical = load_data()
    full_feats = imaging + clinical
    rows3 = run_3class(df, imaging, full_feats)
    rows2 = run_2class(df, imaging, full_feats)
    df3, agg3 = summarize(rows3)
    df2, agg2 = summarize(rows2)
    df3.to_csv(os.path.join(EXP, "final_compare_3class.csv"), index=False)
    df2.to_csv(os.path.join(EXP, "final_compare_2class.csv"), index=False)
    out = {"protocol": "the leak-free protocol; ProtoNet tuned (lr5e-4 do0.3 a0.3 pat40 wd1e-4 ep300); "
                       "binary independent [1,1] + Youden(val); no constraints.",
           "three_class": agg3, "two_class": agg2}
    json.dump(out, open(os.path.join(EXP, "final_compare.json"), "w"), indent=1)
    print("\n=== 3-CLASS FINAL ===")
    for m, v in sorted(agg3.items(), key=lambda kv: -kv[1]["auc_ovr_macro"]["mean"]):
        print(f"  {m:22s}: Acc {v['accuracy']['mean']:.4f}+/-{v['accuracy']['sd']:.4f} | "
              f"AUC {v['auc_ovr_macro']['mean']:.4f}+/-{v['auc_ovr_macro']['sd']:.4f}")
    print("=== 2-CLASS FINAL (independent, Youden) ===")
    for m, v in sorted(agg2.items(), key=lambda kv: -kv[1]["auc_ovr_macro"]["mean"]):
        print(f"  {m:22s}: Acc {v['accuracy']['mean']:.4f}+/-{v['accuracy']['sd']:.4f} | "
              f"AUC {v['auc_ovr_macro']['mean']:.4f}+/-{v['auc_ovr_macro']['sd']:.4f}")
    print("saved -> experiments/final_compare.json")


if __name__ == "__main__":
    main()
