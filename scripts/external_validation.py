"""
External-cohort validation + site-level LOSO under the leak-free protocol.

Protocol:
  * VIF-based collinearity filtering (threshold 10) applied after
    SelectKBest(150); both steps are fit on the training pool and applied to
    the held-out external cohort.
  * Training uses the tuned hyperparameters (lr=5e-4, dropout=0.3, alpha=0.3,
    patience=40, wd=1e-4, epochs=300) from final_compare.train_tuned.
  * AD-vs-nonAD and MCI-vs-AD label definitions (derived from the external
    cohort's `mci_subtype` / `label` columns), bootstrap 95% CIs, and
    site-level LOSO.
"""
import os, json, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix)

import experiments_proto as P
from proto_vif_clean import vif_mask
from final_compare import train_tuned

ROOT = os.environ.get("DATA_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
OUT = os.path.join(ROOT, "experiments")
META = ["subject", "age", "male", "MMSE", "label", "mci_subtype", "site", "dataset"]
K = 150
PPCL = [1, 2, 1]
VIF_THR = 10.0
np.random.seed(42)

# ── load data ──────────────────────────────────────────────────────
train = pd.read_csv(os.path.join(OUT, "train_features.csv"))
ext = pd.read_csv(os.path.join(OUT, "external_features.csv"))
# External clinical labels (label / mci_subtype) are merged into
# external_features.csv by combat_3datasets_leakfree.py, so no separate
# labels file is required here.
if "mci_subtype" not in ext.columns:
    raise RuntimeError(
        "external_features.csv is missing 'mci_subtype'; "
        "re-run combat_3datasets_leakfree.py before external_validation.py.")

feats = [c for c in train.columns if c not in META]
Xtr_raw = train[feats].values.astype(np.float32)
ytr3 = train["label"].values.astype(int)

# feature pipeline: SelectKBest(150) -> VIF(10) -> Scaler, fit on training pool
sel = SelectKBest(f_classif, k=min(K, Xtr_raw.shape[1]))
Xtr_f = sel.fit_transform(Xtr_raw, ytr3)
vif_m, n_kept = vif_mask(Xtr_f)
print(f"[vif] SelectKBest150 -> VIF{int(VIF_THR)} kept {n_kept}/{Xtr_f.shape[1]}")
Xtr_v = Xtr_f[:, vif_m]
sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr_v)
Xext_raw = ext[feats].values.astype(np.float32)
Xext_s = sc.transform(sel.transform(Xext_raw)[:, vif_m])
print(f"[data] train={Xtr_s.shape}, external={Xext_s.shape}")


def train_proto(X, y, seed):
    Xa, Xv, ya, yv = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    return train_tuned(Xa, ya, Xv, yv, PPCL, seed)


# ── external labels ────────────────────────────────────────────────
st = ext["mci_subtype"].fillna("NaN").astype(str).str.strip()
is_ad = st.str.contains("AD|Alzheimer", case=False) & ~st.str.contains("non", case=False)
is_unc = st.str.lower().str.startswith("uncertain") | (st.str.lower() == "unknown")
mask_all = ~is_unc.values
y_all_ad = (is_ad.values.astype(int))[mask_all]
X_all = Xext_s[mask_all]
lab = ext["label"].values
mask_mci_ad = np.isin(lab, [1, 2])
y_mci_ad = (lab[mask_mci_ad] == 2).astype(int)
X_mci_ad = Xext_s[mask_mci_ad]
print(f"[labels] AD-vs-nonAD AD={int(y_all_ad.sum())} nonAD={int((y_all_ad==0).sum())} | "
      f"MCI-vs-AD AD={int(y_mci_ad.sum())} MCI={int((y_mci_ad==0).sum())}")

print("[train] ProtoNet (tuned) on the training pool (3 seeds)...")
probas_ext = []
for s in [42, 123, 2024]:
    m = train_proto(Xtr_s, ytr3, s)
    probas_ext.append(P.predict_proba(m, Xext_s))
Pext = np.mean(probas_ext, axis=0)
# save for downstream reuse
np.save(os.path.join(OUT, "external_Pcam.npy"), Pext)
np.save(os.path.join(OUT, "external_mask74.npy"), mask_mci_ad)
np.save(os.path.join(OUT, "external_y74.npy"), y_mci_ad)


def binary_metrics(y_true, score, thr=0.5):
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {"auc": roc_auc_score(y_true, score),
            "accuracy": accuracy_score(y_true, pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, pred),
            "sensitivity": sens, "specificity": spec,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def bootstrap(y_true, score, n=1000):
    y = np.asarray(y_true); s = np.asarray(score)
    rng = np.random.default_rng(42)
    keys = {k: [] for k in ["auc", "accuracy", "balanced_accuracy",
                            "sensitivity", "specificity", "precision", "f1"]}
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        m = binary_metrics(y[idx], s[idx])
        for k in keys:
            keys[k].append(m[k])
    return {k: [float(np.percentile(np.array(v), 2.5)),
                float(np.percentile(np.array(v), 97.5))] for k, v in keys.items()}


res = {}
P_all = Pext[mask_all]; P_mci_ad = Pext[mask_mci_ad]
res["ad_vs_nonad"] = binary_metrics(y_all_ad, P_all[:, 2])
res["ad_vs_nonad"]["ci95"] = bootstrap(y_all_ad, P_all[:, 2])
res["mci_vs_ad"] = binary_metrics(y_mci_ad, P_mci_ad[:, 2])
res["mci_vs_ad"]["ci95"] = bootstrap(y_mci_ad, P_mci_ad[:, 2])
print("\n=== External cohort (ProtoNet tuned) ===")
for k in ["ad_vs_nonad", "mci_vs_ad"]:
    r = res[k]
    print(f"{k}: AUC={r['auc']:.3f} CI{r['ci95']['auc']} Acc={r['accuracy']:.3f} "
          f"BalAcc={r['balanced_accuracy']:.3f} Sens={r['sensitivity']:.3f} "
          f"Spec={r['specificity']:.3f} Prec={r['precision']:.3f} F1={r['f1']:.3f}")
    print("   confusion:", r["confusion"])

# ── LOSO ───────────────────────────────────────────────────────────
print("\n[LOSO] ProtoNet (tuned)...")
df = train.copy()
df["protocol_group"] = "cohort_A"
df.loc[df["site"].notna(), "protocol_group"] = "cohort_B_site_" + df.loc[df["site"].notna(), "site"].astype(int).astype(str)
groups = sorted(df["protocol_group"].unique())
site_aucs = []
for g in groups:
    trm = df["protocol_group"] != g
    Xg = Xtr_s[trm]; yg = ytr3[trm]
    Xt = Xtr_s[~trm]; yt = ytr3[~trm]
    vm = np.isin(yt, [0, 1, 2])
    if vm.sum() < 10:
        continue
    m = train_proto(Xg, yg, 42)
    prob = P.predict_proba(m, Xt[vm])
    auc = roc_auc_score(yt[vm], prob, multi_class="ovr", average="macro")
    if g.startswith("cohort_B"):
        site_aucs.append(auc)
    print(f"  {g}: 3-class ovr-macro AUC={auc:.3f}")
loso_mean = float(np.mean(site_aucs)); loso_std = float(np.std(site_aucs))
print(f"LOSO (cohort B sites) mean 3-class AUC = {loso_mean:.3f} +/- {loso_std:.3f}")
res["loso_proto"] = {"mean_auc": loso_mean, "std_auc": loso_std,
                     "per_site": site_aucs, "vif_kept": int(n_kept)}

out_path = os.path.join(OUT, "external_proto.json")
json.dump(res, open(out_path, "w"), indent=2)
print("\nSAVED", out_path)
