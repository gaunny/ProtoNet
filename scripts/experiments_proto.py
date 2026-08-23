"""
ProtoNet architecture and experiments (latent=64, BN+GELU+FocalLoss,
prototype supervision). The main entry point runs the nested cross-validation
evaluation of the proposed ProtoNet on the harmonized training pool
(train_features.csv = training cohorts A + B).

Outputs (under experiments/):
  all_results_proto.json   <- consolidated ProtoNet main results
  main_nested_cv_summary.csv
"""
import sys, os, json, warnings, math
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix,
)
from scipy.stats import binomtest, chi2, norm

try:
    from scipy.stats import mcnemar
except ImportError:
    def mcnemar(y_true, y_pred, exact=True, correction=False):
        b = np.array([[sum((y_true == i) & (y_pred == j)) for j in range(2)]
                      for i in range(2)])
        if exact:
            n_dis = int(b[0][1] + b[1][0]); k = min(b[0][1], b[1][0])
            if n_dis == 0:
                p = 1.0
            else:
                p = 2 * binomtest(n_dis, 0.5, alternative='less').pvalue
            class R: statistic = float(n_dis); pvalue = min(p, 1.0)
            return R()
        stat = ((abs(b[0][1] - b[1][0]) - (1 if correction else 0)) ** 2 /
                (b[0][1] + b[1][0])) if (b[0][1] + b[1][0]) > 0 else 0.0
        class R: statistic = float(stat); pvalue = float(chi2.sf(stat, 1))
        return R()

# ══════════════════════════════════════════════════════════════════
#  ProtoNet(latent=64) — proposed architecture
# ══════════════════════════════════════════════════════════════════
PROGRESSION_DIM = 0

class ProtoNet(nn.Module):
    def __init__(self, input_dim, num_classes, prototypes_per_class_list, latent_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.prototypes_per_class_list = prototypes_per_class_list
        self.num_prototypes = sum(prototypes_per_class_list)
        self.latent_dim = latent_dim
        self.progression_dim = PROGRESSION_DIM
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )
        initial_prototypes = torch.randn(self.num_prototypes, latent_dim) * 0.1
        prototype_labels = []
        current_proto_idx = 0
        for i_class in range(num_classes):
            base_prog_val = i_class * 2.0
            for proto_in_class_idx in range(prototypes_per_class_list[i_class]):
                initial_prototypes[current_proto_idx, self.progression_dim] += (
                    base_prog_val + proto_in_class_idx * 0.5)
                prototype_labels.append(i_class)
                current_proto_idx += 1
        self.prototypes = nn.Parameter(initial_prototypes)
        self.register_buffer('prototype_labels',
                             torch.tensor(prototype_labels, dtype=torch.long))

    def forward(self, x):
        z = self.encoder(x)
        dists = torch.cdist(z, self.prototypes)
        sims = -dists
        class_logits = torch.zeros(x.size(0), self.num_classes, device=x.device)
        for c in range(self.num_classes):
            mask = (self.prototype_labels == c).float()
            class_logits[:, c] = (sims * mask).sum(dim=1) / (mask.sum() + 1e-6)
        return class_logits, z


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
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


def prototype_classification_loss(prototypes, features, labels, proto_labels, temperature=1.0):
    sim_matrix = torch.mm(features, prototypes.t()) / temperature
    mask = proto_labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_sim = sim_matrix[mask]; neg_sim = sim_matrix[~mask]
    return (-torch.log(torch.sigmoid(pos_sim) + 1e-6).mean()
            - torch.log(1 - torch.sigmoid(neg_sim) + 1e-6).mean())


def prototype_diversity_loss(prototypes):
    sim_matrix = torch.mm(prototypes, prototypes.t())
    identity = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
    return (sim_matrix * (1 - identity)).mean()


def train_epoch_proto(model, loader, optimizer, device, alpha=0.3, beta=0.3):
    model.train()
    total_loss = 0.0; correct = 0
    all_labels = []
    for _, y_batch in loader:
        all_labels.extend(y_batch.tolist())
    all_labels_t = torch.tensor(all_labels, dtype=torch.long)
    weight = 1. / all_labels_t.bincount(minlength=model.num_classes).float().clamp(min=1)
    weight = weight.to(device)
    criterion = FocalLoss(gamma=2.0, weight=weight)
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits, features = model(x_batch)
        cls_loss = criterion(logits, y_batch)
        proto_cls_loss = prototype_classification_loss(
            model.prototypes, features, y_batch, model.prototype_labels)
        proto_div_loss = prototype_diversity_loss(model.prototypes)
        loss = cls_loss + alpha * proto_cls_loss + beta * proto_div_loss
        loss.backward(); optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def train_protonet(Xtr, ytr, Xva, yva, prototypes_per_class_list,
                   latent_dim=64, epochs=300, seed=42, patience=40):
    torch.manual_seed(seed); np.random.seed(seed)
    # num_classes is derived from the prototype layout (len of ppcl), so the
    # same trainer works for 3-class [1,2,1] and binary [1,1].
    model = ProtoNet(Xtr.shape[1], len(prototypes_per_class_list),
                     prototypes_per_class_list, latent_dim).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)
    best_val_acc, best_state, epochs_no_improve = 0.0, None, 0
    for ep in range(epochs):
        train_epoch_proto(model, loader, optimizer, 'cpu', alpha=0.1, beta=0.1)
        model.eval()
        with torch.no_grad():
            va_acc = (model(Xva_t)[0].argmax(1) == yva_t).float().mean().item()
        scheduler.step(va_acc)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.tensor(X, dtype=torch.float32))
        return F.softmax(logits, 1).cpu().numpy()


def predict_labels(model, X):
    return predict_proba(model, X).argmax(1)


# ══════════════════════════════════════════════════════════════════
#  Constants / data
# ══════════════════════════════════════════════════════════════════
DEVICE = 'cpu'
# Allow importing the leak-free ComBat harmonizer (combat_harmonizer.py lives
# one directory above scripts/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("DATA_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
OUT = os.path.join(ROOT, "experiments")
SEEDS = [42, 123, 2024]
N_FOLDS = 5
LATENT_DIM = 64
K_FEATURES = 150
INNER_VAL_FRAC = 0.25            # -> exact 60/20/20 train/val/test per outer fold
DEFAULT_PPCL = [1, 2, 1]          # deployed 3-class config
META = ["subject", "age", "male", "MMSE", "label", "mci_subtype", "site"]
CLINICAL = ["age", "male", "MMSE"]


# ══════════════════════════════════════════════════════════════════
#  Leak-free per-fold ComBat
# ══════════════════════════════════════════════════════════════════
def combat_harmonize_imaging(X_fit, site_fit, age_fit, male_fit,
                             X_apply, site_apply, age_apply, male_apply):
    """Leak-free empirical-Bayes ComBat on imaging features only.

    Fit ComBat on the training imaging block ``X_fit`` using the 8-site batch
    label (0 = ADNI as the reference, 1..7 = MCADI sub-centres) and age/sex
    covariates; then harmonize ``X_apply`` (the training block itself, a
    validation or a test partition) with the fitted per-batch parameters,
    mapping every sample onto the ADNI reference scale. The apply block never
    informs the fit, so there is no test leakage.

    Returns the harmonized ``X_apply`` (np.ndarray, same shape).
    """
    import pandas as pd
    from combat_harmonizer import ComBatHarmonizer
    n_feat = X_fit.shape[1]
    cols = [f"img{i}" for i in range(n_feat)]
    harm = ComBatHarmonizer()
    harm.fit(pd.DataFrame(np.asarray(X_fit, float), columns=cols),
             batch=np.asarray(site_fit).astype(int),
             covariates=pd.DataFrame({"age": np.asarray(age_fit, float),
                                      "male": np.asarray(male_fit, float)}),
             ref_batch=0)
    d = pd.DataFrame(np.asarray(X_apply, float), columns=cols)
    h = harm.transform(d,
                       batch=np.asarray(site_apply).astype(int),
                       covariates_ext=pd.DataFrame({
                           "age": np.asarray(age_apply, float),
                           "male": np.asarray(male_apply, float)})).values
    return h


def build_fold(df, cols, X_raw, y_all, tr_idx, te_idx, seed, fi):
    """Leak-free per-fold feature pipeline (shared by every experiment).

    Inner 60/20 (train/val) split of the outer training fold; per-fold ComBat on
    imaging (fit on inner-train, applied to inner-val AND outer-test so test
    rows never inform the harmonization); then SelectKBest(150) -> VIF(10) ->
    StandardScaler, all fit on the inner-training subset only.

    Returns a dict with the transformed blocks and bookkeeping used by the
    trainers / evaluators: Xs, yt, Xv_s, yv, Xte_s, y_te, n_kept, plus the raw
    (unscaled) post-VIF blocks Xtr_vif / Xv_vif / Xte_vif and the post-VIF
    feature names ``feat_names`` (for drop-column / ablation analyses).
    """
    from proto_vif_clean import vif_mask  # local import avoids an import cycle
    img_cols = [c for c in cols if c not in CLINICAL]
    cli_cols = [c for c in cols if c in CLINICAL]
    img_pos = [cols.index(c) for c in img_cols]
    cli_pos = [cols.index(c) for c in cli_cols]

    X_tr, y_tr = X_raw[tr_idx], y_all[tr_idx]
    X_te, y_te = X_raw[te_idx], y_all[te_idx]

    # Consistent inner split of the outer training fold.
    n = len(tr_idx)
    it, iv = train_test_split(np.arange(n), test_size=INNER_VAL_FRAC,
                              stratify=y_tr, random_state=seed * 100 + fi)
    Xt, Xv, yt, yv = X_tr[it], X_tr[iv], y_tr[it], y_tr[iv]

    # site / age / male aligned to the outer training block.
    if "site" in df.columns:
        site_tr = df["site"].values.astype(int)[tr_idx]
        site_it, site_iv = site_tr[it], site_tr[iv]
        site_te = df["site"].values.astype(int)[te_idx]
    else:
        # No site column -> skip ComBat (pass imaging through unchanged).
        site_it = site_iv = site_te = None
    age_tr = df["age"].values.astype(float)[tr_idx]
    male_tr = df["male"].values.astype(float)[tr_idx]
    age_it, age_iv = age_tr[it], age_tr[iv]
    male_it, male_iv = male_tr[it], male_tr[iv]
    age_te = df["age"].values.astype(float)[te_idx]
    male_te = df["male"].values.astype(float)[te_idx]

    Xt_img, Xv_img, Xte_img = Xt[:, img_pos], Xv[:, img_pos], X_te[:, img_pos]

    if site_it is not None:
        # Per-fold ComBat: fit on inner-train imaging; register val + test.
        Xt_img_h = combat_harmonize_imaging(
            Xt_img, site_it, age_it, male_it, Xt_img, site_it, age_it, male_it)
        Xv_img_h = combat_harmonize_imaging(
            Xt_img, site_it, age_it, male_it, Xv_img, site_iv, age_iv, male_iv)
        Xte_img_h = combat_harmonize_imaging(
            Xt_img, site_it, age_it, male_it, Xte_img, site_te, age_te, male_te)
    else:
        Xt_img_h, Xv_img_h, Xte_img_h = Xt_img, Xv_img, Xte_img

    def _rebuild(img_h, blk):
        return np.hstack([img_h, blk[:, cli_pos]]) if cli_pos else img_h

    Xt_full = _rebuild(Xt_img_h, Xt)
    Xv_full = _rebuild(Xv_img_h, Xv)
    Xte_full = _rebuild(Xte_img_h, X_te)

    sel = SelectKBest(f_classif, k=min(K_FEATURES, Xt_full.shape[1]))
    Xt_f = sel.fit_transform(Xt_full, yt)
    mask, n_kept = vif_mask(Xt_f)
    Xtr_vif = Xt_f[:, mask]
    Xv_vif = sel.transform(Xv_full)[:, mask]
    Xte_vif = sel.transform(Xte_full)[:, mask]
    sc = StandardScaler()
    Xs = sc.fit_transform(Xtr_vif)
    Xv_s = sc.transform(Xv_vif)
    Xte_s = sc.transform(Xte_vif)
    return {"Xs": Xs, "yt": yt, "Xv_s": Xv_s, "yv": yv,
            "Xte_s": Xte_s, "y_te": y_te, "n_kept": n_kept,
            "Xtr_vif": Xtr_vif, "Xv_vif": Xv_vif, "Xte_vif": Xte_vif,
            "feat_names": [str(x) for x in np.array(cols)[sel.get_support()][mask]]}


def load_data():
    df = pd.read_csv(os.path.join(OUT, "train_features.csv"))
    feats = [c for c in df.columns if c not in META]
    clinical = ["age", "male", "MMSE"]
    imaging = [c for c in feats if c not in clinical]
    print(f"[data] rows={df.shape[0]}, feats={len(feats)} "
          f"(imaging={len(imaging)}, clinical={len(clinical)})")
    return df, feats, imaging, clinical


def prep(X_raw, y, k_feat):
    sel = SelectKBest(f_classif, k=min(k_feat, X_raw.shape[1]))
    Xf = sel.fit_transform(X_raw, y)
    sc = StandardScaler()
    Xs = sc.fit_transform(Xf)
    return Xs


def eval_preds(y_true, y_pred, y_proba):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        if len(np.unique(y_true)) > 1:
            out["auc_ovr_macro"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        pass
    return out


# ── DeLong (paired, binary) ────────────────────────────────────────
def _midrank(x):
    order = np.argsort(x); Z = x[order]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N - 1 and Z[j + 1] == Z[i]:
            j += 1
        T[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    return T[np.argsort(order)]


def delong_2models(posA, negA, posB, negB):
    """Paired DeLong test for AUC of model A vs model B (both score positives high)."""
    m, n = len(posA), len(negA)
    rA = _midrank(np.concatenate([posA, negA])); rA_p = rA[:m]; rA_n = rA[m:]
    rB = _midrank(np.concatenate([posB, negB])); rB_p = rB[:m]; rB_n = rB[m:]
    v10_A = (rA_p - (m + 1) / 2) / n; v01_A = (rA_n - (n + 1) / 2) / m
    v10_B = (rB_p - (m + 1) / 2) / n; v01_B = (rB_n - (n + 1) / 2) / m
    # 2x2 covariance blocks over positives (S10) and negatives (S01)
    S10 = np.cov(np.stack([v10_A, v10_B]))          # 2x2
    S01 = np.cov(np.stack([v01_A, v01_B]))          # 2x2
    var = (S10[0, 0] - 2 * S10[0, 1] + S10[1, 1]) / n \
        + (S01[0, 0] - 2 * S01[0, 1] + S01[1, 1]) / m
    aucA = (np.sum(rA_p) - m * (m + 1) / 2) / (m * n)
    aucB = (np.sum(rB_p) - m * (m + 1) / 2) / (m * n)
    if var <= 0 or not np.isfinite(var):
        return float(aucA), float(aucB), 0.0, 1.0
    z = (aucA - aucB) / math.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(aucA), float(aucB), float(z), float(p)


def bootstrap_auc_ci(y_true, y_pos, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true); s = np.asarray(y_pos)
    if len(np.unique(y)) < 2:
        return (np.nan, np.nan, np.nan)
    ip = np.where(y == 1)[0]; ineg = np.where(y == 0)[0]
    aucs = []
    for _ in range(n_boot):
        bp = rng.choice(ip, len(ip), replace=True)
        bn = rng.choice(ineg, len(ineg), replace=True)
        yy = np.concatenate([y[bp], y[bn]]); ss = np.concatenate([s[bp], s[bn]])
        try:
            aucs.append(roc_auc_score(yy, ss))
        except Exception:
            pass
    if not aucs:
        return (np.nan, np.nan, np.nan)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ══════════════════════════════════════════════════════════════════
#  EXP A: nested-CV main (deployed [1,2,1])
#  also collects probas for significance + ECE
# ══════════════════════════════════════════════════════════════════
def run_nested_cv_main(df, feats):
    """Nested-CV main experiment for the proposed ProtoNet.

    Outer 5-fold x 3 seeds (SEEDS) under the leak-free protocol
    (per-fold ComBat fit on the inner-training partition via the site 0-7 batch
    label -> SelectKBest(150) -> VIF(10) -> StandardScaler, all fit on the
    inner-training subset only; early stopping on the inner validation subset).
    Returns per-fold summary rows and aggregated 3-class / binary metrics.
    """
    y_all = df["label"].values.astype(int)
    X_raw = df[feats].values
    summ_rows = []
    proto_store = []          # per seed: y_global, proto_proba (3-class)
    for si, seed in enumerate(SEEDS):
        print(f"\n{'='*60}\nSeed {seed} ({si+1}/{len(SEEDS)})  MAIN nested-CV\n{'='*60}")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        folds = list(skf.split(X_raw, y_all))
        p_probas, y_glob = [], []
        for fi, (tr_idx, te_idx) in enumerate(folds):
            # Leak-free per-fold pipeline: inner 60/20 split -> per-fold ComBat
            # (fit on inner-train via site 0-7, applied to val + test) ->
            # SelectKBest(150) -> VIF(10) -> StandardScaler, all fit on the
            # inner training subset only; early stopping uses the inner
            # validation subset, never the test fold.
            f = build_fold(df, feats, X_raw, y_all, tr_idx, te_idx, seed, fi)
            pnet = train_protonet(f["Xs"], f["yt"], f["Xv_s"], f["yv"], DEFAULT_PPCL, seed=seed)
            ppr = predict_labels(pnet, f["Xte_s"]); pp = predict_proba(pnet, f["Xte_s"])
            ev = eval_preds(f["y_te"], ppr, pp)
            ev.update({"seed": seed, "fold": fi, "model": "ProtoNet"})
            summ_rows.append(ev)
            p_probas.append(pp); y_glob.append(f["y_te"])
            print(f"  [ProtoNet] s={seed} f={fi} AUC={ev['auc_ovr_macro']:.4f} acc={ev['accuracy']:.4f}")
        y_glob = np.concatenate(y_glob)
        p_proba = np.concatenate(p_probas)
        proto_store.append({"y": y_glob, "proba": p_proba})

    summ_df = pd.DataFrame(summ_rows)
    summ_df.to_csv(os.path.join(OUT, "main_nested_cv_summary.csv"), index=False)

    p = summ_df[summ_df["model"] == "ProtoNet"]
    agg = p[["accuracy", "balanced_accuracy", "f1_macro", "auc_ovr_macro"]].agg(["mean", "std"])
    main3 = {k: (round(agg.loc["mean", k], 4), round(agg.loc["std", k], 4))
             for k in ["accuracy", "balanced_accuracy", "f1_macro", "auc_ovr_macro"]}
    print("\n=== MAIN nested-CV (ProtoNet, [1,2,1]) ===")
    print(agg.to_string())

    yb_pool, pb_pool = [], []
    for st in proto_store:
        y = st["y"]; m = (y != 1)
        yb_pool.append((y[m] == 2).astype(int)); pb_pool.append(st["proba"][m][:, 2])
    yb_pool = np.concatenate(yb_pool); pb_pool = np.concatenate(pb_pool)
    bin_auc = roc_auc_score(yb_pool, pb_pool)
    try:
        _, lo, hi = bootstrap_auc_ci(yb_pool, pb_pool)
    except Exception:
        lo, hi = float('nan'), float('nan')
    print(f"Binary (HC vs AD) AUC = {bin_auc:.4f} CI({lo:.3f}-{hi:.3f})")
    return summ_df, main3, (bin_auc, lo, hi)





def delong_binary(probA, probB, ybin):
    """ybin: 0/1 (1=positive/AD). probA/B: P(AD)."""
    posA = probA[ybin == 1]; negA = probA[ybin == 0]
    posB = probB[ybin == 1]; negB = probB[ybin == 0]
    return delong_2models(posA, negA, posB, negB)


# ══════════════════════════════════════════════════════════════════
#  EXP C: K-ablation (symmetric [K,K,K])
# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("ProtoNet (latent=64) -- main nested-CV experiment")
    print("=" * 70)
    df, feats, imaging, clinical = load_data()

    print("\n[1/1] Nested-CV MAIN (deployed [1,2,1]) ...")
    summ_df, main3, bin_res = run_nested_cv_main(df, feats)
    print(f"  -> 3-class Acc={main3['accuracy']}, AUC={main3['auc_ovr_macro']}")
    print(f"  -> binary AUC={bin_res[0]:.4f} CI({bin_res[1]:.3f}-{bin_res[2]:.3f})")

    consolidated = {
        "architecture": "ProtoNet(latent=64, BN+GELU+FocalLoss+proto_supervision)",
        "data_pool": "train_features.csv (training cohorts A + B)",
        "main_nested_cv_3class": main3,
        "main_binary_AUC": {"mean": round(bin_res[0], 4),
                             "ci_lo": round(bin_res[1], 3), "ci_hi": round(bin_res[2], 3)},
        "note": ("This script reproduces the proposed ProtoNet evaluation "
                 "under the leak-free nested-CV protocol. Baseline comparisons "
                 "and ablations are not part of this release."),
    }
    with open(os.path.join(OUT, "all_results_proto.json"), "w") as f:
        json.dump(consolidated, f, indent=4,
                  default=lambda o: (o.item() if hasattr(o, "item")
                                    else (o.tolist() if hasattr(o, "tolist")
                                          else str(o))))
    print("\n" + "=" * 70)
    print("DONE. Results ->", os.path.join(OUT, "all_results_proto.json"))
    print("=" * 70)



if __name__ == "__main__":
    main()
