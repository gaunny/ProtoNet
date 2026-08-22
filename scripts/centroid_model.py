"""Centroid-initialized ProtoNet trainer for display models.

Same training protocol as final_compare.train_tuned (LR/WD/ALPHA/PATIENCE/
EPOCHS/DROPOUT, FocalLoss, validation early stopping), but prototypes are
initialized at the per-class embedding centroids of the training subset instead
of random noise. This places the learned prototypes inside their class clusters
(the AD prototype lands near AD samples in absolute distance) at a small
accuracy cost.

Used by feature_importance_retrain.py and external_validation.py for the centroid-
initialized display variant.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

from experiments_proto import ProtoNet, prototype_classification_loss, \
    prototype_diversity_loss
from final_compare import LR, WD, ALPHA, PATIENCE, EPOCHS, DROPOUT, FocalLoss


def proto_group_loss(prototypes, features, proto_target, temperature=1.0):
    """Push each sample toward its OWN prototype group only.

    proto_target[i] = index of the prototype that sample i must be close to
    (per-class prototypes; for a multi-prototype class like MCI the two
    prototypes are bound to the two KMeans sub-clusters).
    """
    sim = torch.mm(features, prototypes.t()) / temperature
    mask = proto_target.unsqueeze(1) == torch.arange(len(prototypes),
                                                    device=prototypes.device).unsqueeze(0)
    pos = sim[mask]; neg = sim[~mask]
    return (-torch.log(torch.sigmoid(pos) + 1e-6).mean()
            - torch.log(1 - torch.sigmoid(neg) + 1e-6).mean())


def centroid_init(model, Xs, yt, noise=0.05, rng_seed=7, mode="directional"):
    """Initialize prototypes at class embedding centroids.

    For classes with >1 prototype (e.g. MCI in [1,2,1]), separate them along
    the HC->AD axis so the prototypes represent distinct disease-stage
    sub-regions (one leaning toward HC, one toward AD) instead of collapsing
    onto the class centroid. mode='directional' shifts the k-th prototype
    toward the k-th ordered anchor; 'kmeans' uses KMeans sub-clusters.
    """
    from sklearn.cluster import KMeans
    with torch.no_grad():
        z0 = model.encoder(torch.tensor(Xs, dtype=torch.float32)).numpy()
    pl = model.prototype_labels.numpy()
    rng = np.random.default_rng(rng_seed)
    new = np.zeros_like(model.prototypes.detach().numpy())
    # anchor centroids along the disease-progression axis (HC=0, MCI=1, AD=2)
    anchors = {}
    for c in range(model.num_classes):
        anchors[c] = z0[yt == c].mean(0)
    for c in range(model.num_classes):
        idx = [i for i, p in enumerate(pl) if p == c]
        zc = z0[yt == c]
        if len(idx) > 1 and mode == "directional" and c + 1 in anchors and c - 1 in anchors:
            lo = anchors[c - 1]   # toward the milder stage (HC)
            hi = anchors[c + 1]   # toward the more severe stage (AD)
            span = hi - lo
            for j, k in zip(idx, range(len(idx))):
                frac = (k + 1) / (len(idx) + 1)  # 1/3, 2/3 along the axis
                new[j] = lo + frac * span + rng.normal(0, noise, span.shape)
        elif len(idx) > 1 and mode == "kmeans" and len(zc) >= len(idx):
            km = KMeans(n_clusters=len(idx), random_state=rng_seed, n_init=10)
            km.fit(zc)
            for j, s in zip(idx, range(len(idx))):
                new[j] = km.cluster_centers_[s] + rng.normal(0, noise, km.cluster_centers_[s].shape)
        else:
            cent = zc.mean(0)
            for j in idx:
                new[j] = cent + rng.normal(0, noise, cent.shape)
    with torch.no_grad():
        model.prototypes.copy_(torch.tensor(new, dtype=torch.float32))
    return model


def train_centroid(Xs, yt, Xv_s, yv, ppcl, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ProtoNet(Xs.shape[1], len(ppcl), ppcl, 64).cpu()
    model.encoder = nn.Sequential(
        nn.Linear(Xs.shape[1], 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(DROPOUT),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Linear(128, 64))
    model = centroid_init(model, Xs, yt)
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


def train_centroid_subclust(Xs, yt, Xv_s, yv, ppcl, seed, subclust_class=1, n_sub=2):
    """Centroid-initialized training with MCI sub-cluster supervision.

    The two MCI prototypes are each bound to one KMeans sub-cluster of the MCI
    training embeddings, so they learn to represent distinct MCI subgroups
    (e.g. HC-leaning vs AD-leaning) instead of collapsing onto the MCI centroid.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model = ProtoNet(Xs.shape[1], len(ppcl), ppcl, 64).cpu()
    model.encoder = nn.Sequential(
        nn.Linear(Xs.shape[1], 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(DROPOUT),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Linear(128, 64))
    model = centroid_init(model, Xs, yt)
    pl = model.prototype_labels.numpy()

    # per-sample prototype-group target: class prototypes; MCI split by sub-cluster
    with torch.no_grad():
        z0 = model.encoder(torch.tensor(Xs, dtype=torch.float32)).numpy()
    mci = yt == subclust_class
    km = KMeans(n_clusters=n_sub, random_state=seed, n_init=10).fit(z0[mci])
    sub = np.full(len(yt), 0)
    sub[mci] = km.labels_
    idx_by_class = {c: [i for i, p in enumerate(pl) if p == c] for c in range(len(ppcl))}
    proto_target = np.zeros(len(yt), dtype=int)
    for i in range(len(yt)):
        c = yt[i]
        if c == subclust_class:
            proto_target[i] = idx_by_class[c][sub[i]]
        else:
            proto_target[i] = idx_by_class[c][0]

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    Xs_t = torch.tensor(Xs, dtype=torch.float32); yt_t = torch.tensor(yt, dtype=torch.long)
    pt_t = torch.tensor(proto_target, dtype=torch.long)
    Xv_t = torch.tensor(Xv_s, dtype=torch.float32); yv_t = torch.tensor(yv, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xs_t, yt_t, pt_t), batch_size=32, shuffle=True)
    counts = yt_t.bincount(minlength=len(ppcl)).float().clamp(min=1)
    crit = FocalLoss(gamma=2.0, weight=1. / counts)
    best_acc, best_state, no_imp = 0.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb, pb in loader:
            opt.zero_grad()
            logits, z = model(xb)
            loss = crit(logits, yb) + ALPHA * proto_group_loss(model.prototypes, z, pb) + \
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
