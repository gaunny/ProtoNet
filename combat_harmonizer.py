"""
ComBatHarmonizer
================
A self-contained ComBat (empirical-Bayes) harmonizer with covariate support and a
per-batch *transform* method for external cohorts.

Design
------
* ComBat is FIT on the training cohorts with a fixed reference batch
  (default: the first batch label; the pipeline uses ADNI = site 0).
* Biological covariates (age, sex) are regressed out before the batch
  adjustment and added back afterwards, so they are preserved.
* `transform` adjusts each sample using the batch parameters of its own site
  label (gamma*, delta*, stand_mean, delta_hat estimated at fit time) and maps
  the sample onto the reference scale. Samples whose batch label was never seen
  at fit time (e.g. a fully held-out external cohort) are registered as an
  overall block to the reference scale.
* Nothing from the transformed block ever informs the fitted parameters,
  so validation / test / external data stay fully held out.

The algorithm follows Johnson et al., Biostatistics 2007 (empirical-Bayes
ComBat) with a fixed reference batch.
"""
import numpy as np
import pandas as pd

_EPS = 1e-8


def _aprior(gamma_hat):
    m = gamma_hat.mean(axis=0)
    s2 = np.where(gamma_hat.var(axis=0, ddof=1) < _EPS, _EPS, gamma_hat.var(axis=0, ddof=1))
    return (2.0 * s2 + m**2) / s2


def _bprior(delta_hat):
    m = delta_hat.mean(axis=0)
    s2 = np.where(delta_hat.var(axis=0, ddof=1) < _EPS, _EPS, delta_hat.var(axis=0, ddof=1))
    return (2.0 * s2 + m**2) / s2


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sumsq, n, a, b):
    return (0.5 * sumsq + b) / (n / 2.0 + a - 1.0)


class ComBatHarmonizer:
    """Empirical-Bayes ComBat with covariates, a reference batch, and per-batch
    transform for validation / test / external data."""

    def __init__(self):
        self.fitted_ = False

    def fit(self, X, batch, covariates=None, ref_batch=None):
        """Fit ComBat on training data.

        X          : pd.DataFrame, samples x features.
        batch      : array-like of length n_samples (integer batch labels;
                     site 0 = ADNI, 1..7 = MCADI sub-centres in the pipeline).
        covariates : pd.DataFrame, samples x k biological covariates (age, sex).
        ref_batch  : batch label used as the reference scale (default: None,
                     i.e. the pooled statistics of all training batches).
        """
        X = X.copy()
        n_samples, n_features = X.shape
        batches = np.asarray(batch)
        batch_labels = np.unique(batches)
        n_batches = len(batch_labels)
        if ref_batch is not None and ref_batch not in batch_labels:
            raise ValueError(f"ref_batch {ref_batch} not in batch labels {batch_labels}")
        self.ref_batch_ = ref_batch

        if covariates is not None:
            cov = np.asarray(covariates, dtype=float)
            if cov.ndim == 1:
                cov = cov.reshape(-1, 1)
            design = np.column_stack([np.ones(n_samples), cov])
        else:
            design = np.ones((n_samples, 1))
        n_cov = design.shape[1]

        Xv = X.values.astype(float)

        # --- regress out covariates per feature ---
        beta_hat = np.zeros((n_cov, n_features))
        resid_all = np.zeros_like(Xv)
        for j in range(n_features):
            y = Xv[:, j]
            XtX_inv = np.linalg.pinv(design.T @ design + 1e-8 * np.eye(n_cov))
            beta = XtX_inv @ (design.T @ y)
            beta_hat[:, j] = beta
            resid_all[:, j] = y - design @ beta

        # --- per-batch residual mean / variance ---
        stand_mean = np.zeros((n_batches, n_features))
        delta_hat = np.zeros((n_batches, n_features))
        for bi, b in enumerate(batch_labels):
            r = resid_all[batches == b]
            stand_mean[bi] = r.mean(axis=0)
            v = r.var(axis=0, ddof=1)
            delta_hat[bi] = np.where(v < _EPS, _EPS, v)

        # --- reference statistics ---
        if ref_batch is None:
            grand_mean_ref = resid_all.mean(axis=0)
            vref = resid_all.var(axis=0, ddof=0)
            var_ref = np.where(vref < _EPS, _EPS, vref)
        else:
            ri = int(np.where(batch_labels == ref_batch)[0][0])
            grand_mean_ref = stand_mean[ri]
            var_ref = delta_hat[ri]

        # --- standardize within batch ---
        Z = np.zeros_like(Xv)
        for bi, b in enumerate(batch_labels):
            mask = batches == b
            Z[mask] = (resid_all[mask] - stand_mean[bi]) / np.sqrt(delta_hat[bi])

        gamma_hat = np.zeros((n_batches, n_features))
        delta_log = np.zeros((n_batches, n_features))
        for bi in range(n_batches):
            Zb = Z[batches == batch_labels[bi]]
            gamma_hat[bi] = Zb.mean(axis=0)
            zv = Zb.var(axis=0, ddof=1)
            delta_log[bi] = np.log(np.where(zv < _EPS, _EPS, zv))

        # --- empirical-Bayes shrinkage on non-reference batches ---
        ref_idx = None
        if ref_batch is not None:
            ref_idx = int(np.where(batch_labels == ref_batch)[0][0])
        non_ref = [bi for bi in range(n_batches) if bi != ref_idx]
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.zeros_like(delta_log)
        if ref_idx is not None:
            gamma_star[ref_idx] = 0.0
            delta_star[ref_idx] = 0.0
        if len(non_ref) >= 2:
            g = gamma_hat[non_ref]
            d = delta_log[non_ref]
            gbar, gvar = g.mean(axis=0), g.var(axis=0, ddof=1)
            dbar, dvar = d.mean(axis=0), d.var(axis=0, ddof=1)
            for bi in non_ref:
                nb = np.sum(batches == batch_labels[bi])
                gamma_star[bi] = _postmean(gamma_hat[bi], gbar, nb, np.maximum(gvar, _EPS), np.maximum(dvar, _EPS))
                lam = (nb * dvar + 2.0 * dvar**2) / (nb + 2.0 * dvar)
                delta_star[bi] = (nb * delta_log[bi] + lam * dbar) / (nb + lam)
        else:
            for bi in non_ref:
                gamma_star[bi] = gamma_hat[bi]
                delta_star[bi] = delta_log[bi]

        # --- store ---
        self.features_ = list(X.columns)
        self.batch_labels_ = batch_labels
        self.n_cov_ = n_cov
        self.beta_hat_ = beta_hat
        self.stand_mean_ = stand_mean
        self.delta_hat_ = delta_hat
        self.gamma_star_ = gamma_star
        self.delta_star_ = delta_star
        self.grand_mean_ref_ = grand_mean_ref
        self.var_ref_ = var_ref
        self.covariate_cols_ = list(covariates.columns) if covariates is not None else None
        self.fitted_ = True
        return self

    def transform(self, X_ext, batch=None, covariates_ext=None):
        """Harmonize data to the reference scale.

        X_ext        : pd.DataFrame, samples x features (same feature names).
        batch        : optional per-sample batch labels. Samples with a label
                       seen at fit time are adjusted with that batch's
                       parameters; samples with an unseen label (or batch=None)
                       are registered as an overall block to the reference.
        covariates_ext : pd.DataFrame, samples x k, same columns as in fit.
        """
        if not self.fitted_:
            raise RuntimeError("ComBatHarmonizer must be fit before transform.")
        X_ext = X_ext[self.features_].copy()
        n_ext = X_ext.shape[0]
        if covariates_ext is not None:
            cov = np.asarray(covariates_ext[self.covariate_cols_], dtype=float)
            if cov.ndim == 1:
                cov = cov.reshape(-1, 1)
            design_ext = np.column_stack([np.ones(n_ext), cov])
        else:
            design_ext = np.ones((n_ext, 1))
        Xv = X_ext.values.astype(float)
        resid = Xv - design_ext @ self.beta_hat_
        out = np.zeros_like(Xv)
        rstd = np.sqrt(self.var_ref_)

        if batch is None:
            out = (resid - resid.mean(axis=0)) / np.sqrt(
                np.where(resid.var(axis=0, ddof=1) < _EPS, _EPS, resid.var(axis=0, ddof=1))
            ) * rstd + self.grand_mean_ref_
        else:
            b = np.asarray(batch)
            for label in np.unique(b):
                mask = b == label
                if label in self.batch_labels_:
                    bi = int(np.where(self.batch_labels_ == label)[0][0])
                    Z = (resid[mask] - self.stand_mean_[bi]) / np.sqrt(self.delta_hat_[bi])
                    Z_adj = (Z - self.gamma_star_[bi]) / np.exp(self.delta_star_[bi])
                    out[mask] = Z_adj * rstd + self.grand_mean_ref_
                else:
                    r = resid[mask]
                    sd = np.sqrt(np.where(r.var(axis=0, ddof=1) < _EPS, _EPS, r.var(axis=0, ddof=1)))
                    out[mask] = (r - r.mean(axis=0)) / sd * rstd + self.grand_mean_ref_

        out = out + design_ext @ self.beta_hat_
        return pd.DataFrame(out, columns=self.features_, index=X_ext.index)

    # ----------------------------------------------------------------- validate
    def validate_against_pycombat(self, X, batch, covariates=None):
        """Fit pycombat (no-transform variant) on same data; compare harmonized means."""
        from combat.pycombat import pycombat

        Xt = X[self.features_].T  # features x samples for pycombat
        batches = np.asarray(batch)
        if covariates is not None:
            cov_names = list(covariates.columns)
            cov_df = covariates[cov_names].copy()
            cov_df.index = Xt.columns
            harmonized = pycombat(data=Xt, batch=batches, covariates=cov_df)
        else:
            harmonized = pycombat(data=Xt, batch=batches)
        harmonized = harmonized.T  # back to samples x features
        ext = self.transform(X, batch=batches, covariates_ext=covariates)
        diff = np.abs(harmonized.values - ext.values)
        return {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "n_features": len(self.features_),
        }
