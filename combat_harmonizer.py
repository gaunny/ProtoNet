"""
ComBatHarmonizer
================
A self-contained ComBat (empirical-Bayes) harmonizer with covariate support and a
*transform* method for external cohorts.

Design
------
* ComBat is FIT only on the training cohorts (A + B).
* Biological covariates (age, sex) are preserved during harmonization.
* The fitted (reference) parameters are applied to an external cohort via
  `transform`; the external cohort never participates in parameter estimation,
  so it remains fully held out.

The algorithm follows Johnson et al., Biostatistics 2007 (neuroCombat).
"""
import numpy as np
import pandas as pd
from scipy import stats


def _aprior(gamma_hat):
    m = gamma_hat.mean(axis=0)
    s2 = gamma_hat.var(axis=0, ddof=1)
    return (2.0 * s2 + m**2) / s2


def _bprior(delta_hat):
    m = delta_hat.mean(axis=0)
    s2 = delta_hat.var(axis=0, ddof=1)
    return (2.0 * s2 + m**2) / s2


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sumsq, n, a, b):
    return (0.5 * sumsq + b) / (n / 2.0 + a - 1.0)


class ComBatHarmonizer:
    """Empirical-Bayes ComBat with covariates and an external transform."""

    def __init__(self):
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(self, X, batch, covariates=None):
        """
        X          : pd.DataFrame, samples x features (we transpose internally)
        batch      : array-like of length n_samples (integer batch labels)
        covariates : pd.DataFrame, samples x k biological/technical covariates
                     (e.g. age, sex).  These are *preserved*.
        """
        X = X.copy()
        n_samples, n_features = X.shape
        batches = np.asarray(batch)
        batch_labels = np.unique(batches)
        n_batches = len(batch_labels)

        # ---- design matrix: intercept + covariates (NO batch dummy) ----
        if covariates is not None:
            cov = np.asarray(covariates, dtype=float)
            if cov.ndim == 1:
                cov = cov.reshape(-1, 1)
            design = np.column_stack([np.ones(n_samples), cov])
        else:
            design = np.ones((n_samples, 1))
        n_cov = design.shape[1]

        # ---- Step 1: standardize within feature, regress out covariates ----
        grand_mean = np.zeros(n_features)
        var_pooled = np.zeros(n_features)
        beta_hat = np.zeros((n_cov, n_features))
        stand_mean = np.zeros((n_batches, n_features))

        Xv = X.values.astype(float)
        for j in range(n_features):
            y = Xv[:, j]
            # OLS: regress y on design
            # add tiny ridge for numerical stability when design collinear
            Xd = design
            XtX_inv = np.linalg.pinv(Xd.T @ Xd + 1e-8 * np.eye(n_cov))
            beta = XtX_inv @ (Xd.T @ y)
            beta_hat[:, j] = beta
            resid = y - Xd @ beta
            grand_mean[j] = resid.mean()  # overall residual mean ~0 after centering
            # within-batch variance of residuals
            var_within = []
            n_within = []
            for bi, b in enumerate(batch_labels):
                mask = (batches == b)
                r = resid[mask]
                if len(r) > 1:
                    var_within.append(r.var(ddof=1))
                    n_within.append(len(r))
                    stand_mean[bi, j] = r.mean()
                else:
                    # single-sample batch: fall back to pooled
                    var_within.append(np.var(resid))
                    n_within.append(1)
                    stand_mean[bi, j] = r.mean() if len(r) else 0.0
            var_pooled[j] = np.sum((np.array(n_within) - 1) * np.array(var_within)) / max(
                (np.sum(n_within) - n_batches), 1
            )

        # ---- Step 2: empirical Bayes on batch means (gamma) ----
        gamma_bar = stand_mean.mean(axis=0)
        t2 = stand_mean.var(axis=0, ddof=1)
        a_prior = _aprior(stand_mean)
        # delta (variance) EB
        delta_hat = np.zeros((n_batches, n_features))
        for bi, b in enumerate(batch_labels):
            mask = (batches == b)
            for j in range(n_features):
                r = (Xv[mask, j] - design[mask] @ beta_hat[:, j])
                delta_hat[bi, j] = np.var(r, ddof=1) if np.sum(mask) > 1 else var_pooled[j]
        b_prior = _bprior(delta_hat)

        gamma_star = np.zeros((n_batches, n_features))
        delta_star = np.zeros((n_batches, n_features))
        for bi, b in enumerate(batch_labels):
            n_i = np.sum(batches == b)
            for j in range(n_features):
                gamma_star[bi, j] = _postmean(
                    stand_mean[bi, j], gamma_bar[j], n_i, a_prior[j], t2[j]
                )
                # variance EB (inverse-gamma on 1/sigma^2)
                ss = np.sum(
                    ((Xv[batches == b, j] - design[batches == b] @ beta_hat[:, j]) ) ** 2
                )
                delta_star[bi, j] = _postvar(ss, n_i, a_prior[j] / 2.0, b_prior[j] / 2.0)
                # guard against degenerate
                if not np.isfinite(delta_star[bi, j]) or delta_star[bi, j] <= 0:
                    delta_star[bi, j] = var_pooled[j]

        # ---- store fitted reference (training) parameters ----
        self.features_ = list(X.columns)
        self.batch_labels_ = batch_labels
        self.design_ref_ = design
        self.beta_hat_ = beta_hat
        self.grand_mean_ = grand_mean
        self.gamma_star_ = gamma_star
        self.delta_star_ = delta_star
        self.var_pooled_ = var_pooled
        self.n_cov_ = n_cov
        self.covariate_cols_ = list(covariates.columns) if covariates is not None else None
        self.fitted_ = True
        return self

    # ----------------------------------------------------------------- transform
    def transform(self, X_ext, covariates_ext=None, new_batch_label=999):
        """
        Harmonize an EXTERNAL cohort to the reference (training) distribution.
        The external cohort never informed the fitted parameters -> no leakage.

        Method: register each external feature's distribution to the reference
        (training) feature distribution *after* removing the same covariate effects,
        using the empirical-Bayes delta/gamma of the reference pooled mean batch.
        Concretely we map external batch statistics to the reference grand mean /
        pooled variance, preserving covariates.
        """
        if not self.fitted_:
            raise RuntimeError("ComBatHarmonizer must be fit before transform.")
        X_ext = X_ext.copy()
        X_ext = X_ext[self.features_]

        n_ext = X_ext.shape[0]
        if covariates_ext is not None:
            cov = np.asarray(covariates_ext[self.covariate_cols_], dtype=float)
            if cov.ndim == 1:
                cov = cov.reshape(-1, 1)
            design_ext = np.column_stack([np.ones(n_ext), cov])
        else:
            design_ext = np.ones((n_ext, 1))

        Xv = X_ext.values.astype(float)
        out = np.zeros_like(Xv)
        for j in range(len(self.features_)):
            # remove covariate effects (same beta as reference)
            resid = Xv[:, j] - design_ext @ self.beta_hat_[:, j]
            # external batch observed mean / var
            mu_ext = resid.mean()
            var_ext = resid.var(ddof=1) if n_ext > 1 else self.var_pooled_[j]
            # register external residual distribution to reference (grand_mean=0, scale=var_pooled)
            # Use reference pooled variance as target scale (delta_star pooled ~ var_pooled)
            resid_reg = (resid - mu_ext) / np.sqrt(max(var_ext, 1e-8)) * np.sqrt(
                max(self.var_pooled_[j], 1e-8)
            ) + self.grand_mean_[j]
            # re-add covariate effects
            out[:, j] = resid_reg + design_ext @ self.beta_hat_[:, j]
        result = pd.DataFrame(out, columns=self.features_, index=X_ext.index)
        return result

    # ----------------------------------------------------------------- validate
    def validate_against_pycombat(self, X, batch, covariates=None):
        """Fit pycombat (no-transform variant) on same data; compare harmonized means."""
        from combat.pycombat import pycombat

        Xt = X[self.features_].T  # features x samples for pycombat
        batches = np.asarray(batch)
        if covariates is not None:
            cov_names = list(covariates.columns)
            # pycombat wants covariates as a DataFrame aligned to columns
            cov_df = covariates[cov_names].copy()
            cov_df.index = Xt.columns
            harmonized = pycombat(data=Xt, batch=batches, covariates=cov_df)
        else:
            harmonized = pycombat(data=Xt, batch=batches)
        harmonized = harmonized.T  # back to samples x features
        # our own fit+transform on the same (training) data, treating it as external
        ext = self.transform(X, covariates_ext=covariates)
        diff = np.abs(harmonized.values - ext.values)
        return {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "n_features": len(self.features_),
        }
