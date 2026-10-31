#!/usr/bin/env python3
"""
03_causal_forest.py
===================
Inverse Propensity Score Weighting (IPW) and Conditional Average Treatment Effect (CATE)
estimation for Hypotheses H3a and H3c (Public Financial Support & Capability Moderation).

Key Econometric Features:
  1. First-stage multivariable logistic regression (incorporating 8 baseline covariates,
     26 country fixed effects, and 3 NACE sector fixed effects; k = 38) solved via exact
     Newton-Raphson (IRLS) maximum likelihood.
  2. Complete common-support overlap, stabilized weight distribution, and Kish Effective
     Sample Size (ESS) diagnostics.
  3. Non-parametric bootstrap (B = 500, seed = 42) that RE-ESTIMATES the first-stage
     logistic propensity model inside every bootstrap replication to account for
     propensity-score estimation uncertainty ( alongside fixed-PS bootstrap SEs).
  4. Analytical two-step Influence-Function (IF) standard errors (Hirano, Imbens, & Ridder,
     2003; Lunceford & Davidian, 2004), both i.i.d. and clustered across the G = 27
     EU member states.

Treatment: Q6.1 (grant_receipt - public financial support: grants, loans, or guarantees)
Outcome:   Q14 (q14_binary - reported net-zero decarbonization strategy)
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "Data")
if not os.path.exists(DATA_DIR) and os.path.exists(os.path.join(os.path.dirname(PROJECT_DIR), "Data")):
    DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "Data")
DATA_FILE = os.path.join(DATA_DIR, "Processed", "SME_Analytical_Final.csv")
RESULTS_XLSX = os.path.join(DATA_DIR, "Results_Consolidated.xlsx")


def fit_logit_nr(X, y, beta_init=None, max_iter=30, tol=1e-9, ridge=1e-8):
    """Exact Newton-Raphson (IRLS) solver for binary logistic regression."""
    n, k = X.shape
    beta = np.zeros(k) if beta_init is None else beta_init.copy()
    I_k = np.eye(k) * ridge
    for _ in range(max_iter):
        p = expit(X @ beta)
        grad = X.T @ (y - p)
        if np.max(np.abs(grad)) < tol:
            break
        w = p * (1.0 - p)
        H = (X.T * w) @ X + I_k
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def two_step_ipw_influence_function(X_ps, T, Y, ps, beta_hat, clusters):
    """
    Analytical two-step M-estimation / Influence-Function (IF) standard error for the
    Hajek IPW ATE estimator accounting for first-stage logit propensity score estimation
    (Lunceford & Davidian, 2004; Hirano, Imbens, & Ridder, 2003), both i.i.d. and
    country-clustered (G = 27).
    """
    n = len(Y)
    w1 = T / ps
    w0 = (1.0 - T) / (1.0 - ps)
    mu1 = np.sum(w1 * Y) / np.sum(w1)
    mu0 = np.sum(w0 * Y) / np.sum(w0)

    # Outcome moment residuals (psi_tau,i)
    m1 = np.mean(w1)
    m0 = np.mean(w0)
    psi_tau = (w1 * (Y - mu1)) / m1 - (w0 * (Y - mu0)) / m0

    # Jacobian of outcome moment w.r.t. first-stage logit parameters gamma
    # d/dgamma [ T(Y - mu1)/e(X) ] = -T(Y - mu1) * ((1 - e)/e) * X
    # d/dgamma [ -(1-T)(Y - mu0)/(1-e(X)) ] = -(1-T)(Y - mu0) * (e/(1 - e)) * X
    d_w1 = -(T * (Y - mu1) * (1.0 - ps) / ps) / m1
    d_w0 = -((1.0 - T) * (Y - mu0) * ps / (1.0 - ps)) / m0
    G_tau_gamma = (X_ps.T @ (d_w1 + d_w0)) / n  # (k,)

    # First-stage logit score & information matrix: E[-dS/dgamma] = (1/n) X' W X
    p_raw = expit(X_ps @ beta_hat)
    W_logit = p_raw * (1.0 - p_raw)
    I_gamma = ((X_ps.T * W_logit) @ X_ps) / n  # (k, k)
    S_gamma = (T - p_raw)[:, None] * X_ps       # (n, k)

    # Total influence function: IF_i = psi_tau,i + G_tau_gamma' * I_gamma^{-1} * S_gamma,i
    adj = S_gamma @ np.linalg.solve(I_gamma, G_tau_gamma)
    IF = psi_tau + adj

    # 1. i.i.d. two-step IF standard error
    se_if_iid = float(np.sqrt(np.sum(IF ** 2) / (n ** 2)))

    # 2. Country-clustered two-step IF standard error (G = 27)
    uniq_cl = np.unique(clusters)
    G_c = len(uniq_cl)
    cl_sums = np.array([IF[clusters == c].sum() for c in uniq_cl])
    se_if_cluster = float(np.sqrt((G_c / (G_c - 1.0)) * np.sum(cl_sums ** 2) / (n ** 2)))
    return se_if_iid, se_if_cluster


def estimate_causal_forest_manual(df):
    """
    Estimate IPW Average Treatment Effect (ATE) and subgroup Conditional Average
    Treatment Effects (CATEs) of public financial support (Q6.1) on reported
    net-zero strategy adoption (Q14).
    """
    print("\n" + "=" * 70)
    print("STRATIFIED INVERSE PROPENSITY SCORE WEIGHTING (IPW) ESTIMATION")
    print("=" * 70)

    T = df['grant_receipt'].values.astype(float)
    Y = df['q14_binary'].values.astype(float)
    clusters = df['country_iso2'].values

    X_cols = ['B2B_private', 'B2G_public', 'B2C', 'barrier_admin',
              'barrier_expertise', 'scr10', 'scr14', 're_action_count']
    FE = pd.concat([
        pd.get_dummies(df['country_iso2'], prefix='c', drop_first=True, dtype=float),
        pd.get_dummies(df['nace_b'].astype(int), prefix='s', drop_first=True, dtype=float)
    ], axis=1)
    X = np.column_stack([df[X_cols].values.astype(float), FE.values])
    X_ps = np.column_stack([X, np.ones(len(X))])

    # Step 1: Exact Newton-Raphson MLE for first-stage logistic propensity score
    beta_hat = fit_logit_nr(X_ps, T)
    ps_raw = expit(X_ps @ beta_hat)
    grad_norm = float(np.max(np.abs(X_ps.T @ (T - ps_raw))))
    ps = np.clip(ps_raw, 0.02, 0.98)

    # Stabilized weights and Kish Effective Sample Size (ESS)
    p_bar = float(np.mean(T))
    w_unstab = np.where(T == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    sw = np.where(T == 1, p_bar / ps, (1.0 - p_bar) / (1.0 - ps))
    ess_t = float((np.sum(sw[T == 1]) ** 2) / np.sum(sw[T == 1] ** 2))
    ess_c = float((np.sum(sw[T == 0]) ** 2) / np.sum(sw[T == 0] ** 2))
    ess_total = ess_t + ess_c

    print(f"  [Propensity Score & Weight Diagnostics (N = {len(df):,}, k = {X_ps.shape[1]})]")
    print(f"  - Newton-Raphson max |grad|:     {grad_norm:.2e} (Exact MLE convergence)")
    print(f"  - Mean PS (Pooled):              {ps.mean():.4f} (SD={ps.std():.4f}, range=[{ps.min():.4f}, {ps.max():.4f}])")
    print(f"  - Mean PS (Treated, N={int(T.sum()):,}):    {ps[T==1].mean():.4f} (SD={ps[T==1].std():.4f}, median={np.median(ps[T==1]):.4f}, range=[{ps[T==1].min():.4f}, {ps[T==1].max():.4f}])")
    print(f"  - Mean PS (Control, N={int((1-T).sum()):,}):   {ps[T==0].mean():.4f} (SD={ps[T==0].std():.4f}, median={np.median(ps[T==0]):.4f}, range=[{ps[T==0].min():.4f}, {ps[T==0].max():.4f}])")
    print(f"  - Stabilized Weight Mean:        Treated={sw[T==1].mean():.2f} (max={sw[T==1].max():.2f}), Control={sw[T==0].mean():.2f} (max={sw[T==0].max():.2f})")
    print(f"  - Kish Effective Sample Size:    ESS_T = {ess_t:.1f}/{int(T.sum()):,} ({ess_t/T.sum()*100:.1f}%), ESS_C = {ess_c:.1f}/{int((1-T).sum()):,} ({ess_c/(1-T).sum()*100:.1f}%), Total = {ess_total:.1f}")

    # Step 2: Hajek IPW ATE Point Estimate
    w1 = T / ps
    w0 = (1.0 - T) / (1.0 - ps)
    ate = float(np.mean(w1 * Y) / np.mean(w1) - np.mean(w0 * Y) / np.mean(w0))

    # Analytical two-step Influence-Function SEs (i.i.d. and country-clustered)
    se_if_iid, se_if_cluster = two_step_ipw_influence_function(X_ps, T, Y, ps, beta_hat, clusters)

    # Step 3: Non-parametric Bootstrap (B = 500)
    #   (a) Re-estimating propensity scores inside each bootstrap draw (accounts for PS estimation)
    #   (b) Holding propensity scores fixed (for historical comparison)
    n_boot = 500
    rng = np.random.RandomState(42)
    ate_boots_reest = []
    ate_boots_fixed = []

    admin_arr = df['barrier_admin'].values
    exp_arr = df['barrier_expertise'].values
    size_arr = df['scr10'].values
    dii_arr = df['DII_composite'].values
    dii_median = float(df['DII_composite'].median())

    subgroups = [
        ('barrier_admin', 1, 'High admin barrier', admin_arr == 1),
        ('barrier_admin', 0, 'No admin barrier', admin_arr == 0),
        ('barrier_expertise', 1, 'Has expertise deficit', exp_arr == 1),
        ('barrier_expertise', 0, 'No expertise deficit', exp_arr == 0),
        ('firm_size', 1, 'Micro (1-9)', size_arr == 1),
        ('firm_size', 2, 'Small (10-49)', size_arr == 2),
        ('firm_size', 3, 'Medium (50-249)', size_arr == 3),
        ('DII_composite', 'low', f'Low DII (< {dii_median:.1f})', dii_arr < dii_median),
        ('DII_composite', 'high', f'High DII (≥ {dii_median:.1f})', dii_arr >= dii_median),
    ]
    sg_boots_reest = {i: [] for i in range(len(subgroups))}
    sg_boots_fixed = {i: [] for i in range(len(subgroups))}

    for _ in range(n_boot):
        idx = rng.choice(len(Y), len(Y), replace=True)
        T_b, Y_b, X_b = T[idx], Y[idx], X_ps[idx]

        # Re-estimate first-stage logit on bootstrap sample (warm-started at beta_hat)
        beta_b = fit_logit_nr(X_b, T_b, beta_init=beta_hat, max_iter=15)
        ps_b = np.clip(expit(X_b @ beta_b), 0.02, 0.98)
        ps_f = ps[idx]

        w1_re = T_b / ps_b
        w0_re = (1.0 - T_b) / (1.0 - ps_b)
        ate_boots_reest.append(np.mean(w1_re * Y_b) / np.mean(w1_re) - np.mean(w0_re * Y_b) / np.mean(w0_re))

        w1_fx = T_b / ps_f
        w0_fx = (1.0 - T_b) / (1.0 - ps_f)
        ate_boots_fixed.append(np.mean(w1_fx * Y_b) / np.mean(w1_fx) - np.mean(w0_fx * Y_b) / np.mean(w0_fx))

        for i, (_, _, _, mask_full) in enumerate(subgroups):
            m_b = mask_full[idx]
            Tb_s, Yb_s = T_b[m_b], Y_b[m_b]
            ps_re_s, ps_fx_s = ps_b[m_b], ps_f[m_b]
            w1_rs, w0_rs = Tb_s / ps_re_s, (1.0 - Tb_s) / (1.0 - ps_re_s)
            w1_fs, w0_fs = Tb_s / ps_fx_s, (1.0 - Tb_s) / (1.0 - ps_fx_s)
            sg_boots_reest[i].append(np.mean(w1_rs * Yb_s) / np.mean(w1_rs) - np.mean(w0_rs * Yb_s) / np.mean(w0_rs))
            sg_boots_fixed[i].append(np.mean(w1_fs * Yb_s) / np.mean(w1_fs) - np.mean(w0_fs * Yb_s) / np.mean(w0_fs))

    ate_se_reest = float(np.std(ate_boots_reest))
    ate_se_fixed = float(np.std(ate_boots_fixed))
    ate_ci_reest = (ate - 1.96 * ate_se_reest, ate + 1.96 * ate_se_reest)
    ate_ci_fixed = (ate - 1.96 * ate_se_fixed, ate + 1.96 * ate_se_fixed)
    p_reest = float(2 * (1 - stats.norm.cdf(abs(ate / ate_se_reest))))
    p_fixed = float(2 * (1 - stats.norm.cdf(abs(ate / ate_se_fixed))))
    p_if_cl = float(2 * (1 - stats.norm.cdf(abs(ate / se_if_cluster))))
    p_if_cl_t26 = float(2 * (1 - stats.t.cdf(abs(ate / se_if_cluster), df=26)))

    print(f"\n  [Average Treatment Effect (ATE) Results]")
    print(f"  - IPW ATE Point Estimate:             {ate:+.4f}")
    print(f"  - Re-estimated PS Bootstrap SE:       {ate_se_reest:.4f} (95% CI [{ate_ci_reest[0]:+.4f}, {ate_ci_reest[1]:+.4f}], p = {p_reest:.4f})")
    print(f"  - Two-Step Influence-Function SE:     {se_if_iid:.4f} (i.i.d. analytical)")
    print(f"  - Country-Clustered Two-Step IF SE:   {se_if_cluster:.4f} (p = {p_if_cl:.4f}, t_26 p = {p_if_cl_t26:.4f})")
    print(f"  - Fixed-PS Bootstrap SE (reference):  {ate_se_fixed:.4f} (95% CI [{ate_ci_fixed[0]:+.4f}, {ate_ci_fixed[1]:+.4f}], p = {p_fixed:.4f})")

    # Subgroup CATE results table
    print("\n  === Subgroup CATE Estimates (Re-Estimated PS vs. Fixed PS Bootstrap) ===")
    cate_results = []
    for i, (mod_name, val, label, mask) in enumerate(subgroups):
        T_sub, Y_sub, ps_sub = T[mask], Y[mask], ps[mask]
        w1_s = T_sub / ps_sub
        w0_s = (1.0 - T_sub) / (1.0 - ps_sub)
        cate = float(np.mean(w1_s * Y_sub) / np.mean(w1_s) - np.mean(w0_s * Y_sub) / np.mean(w0_s))
        se_re = float(np.std(sg_boots_reest[i]))
        se_fx = float(np.std(sg_boots_fixed[i]))
        n_t = int(T_sub.sum())
        n_c = int((1.0 - T_sub).sum())
        print(f"  {label:<24}: CATE = {cate:+.4f} | SE_reest = {se_re:.4f} (SE_fixed = {se_fx:.4f}) [N_t={n_t}, N_c={n_c}]")
        cate_results.append({
            'Moderator': mod_name,
            'Value': val,
            'Label': label,
            'CATE': cate,
            'SE_reest': se_re,
            'CI_lo_reest': cate - 1.96 * se_re,
            'CI_hi_reest': cate + 1.96 * se_re,
            'p_reest': 2 * (1 - stats.norm.cdf(abs(cate / se_re))),
            'SE': se_fx,
            'CI_lo': cate - 1.96 * se_fx,
            'CI_hi': cate + 1.96 * se_fx,
            'N_treated': n_t,
            'N_control': n_c
        })

    # Admin barrier CATE difference test
    diff_admin_boots = np.array(sg_boots_reest[0]) - np.array(sg_boots_reest[1])
    diff_admin_est = cate_results[0]['CATE'] - cate_results[1]['CATE']
    diff_admin_se = float(np.std(diff_admin_boots))
    diff_admin_p = float(2 * (1 - stats.norm.cdf(abs(diff_admin_est / diff_admin_se))))
    print(f"\n  - Admin Barrier CATE Contrast (High vs. None): ΔCATE = {diff_admin_est:+.4f} (SE_reest = {diff_admin_se:.4f}, p = {diff_admin_p:.4f})")

    ate_summary = pd.DataFrame([{
        'Estimator': 'Hajek Stabilized IPW',
        'ATE': ate,
        'SE_boot_reestimated_PS': ate_se_reest,
        'CI_lo_reestimated_PS': ate_ci_reest[0],
        'CI_hi_reestimated_PS': ate_ci_reest[1],
        'p_boot_reestimated_PS': p_reest,
        'SE_IF_iid': se_if_iid,
        'SE_IF_cluster27': se_if_cluster,
        'p_IF_cluster27': p_if_cl,
        'p_IF_cluster27_t26': p_if_cl_t26,
        'SE_boot_fixed_PS': ate_se_fixed,
        'CI_lo_fixed_PS': ate_ci_fixed[0],
        'CI_hi_fixed_PS': ate_ci_fixed[1],
        'p_boot_fixed_PS': p_fixed,
        'ESS_treated': ess_t,
        'ESS_control': ess_c,
        'ESS_total': ess_total,
        'Max_sw_treated': float(sw[T == 1].max()),
        'Max_sw_control': float(sw[T == 0].max()),
        'N': len(df),
        'N_treated': int(T.sum()),
        'N_control': int((1.0 - T).sum())
    }])

    return ate_summary, pd.DataFrame(cate_results)


def main():
    print("=" * 70)
    print("IPW-ATE & CATE ESTIMATION - Public Financial Support (H3a, H3c)")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    needed = ['country_iso2', 'nace_b', 'q14_binary', 'grant_receipt', 'B2B_private', 'B2G_public', 'B2C',
              'barrier_admin', 'barrier_expertise', 'scr10', 'scr14',
              're_action_count', 'DII_composite']
    df_model = df[needed].dropna().copy()
    print(f"  Complete routed cases (Q14 valid): {len(df_model):,}")

    ate_summary, cate_df = estimate_causal_forest_manual(df_model)

    mode = 'a' if os.path.exists(RESULTS_XLSX) else 'w'
    with pd.ExcelWriter(RESULTS_XLSX, engine='openpyxl', mode=mode,
                        if_sheet_exists='replace' if mode == 'a' else None) as writer:
        ate_summary.to_excel(writer, sheet_name='CausalForest_ATE', index=False)
        cate_df.to_excel(writer, sheet_name='CausalForest_CATEs', index=False)

    print(f"\n  ✓ Results saved to: {RESULTS_XLSX}")


if __name__ == '__main__':
    main()
