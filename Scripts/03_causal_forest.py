#!/usr/bin/env python3
"""
03_causal_forest.py
===================
Inverse Propensity Score Weighting (IPW) and Conditional Average Treatment Effect (CATE)
estimation for Hypotheses H3a and H3c (Public Financial Support & Capability Moderation).

Specification:
  - Sample: All routed EU-27 SMEs with valid Q14 (N = 11,521; N_T = 1,292, N_C = 10,229)
  - Propensity Score Model: Exact Newton-Raphson logistic MLE (||grad||_inf < 1e-11) on:
      7 baseline covariates (B2B_private, B2G_public, B2C, barrier_admin, barrier_expertise,
      scr10, 9-item re_action_count) + 9 categorical turnover bracket FEs (scr14_cat) +
      26 country FEs + 3 NACE macro-sector FEs + intercept (k = 46)
  - Inference:
      1. Non-parametric bootstrap (B = 500, seed = 42) re-estimating the first-stage logit PS
         inside every bootstrap replication
      2. Analytical two-step M-estimation / Influence-Function (IF) SEs (i.i.d. and clustered
         across G = 27 EU member states + finite-cluster t_26)

Outputs:
  - Data/Results_Consolidated.xlsx (sheets: CausalForest_ATE, CausalForest_CATEs, IPW_Diagnostics)
  - Data/Processed/IPW_Propensity_Scores.csv (exact firm-level NR propensity scores & weights for Figure_S6)
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
PS_CSV = os.path.join(DATA_DIR, "Processed", "IPW_Propensity_Scores.csv")
RESULTS_XLSX = os.path.join(DATA_DIR, "Results_Consolidated.xlsx")


def fit_logit_nr(X, y, beta_init=None, max_iter=50, tol=1e-13, ridge=0.0):
    """Exact Newton-Raphson (IRLS) solver for binary logistic regression (ridge=0 exact MLE)."""
    n, k = X.shape
    beta = np.zeros(k) if beta_init is None else beta_init.copy()
    for _ in range(max_iter):
        p = expit(X @ beta)
        grad = X.T @ (y - p)
        if np.max(np.abs(grad)) < tol:
            break
        w = np.clip(p * (1.0 - p), 1e-14, None)
        H = (X.T * w) @ X
        if ridge > 0.0:
            H = H + np.eye(k) * ridge
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.solve(H + np.eye(k) * 1e-10, grad)
        beta += step
        if np.max(np.abs(step)) < tol and np.max(np.abs(X.T @ (y - expit(X @ beta)))) < tol:
            break
    return beta


def two_step_ipw_influence_function(X_ps, T, Y, ps, beta_hat, clusters):
    """
    Analytical two-step Influence-Function (IF) standard error for the Hajek IPW ATE
    accounting for first-stage logit propensity score estimation (Lunceford & Davidian, 2004;
    Hirano, Imbens, & Ridder, 2003), both i.i.d. and country-clustered (G = 27).
    """
    n = len(Y)
    w1 = T / ps
    w0 = (1.0 - T) / (1.0 - ps)
    mu1 = np.sum(w1 * Y) / np.sum(w1)
    mu0 = np.sum(w0 * Y) / np.sum(w0)

    m1 = np.mean(w1)
    m0 = np.mean(w0)
    psi_tau = (w1 * (Y - mu1)) / m1 - (w0 * (Y - mu0)) / m0

    d_w1 = -(T * (Y - mu1) * (1.0 - ps) / ps) / m1
    d_w0 = -((1.0 - T) * (Y - mu0) * ps / (1.0 - ps)) / m0
    G_tau_gamma = (X_ps.T @ (d_w1 + d_w0)) / n

    p_raw = expit(X_ps @ beta_hat)
    W_logit = p_raw * (1.0 - p_raw)
    I_gamma = ((X_ps.T * W_logit) @ X_ps) / n
    S_gamma = (T - p_raw)[:, None] * X_ps

    adj = S_gamma @ np.linalg.solve(I_gamma, G_tau_gamma)
    IF = psi_tau + adj

    se_if_iid = float(np.sqrt(np.sum(IF ** 2) / (n ** 2)))

    uniq_cl = np.unique(clusters)
    G_c = len(uniq_cl)
    cl_sums = np.array([IF[clusters == c].sum() for c in uniq_cl])
    se_if_cluster = float(np.sqrt((G_c / (G_c - 1.0)) * np.sum(cl_sums ** 2) / (n ** 2)))
    return se_if_iid, se_if_cluster


def estimate_ipw_and_cates(df):
    print("\n" + "=" * 72)
    print("03_causal_forest.py: Exact NR Propensity Score, Hajek IPW & Subgroup CATEs")
    print("=" * 72)

    T = df['grant_receipt'].values.astype(float)
    Y = df['q14_binary'].values.astype(float)
    clusters = df['country_iso2'].values

    X_cols = [
        'B2B_private', 'B2G_public', 'B2C',
        'barrier_admin', 'barrier_expertise',
        'scr10', 're_action_count'
    ]
    FE = pd.concat([
        pd.get_dummies(df['scr14_cat'].astype(int), prefix='turn', drop_first=True, dtype=float),
        pd.get_dummies(df['country_iso2'], prefix='c', drop_first=True, dtype=float),
        pd.get_dummies(df['nace_b'].astype(int), prefix='s', drop_first=True, dtype=float)
    ], axis=1)
    X = np.column_stack([df[X_cols].values.astype(float), FE.values])
    X_ps = np.column_stack([X, np.ones(len(X))])

    beta_hat = fit_logit_nr(X_ps, T, tol=1e-13, ridge=0.0)
    eta_hat = X_ps @ beta_hat
    ps_raw = expit(eta_hat)
    grad_norm = float(np.max(np.abs(X_ps.T @ (T - ps_raw))))
    assert grad_norm < 1e-11, f"NR gradient norm {grad_norm:.2e} >= 1e-11!"

    ps = np.clip(ps_raw, 0.02, 0.98)

    p_bar = float(np.mean(T))
    sw = np.where(T == 1.0, p_bar / ps, (1.0 - p_bar) / (1.0 - ps))
    ess_t = float((np.sum(sw[T == 1]) ** 2) / np.sum(sw[T == 1] ** 2))
    ess_c = float((np.sum(sw[T == 0]) ** 2) / np.sum(sw[T == 0] ** 2))
    ess_total = ess_t + ess_c

    print(f"  - Sample: N = {len(df):,} (Treated = {int(T.sum()):,}, Control = {int((1-T).sum()):,}), k = {X_ps.shape[1]}")
    print(f"  - Exact NR ||grad||_inf = {grad_norm:.2e}")
    print(f"  - PS (raw):     Treated mean={ps_raw[T==1].mean():.4f} [{ps_raw[T==1].min():.4f}, {ps_raw[T==1].max():.4f}], "
          f"Control mean={ps_raw[T==0].mean():.4f} [{ps_raw[T==0].min():.4f}, {ps_raw[T==0].max():.4f}]")
    print(f"  - Kish ESS:     Treated = {ess_t:.1f}/{int(T.sum()):,} ({ess_t/T.sum()*100:.1f}%), "
          f"Control = {ess_c:.1f}/{int((1-T).sum()):,} ({ess_c/(1-T).sum()*100:.1f}%), Total = {ess_total:.1f}")

    # Save firm-level propensity scores for Figure_S6
    os.makedirs(os.path.dirname(PS_CSV), exist_ok=True)
    ps_out_df = pd.DataFrame({
        'uniqid': df['uniqid'].values,
        'country_iso2': clusters,
        'grant_receipt': T.astype(int),
        'q14_binary': Y.astype(int),
        'eta_hat': eta_hat,
        'ps_raw': ps_raw,
        'ps_clipped': ps,
        'stabilized_weight': sw
    })
    ps_out_df.to_csv(PS_CSV, index=False)

    # Hajek IPW ATE & analytical two-step Influence-Function SEs
    w1 = T / ps
    w0 = (1.0 - T) / (1.0 - ps)
    ate = float(np.mean(w1 * Y) / np.mean(w1) - np.mean(w0 * Y) / np.mean(w0))
    se_if_iid, se_if_cluster = two_step_ipw_influence_function(X_ps, T, Y, ps, beta_hat, clusters)

    # Non-parametric Bootstrap (B = 500, seed = 42) re-estimating first-stage PS
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
        ('DII_composite', 'high', f'High DII (>= {dii_median:.1f})', dii_arr >= dii_median),
    ]
    sg_boots_reest = {i: [] for i in range(len(subgroups))}
    sg_boots_fixed = {i: [] for i in range(len(subgroups))}

    for _ in range(n_boot):
        idx = rng.choice(len(Y), len(Y), replace=True)
        T_b, Y_b, X_b = T[idx], Y[idx], X_ps[idx]

        beta_b = fit_logit_nr(X_b, T_b, beta_init=beta_hat, max_iter=20, tol=1e-10, ridge=1e-10)
        ps_b = np.clip(expit(X_b @ beta_b), 0.02, 0.98)
        ps_f = ps[idx]

        w1_re, w0_re = T_b / ps_b, (1.0 - T_b) / (1.0 - ps_b)
        ate_boots_reest.append(np.mean(w1_re * Y_b) / np.mean(w1_re) - np.mean(w0_re * Y_b) / np.mean(w0_re))

        w1_fx, w0_fx = T_b / ps_f, (1.0 - T_b) / (1.0 - ps_f)
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
    p_reest = float(2.0 * (1.0 - stats.norm.cdf(abs(ate / ate_se_reest))))
    p_fixed = float(2.0 * (1.0 - stats.norm.cdf(abs(ate / ate_se_fixed))))
    p_if_cl = float(2.0 * (1.0 - stats.norm.cdf(abs(ate / se_if_cluster))))
    p_if_cl_t26 = float(2.0 * (1.0 - stats.t.cdf(abs(ate / se_if_cluster), df=26)))

    print(f"\n  [IPW ATE Results]")
    print(f"  - IPW ATE = {ate:+.4f} | SE_boot_reest = {ate_se_reest:.4f} (95% CI [{ate_ci_reest[0]:+.4f}, {ate_ci_reest[1]:+.4f}], p = {p_reest:.4f})")
    print(f"  - Analytical IF SE (i.i.d.) = {se_if_iid:.4f} | Clustered IF SE (G=27) = {se_if_cluster:.4f} (p = {p_if_cl:.4f}, t_26 p = {p_if_cl_t26:.4f})")

    cate_results = []
    for i, (mod_name, val, label, mask) in enumerate(subgroups):
        T_sub, Y_sub, ps_sub = T[mask], Y[mask], ps[mask]
        w1_s, w0_s = T_sub / ps_sub, (1.0 - T_sub) / (1.0 - ps_sub)
        cate = float(np.mean(w1_s * Y_sub) / np.mean(w1_s) - np.mean(w0_s * Y_sub) / np.mean(w0_s))
        se_re = float(np.std(sg_boots_reest[i]))
        se_fx = float(np.std(sg_boots_fixed[i]))
        n_t = int(T_sub.sum())
        n_c = int((1.0 - T_sub).sum())
        p_re = float(2.0 * (1.0 - stats.norm.cdf(abs(cate / se_re))))
        print(f"  - {label:<24}: CATE = {cate:+.4f} (SE_reest = {se_re:.4f}, p = {p_re:.4f}) [N_T={n_t}, N_C={n_c}]")
        cate_results.append({
            'Moderator': mod_name,
            'Value': val,
            'Label': label,
            'CATE': cate,
            'SE_reest': se_re,
            'CI_lo_reest': cate - 1.96 * se_re,
            'CI_hi_reest': cate + 1.96 * se_re,
            'p_reest': p_re,
            'SE': se_fx,
            'CI_lo': cate - 1.96 * se_fx,
            'CI_hi': cate + 1.96 * se_fx,
            'N_treated': n_t,
            'N_control': n_c
        })

    diff_admin_boots = np.array(sg_boots_reest[0]) - np.array(sg_boots_reest[1])
    diff_admin_est = float(cate_results[0]['CATE'] - cate_results[1]['CATE'])
    diff_admin_se = float(np.std(diff_admin_boots))
    diff_admin_p = float(2.0 * (1.0 - stats.norm.cdf(abs(diff_admin_est / diff_admin_se))))

    diff_exp_boots = np.array(sg_boots_reest[2]) - np.array(sg_boots_reest[3])
    diff_exp_est = float(cate_results[2]['CATE'] - cate_results[3]['CATE'])
    diff_exp_se = float(np.std(diff_exp_boots))
    diff_exp_p = float(2.0 * (1.0 - stats.norm.cdf(abs(diff_exp_est / diff_exp_se))))

    print(f"  - Delta CATE (Admin High vs None):    {diff_admin_est:+.4f} (SE = {diff_admin_se:.4f}, p = {diff_admin_p:.4f})")
    print(f"  - Delta CATE (Expertise Def vs None): {diff_exp_est:+.4f} (SE = {diff_exp_se:.4f}, p = {diff_exp_p:.4f})")

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
        'Delta_CATE_Admin': diff_admin_est,
        'SE_Delta_CATE_Admin': diff_admin_se,
        'p_Delta_CATE_Admin': diff_admin_p,
        'Delta_CATE_Expertise': diff_exp_est,
        'SE_Delta_CATE_Expertise': diff_exp_se,
        'p_Delta_CATE_Expertise': diff_exp_p,
        'ESS_treated': ess_t,
        'ESS_control': ess_c,
        'ESS_total': ess_total,
        'Max_sw_treated': float(sw[T == 1].max()),
        'Max_sw_control': float(sw[T == 0].max()),
        'N': len(df),
        'N_treated': int(T.sum()),
        'N_control': int((1.0 - T).sum()),
        'k_ps': int(X_ps.shape[1]),
        'nr_grad_inf_norm': grad_norm
    }])

    diag_rows = []
    for grp_lbl, arr in [('Pooled_PS_raw', ps_raw),
                         ('Treated_PS_raw', ps_raw[T == 1]),
                         ('Control_PS_raw', ps_raw[T == 0]),
                         ('Treated_PS_clipped', ps[T == 1]),
                         ('Control_PS_clipped', ps[T == 0]),
                         ('Treated_Stabilized_Weight', sw[T == 1]),
                         ('Control_Stabilized_Weight', sw[T == 0])]:
        qs = np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        diag_rows.append({
            'Series': grp_lbl,
            'N': int(len(arr)),
            'Mean': float(np.mean(arr)),
            'SD': float(np.std(arr)),
            'Min': float(np.min(arr)),
            'P01': float(qs[0]), 'P05': float(qs[1]), 'P10': float(qs[2]),
            'P25': float(qs[3]), 'Median': float(qs[4]), 'P75': float(qs[5]),
            'P90': float(qs[6]), 'P95': float(qs[7]), 'P99': float(qs[8]),
            'Max': float(np.max(arr)),
            'Kish_ESS': float((np.sum(arr) ** 2) / np.sum(arr ** 2)) if 'Weight' in grp_lbl else np.nan,
            'NR_grad_inf_norm': grad_norm,
            'k_params': int(X_ps.shape[1])
        })
    ipw_diag_df = pd.DataFrame(diag_rows)

    return ate_summary, pd.DataFrame(cate_results), ipw_diag_df


def main():
    df = pd.read_csv(DATA_FILE)
    needed = [
        'uniqid', 'country_iso2', 'nace_b', 'scr14_cat',
        'q14_binary', 'grant_receipt',
        'B2B_private', 'B2G_public', 'B2C',
        'barrier_admin', 'barrier_expertise',
        'scr10', 're_action_count', 'DII_composite'
    ]
    df_model = df[needed].dropna().copy()
    ate_summary, cate_df, ipw_diag_df = estimate_ipw_and_cates(df_model)

    mode = 'a' if os.path.exists(RESULTS_XLSX) else 'w'
    with pd.ExcelWriter(RESULTS_XLSX, engine='openpyxl', mode=mode,
                        if_sheet_exists='replace' if mode == 'a' else None) as writer:
        ate_summary.to_excel(writer, sheet_name='CausalForest_ATE', index=False)
        cate_df.to_excel(writer, sheet_name='CausalForest_CATEs', index=False)
        ipw_diag_df.to_excel(writer, sheet_name='IPW_Diagnostics', index=False)

    print(f"\n  ✓ Saved IPW/CATE sheets to {RESULTS_XLSX} and firm-level PS to {PS_CSV}")


if __name__ == '__main__':
    main()
