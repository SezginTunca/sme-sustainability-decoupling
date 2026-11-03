#!/usr/bin/env python3
"""
02_bivariate_probit.py
======================
Joint Full-Information Maximum Likelihood (FIML) Bivariate Probit with:
  - Categorical turnover bracket fixed effects (scr14_cat: 9 indicators covering 1..9 + DK=10)
  - 26 EU country fixed effects and 3 NACE macro-sector fixed effects
  - Strict 9-item substantive resource-efficiency action count (re_action_count)
  - Country-clustered sandwich covariance across G = 27 EU member states + finite-cluster t_26
  - Exact analytical score vector and Plackett Gauss-Legendre quadrature (24- vs. 48-node verified)
  - Sample-average discrete marginal effects (AMEs) via joint delta method
  - Joint cluster-robust Wald chi-squared tests on Turnover, Country, and Sector fixed effects
  - Primary Holm-Bonferroni adjustment across m = 6 focal hypothesized predictors (plus m = 7 stored)

Outputs:
  - Data/Processed/BiProbit_Joint_FE.xlsx
  - Data/Processed/BiProbit_Joint_FE.csv
  - Data/Results_Consolidated.xlsx (sheets: BiProbit_Coefficients, BiProbit_CrossEq_Tests,
    BiProbit_AME, BiProbit_Diagnostics, plus legacy sheet aliases BiProbit_Joint_FE & Cross_Eq_Tests)
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Data")
if not os.path.exists(DATA_DIR) and os.path.exists(os.path.join(os.path.dirname(ROOT), "Data")):
    DATA_DIR = os.path.join(os.path.dirname(ROOT), "Data")
DATA = os.path.join(DATA_DIR, "Processed", "SME_Analytical_Final.csv")
OUT_CSV = os.path.join(DATA_DIR, "Processed", "BiProbit_Joint_FE.csv")
OUT_XLSX = os.path.join(DATA_DIR, "Processed", "BiProbit_Joint_FE.xlsx")
RESULTS_XLSX = os.path.join(DATA_DIR, "Results_Consolidated.xlsx")

IV = [
    'B2B_private', 'B2G_public', 'B2C',
    'barrier_admin', 'barrier_expertise',
    'grant_receipt', 'advice_public'
]
FOCAL_6 = [
    'B2C', 'B2G_public', 'B2B_private',
    'barrier_admin', 'barrier_expertise', 'grant_receipt'
]
CTRL = ['scr10', 're_action_count']
GL_X, GL_W = np.polynomial.legendre.leggauss(24)


def bvn_cdf(h, k, rho, gl_x=GL_X, gl_w=GL_W):
    """Phi_2(h, k; rho) via Plackett identity with Gauss-Legendre quadrature."""
    r = 0.5 * rho[:, None] * (gl_x[None, :] + 1.0)
    w = 0.5 * rho[:, None] * gl_w[None, :]
    hh, kk = h[:, None], k[:, None]
    one = np.clip(1.0 - r ** 2, 1e-14, None)
    dens = np.exp(-(hh ** 2 - 2.0 * r * hh * kk + kk ** 2) / (2.0 * one)) / (2.0 * np.pi * np.sqrt(one))
    return stats.norm.cdf(h) * stats.norm.cdf(k) + (w * dens).sum(axis=1)


def obs_ll(theta, X, y1, y2, weights=None):
    k = X.shape[1]
    b1, b2, rho = theta[:k], theta[k:2 * k], np.tanh(theta[2 * k])
    q1, q2 = 2.0 * y1 - 1.0, 2.0 * y2 - 1.0
    p = bvn_cdf(q1 * (X @ b1), q2 * (X @ b2), q1 * q2 * rho * np.ones(len(y1)))
    ll_i = np.log(np.clip(p, 1e-300, None))
    return ll_i if weights is None else weights * ll_i


def scores(theta, X, y1, y2, weights=None):
    """Exact analytical score matrix d(w_i * log Phi_2,i)/d(theta) of shape (N, 2k+1)."""
    k = X.shape[1]
    b1, b2, z_rho = theta[:k], theta[k:2 * k], theta[2 * k]
    rho = np.tanh(z_rho)
    q1, q2 = 2.0 * y1 - 1.0, 2.0 * y2 - 1.0
    h = q1 * (X @ b1)
    m = q2 * (X @ b2)
    r = q1 * q2 * rho
    s_1mr2 = np.sqrt(np.clip(1.0 - r ** 2, 1e-12, None))
    p = np.clip(bvn_cdf(h, m, r), 1e-300, None)
    dp_dh = stats.norm.pdf(h) * stats.norm.cdf((m - r * h) / s_1mr2)
    dp_dm = stats.norm.pdf(m) * stats.norm.cdf((h - r * m) / s_1mr2)
    dp_dr = np.exp(-(h ** 2 - 2.0 * r * h * m + m ** 2) / (2.0 * np.clip(1.0 - r ** 2, 1e-12, None))) / (2.0 * np.pi * s_1mr2)
    g1 = ((q1 * dp_dh / p)[:, None]) * X
    g2 = ((q2 * dp_dm / p)[:, None]) * X
    g_z = (q1 * q2 * (1.0 - rho ** 2) * dp_dr / p)[:, None]
    S = np.hstack([g1, g2, g_z])
    return S if weights is None else S * weights[:, None]


def holm_adjust(p_vals):
    """Step-down Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p_vals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    run = 0.0
    for r, idx in enumerate(order):
        run = max(run, (m - r) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def ames(theta, X, names, vars_):
    """Discrete counterfactual AMEs for binary variables; marginal derivative for continuous."""
    k = X.shape[1]
    b1, b2 = theta[:k], theta[k:2 * k]
    out = []
    for v in vars_:
        j = names.index(v)
        col_vals = np.unique(X[:, j])
        if set(col_vals).issubset({0.0, 1.0}):
            X1, X0 = X.copy(), X.copy()
            X1[:, j] = 1.0
            X0[:, j] = 0.0
            out.append([(stats.norm.cdf(X1 @ b) - stats.norm.cdf(X0 @ b)).mean() for b in (b1, b2)])
        else:
            out.append([(stats.norm.pdf(X @ b) * b[j]).mean() for b in (b1, b2)])
    return np.array(out)


def build_design_matrix(df, y1_col='q9_binary', y2_col='q14_binary',
                        iv_cols=IV, ctrl_cols=CTRL,
                        turnover_mode='cat', sector_col='nace_b',
                        extra_cols=None, merge_top_turnover=False, keep_cols=None):
    """
    Construct clean design matrix X for Joint FIML Bivariate Probit.
    turnover_mode:
      - 'cat': categorical turnover bracket FEs (scr14_cat 1..10, drop_first=True)
      - 'clean': complete-case ordinal scr14_clean (1..9)
      - 'imp_dk': ordinal scr14_imp + scr14_dk indicator
    """
    extra_cols = extra_cols or []
    keep_cols = keep_cols or []
    req = list(dict.fromkeys(list(iv_cols) + list(ctrl_cols) + list(extra_cols) + list(keep_cols) + [y1_col, y2_col, 'country_iso2', sector_col]))
    if turnover_mode == 'cat':
        req.append('scr14_cat')
    elif turnover_mode == 'clean':
        req.append('scr14_clean')
    elif turnover_mode == 'imp_dk':
        req.extend(['scr14_imp', 'scr14_dk'])
    elif turnover_mode == 'none':
        pass
    else:
        raise ValueError(f"Unknown turnover_mode: {turnover_mode}")

    d = df[req].dropna().copy()
    parts = [d[list(iv_cols) + list(ctrl_cols) + list(extra_cols)].astype(float)]

    if turnover_mode == 'cat':
        t_cat = d['scr14_cat'].astype(int).copy()
        if merge_top_turnover:
            # Merge bracket 9 (>50M) into bracket 8 (>10M) if top cell causes near-separation in tiny subsample
            t_cat = t_cat.replace({9: 8})
        turn_fe = pd.get_dummies(t_cat, prefix='turn', drop_first=True, dtype=float)
        parts.append(turn_fe)
    elif turnover_mode == 'clean':
        parts.append(d[['scr14_clean']].astype(float))
    elif turnover_mode == 'imp_dk':
        parts.append(d[['scr14_imp', 'scr14_dk']].astype(float))

    c_fe = pd.get_dummies(d['country_iso2'], prefix='c', drop_first=True, dtype=float)
    s_fe = pd.get_dummies(d[sector_col].astype(int), prefix='s', drop_first=True, dtype=float)
    parts.extend([c_fe, s_fe])

    Xdf = pd.concat(parts, axis=1)
    # Drop any strictly zero-variance column in tiny subsamples (if any)
    zero_var = [c for c in Xdf.columns if Xdf[c].nunique() <= 1]
    if zero_var:
        Xdf = Xdf.drop(columns=zero_var)
    Xdf.insert(0, 'const', 1.0)
    return d, Xdf


def fit_biprobit_fiml(d, Xdf, y1_col='q9_binary', y2_col='q14_binary',
                      cluster_col='country_iso2', weight_col=None,
                      compute_ames_for=None, focal_6=FOCAL_6):
    """
    Fit Joint FIML Bivariate Probit (or Binder survey-weighted joint PML when weight_col is given)
    with analytical gradients, BFGS + L-BFGS-B polishing to ||grad||_inf < 1e-7, PD Hessian check,
    and country-clustered sandwich covariance.
    """
    names = list(Xdf.columns)
    X = Xdf.values.astype(float)
    y1 = d[y1_col].values.astype(float)
    y2 = d[y2_col].values.astype(float)
    N, k = X.shape

    w = None
    if weight_col is not None:
        w_raw = d[weight_col].values.astype(float)
        w = w_raw * (N / np.sum(w_raw))

    # Warm-start from separate probits
    s1 = sm.Probit(y1, X).fit(disp=0, maxiter=200)
    s2 = sm.Probit(y2, X).fit(disp=0, maxiter=200)
    th0 = np.concatenate([s1.params, s2.params, [0.15]])

    f = lambda t: -obs_ll(t, X, y1, y2, weights=w).sum()
    gr = lambda t: -scores(t, X, y1, y2, weights=w).sum(axis=0)

    res = minimize(f, th0, jac=gr, method='BFGS', options={'maxiter': 500, 'gtol': 1e-8})
    th = res.x
    grad_max = float(np.abs(gr(th)).max())
    if grad_max >= 1e-7:
        res2 = minimize(f, th, jac=gr, method='L-BFGS-B',
                        options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-10})
        if np.abs(gr(res2.x)).max() < grad_max:
            res = res2
            th = res2.x
            grad_max = float(np.abs(gr(th)).max())
    if grad_max >= 1e-7:
        res3 = minimize(f, th, jac=gr, method='BFGS', options={'maxiter': 500, 'gtol': 1e-10})
        if np.abs(gr(res3.x)).max() < grad_max:
            res = res3
            th = res3.x
            grad_max = float(np.abs(gr(th)).max())

    def compute_hess(theta_vec):
        n_par_ = len(theta_vec)
        H_ = np.empty((n_par_, n_par_))
        eps_h_ = 1e-5
        for j_ in range(n_par_):
            e_ = np.zeros(n_par_)
            e_[j_] = eps_h_
            s_plus_ = scores(theta_vec + e_, X, y1, y2, weights=w).sum(axis=0)
            s_minus_ = scores(theta_vec - e_, X, y1, y2, weights=w).sum(axis=0)
            H_[:, j_] = -(s_plus_ - s_minus_) / (2.0 * eps_h_)
        return 0.5 * (H_ + H_.T)

    # Exact Newton-Raphson polishing to machine precision
    for _ in range(4):
        if grad_max < 1e-9:
            break
        H_pol = compute_hess(th)
        try:
            step = np.linalg.solve(H_pol, gr(th))
        except np.linalg.LinAlgError:
            break
        for alpha in (1.0, 0.5, 0.25, 0.1):
            th_cand = th - alpha * step
            g_cand = float(np.abs(gr(th_cand)).max())
            if g_cand < grad_max:
                th = th_cand
                grad_max = g_cand
                break
        else:
            break

    # 24-pt vs 48-pt Gauss-Legendre quadrature accuracy check
    gl48_x, gl48_w = np.polynomial.legendre.leggauss(48)
    b1_hat, b2_hat, rho_hat = th[:k], th[k:2 * k], float(np.tanh(th[2 * k]))
    q1_v, q2_v = 2.0 * y1 - 1.0, 2.0 * y2 - 1.0
    h_v, k_v = q1_v * (X @ b1_hat), q2_v * (X @ b2_hat)
    r_v = q1_v * q2_v * rho_hat * np.ones(N)
    p24 = bvn_cdf(h_v, k_v, r_v)
    p48 = bvn_cdf(h_v, k_v, r_v, gl_x=gl48_x, gl_w=gl48_w)
    quad_diff_max = float(np.max(np.abs(np.log(np.clip(p24, 1e-300, None)) - np.log(np.clip(p48, 1e-300, None)))))

    n_par = len(th)
    H = compute_hess(th)
    ev = np.linalg.eigvalsh(H)
    Hi = np.linalg.inv(H)

    S = scores(th, X, y1, y2, weights=w)
    cl = d[cluster_col].values
    uniq_cl = np.unique(cl)
    G_c = len(uniq_cl)
    G_mat = np.vstack([S[cl == c].sum(axis=0) for c in uniq_cl])
    dof_scale = (G_c / (G_c - 1.0)) * ((N - 1.0) / (N - n_par))
    meat = G_mat.T @ G_mat * dof_scale
    V = Hi @ meat @ Hi
    V_rob = Hi @ (S.T @ S) @ Hi
    se = np.sqrt(np.diag(V))
    df_c = G_c - 1

    rho = float(np.tanh(th[-1]))
    se_rho = float((1.0 - rho ** 2) * np.sqrt(V[-1, -1]))

    # Joint Wald tests for fixed-effect blocks
    # Note: With G=27 country clusters, rank(V_cluster) <= G-1 = 26 (Cameron & Miller, 2015),
    # so Turnover_Bracket_FE (df=9) and Sector_FE (df=3) use cluster-robust V, whereas
    # Country_FE (df=26 per eq, df=52 joint) uses full-rank heteroskedasticity-robust V_rob.
    fe_wald_rows = []
    for block_name, prefix in [('Turnover_Bracket_FE', 'turn_'),
                               ('Country_FE', 'c_'),
                               ('Sector_FE', 's_')]:
        idxs = [i for i, nm in enumerate(names) if nm.startswith(prefix)]
        if not idxs:
            continue
        df_b = len(idxs)
        use_cluster_1eq = (df_b < df_c)
        V_1eq = V if use_cluster_1eq else V_rob
        cov_lbl_1eq = f'Cluster_{G_c}' if use_cluster_1eq else 'Huber_White_Robust'
        for eq_lbl, off in [('Y1_Q9', 0), ('Y2_Q14', k)]:
            q_idx = [i + off for i in idxs]
            b_sub = th[q_idx]
            V_sub = V_1eq[np.ix_(q_idx, q_idx)]
            chi2_val = float(b_sub @ np.linalg.solve(V_sub, b_sub))
            p_chi2 = float(1.0 - stats.chi2.cdf(chi2_val, df=df_b))
            fe_wald_rows.append({
                'FE_Block': block_name,
                'Equation': eq_lbl,
                'Cov_Type': cov_lbl_1eq,
                'df': df_b,
                'Wald_chi2': chi2_val,
                'p_chi2': p_chi2
            })
        use_cluster_2eq = (2 * df_b < df_c)
        V_2eq = V if use_cluster_2eq else V_rob
        cov_lbl_2eq = f'Cluster_{G_c}' if use_cluster_2eq else 'Huber_White_Robust'
        q_both = idxs + [i + k for i in idxs]
        b_both = th[q_both]
        V_both = V_2eq[np.ix_(q_both, q_both)]
        chi2_both = float(b_both @ np.linalg.solve(V_both, b_both))
        fe_wald_rows.append({
            'FE_Block': block_name,
            'Equation': 'Joint_Y1_Y2',
            'Cov_Type': cov_lbl_2eq,
            'df': 2 * df_b,
            'Wald_chi2': chi2_both,
            'p_chi2': float(1.0 - stats.chi2.cdf(chi2_both, df=2 * df_b))
        })
    fe_wald_df = pd.DataFrame(fe_wald_rows)

    # Cross-equation Wald tests on compute_ames_for (defaults to IVs present in names)
    eval_vars = [v for v in (compute_ames_for or IV) if v in names]
    tests = []
    for v in eval_vars:
        j = names.index(v)
        d_b = float(th[j] - th[k + j])
        var_d = float(V[j, j] + V[k + j, k + j] - 2.0 * V[j, k + j])
        se_d = float(np.sqrt(var_d))
        z_d = d_b / se_d
        tests.append(dict(
            var=v, test='beta_diff', diff=d_b, se=se_d, z=z_d,
            p=float(2.0 * (1.0 - stats.norm.cdf(abs(z_d)))),
            p_t26=float(2.0 * (1.0 - stats.t.cdf(abs(z_d), df=df_c)))
        ))

    # Sample-average discrete marginal effects (AMEs) + joint delta method
    a0 = ames(th, X, names, eval_vars)
    J = np.zeros((len(eval_vars) * 2, n_par))
    eps_a = 1e-5
    for j in range(2 * k):
        e = np.zeros(n_par)
        e[j] = eps_a
        diff_a = (ames(th + e, X, names, eval_vars) - ames(th - e, X, names, eval_vars)) / (2.0 * eps_a)
        J[:, j] = diff_a.reshape(-1)
    Vam = J @ V @ J.T
    am_rows = []
    for i, v in enumerate(eval_vars):
        i1, i2 = 2 * i, 2 * i + 1
        dd = float(a0[i, 0] - a0[i, 1])
        vd = float(Vam[i1, i1] + Vam[i2, i2] - 2.0 * Vam[i1, i2])
        se_dd = float(np.sqrt(vd))
        z_dd = dd / se_dd
        am_rows.append(dict(
            var=v,
            AME_Q9=float(a0[i, 0]), se_Q9=float(np.sqrt(Vam[i1, i1])),
            p_AME_Q9=float(2.0 * (1.0 - stats.norm.cdf(abs(a0[i, 0] / np.sqrt(Vam[i1, i1]))))),
            AME_Q14=float(a0[i, 1]), se_Q14=float(np.sqrt(Vam[i2, i2])),
            p_AME_Q14=float(2.0 * (1.0 - stats.norm.cdf(abs(a0[i, 1] / np.sqrt(Vam[i2, i2]))))),
            AME_diff=dd, se_AME_diff=se_dd, z_AME_diff=z_dd,
            p_AME_diff=float(2.0 * (1.0 - stats.norm.cdf(abs(z_dd)))),
            p_AME_diff_t26=float(2.0 * (1.0 - stats.t.cdf(abs(z_dd), df=df_c)))
        ))
        tests[i]['p_AME_diff'] = am_rows[-1]['p_AME_diff']
        tests[i]['p_AME_diff_t26'] = am_rows[-1]['p_AME_diff_t26']

    tdf = pd.DataFrame(tests)
    if len(tdf) > 0:
        # Primary Holm-Bonferroni over m = 6 focal hypothesized predictors
        focal_mask = tdf['var'].isin(focal_6)
        tdf['p_holm'] = np.nan
        tdf['p_t26_holm'] = np.nan
        if focal_mask.sum() > 0:
            tdf.loc[focal_mask, 'p_holm'] = holm_adjust(tdf.loc[focal_mask, 'p'].values)
            tdf.loc[focal_mask, 'p_t26_holm'] = holm_adjust(tdf.loc[focal_mask, 'p_t26'].values)
        tdf['p_holm_m6'] = tdf['p_holm']
        tdf['p_t26_holm_m6'] = tdf['p_t26_holm']
        # Also store m = 7 Holm adjustment (including advice_public) for full transparency
        tdf['p_holm_m7'] = holm_adjust(tdf['p'].values)
        tdf['p_t26_holm_m7'] = holm_adjust(tdf['p_t26'].values)

    am_df = pd.DataFrame(am_rows)
    diag = dict(
        N=N, k_per_eq=k, total_params=n_par, clusters=G_c, df_cluster=df_c,
        rho=rho, se_rho=se_rho, z_rho=rho / se_rho,
        p_rho=float(2.0 * (1.0 - stats.norm.cdf(abs(rho / se_rho)))),
        ll=float(-res.fun), converged=bool(res.success or grad_max < 1e-7),
        grad_inf_norm=grad_max,
        hess_min_eig=float(ev.min()), hess_max_eig=float(ev.max()),
        hess_cond_num=float(ev.max() / ev.min()) if ev.min() > 0 else np.nan,
        quad_diff_24_vs_48=quad_diff_max
    )
    # Add FE Wald test summary columns to diag for convenient lookup
    for _, r_fe in fe_wald_df.iterrows():
        key_base = f"{r_fe['FE_Block']}_{r_fe['Equation']}"
        diag[f"{key_base}_chi2"] = float(r_fe['Wald_chi2'])
        diag[f"{key_base}_df"] = int(r_fe['df'])
        diag[f"{key_base}_p"] = float(r_fe['p_chi2'])

    return {
        'theta': th, 'V': V, 'V_rob': V_rob, 'se': se, 'names': names, 'k': k,
        'df_c': df_c, 'tdf': tdf, 'am_df': am_df, 'fe_wald_df': fe_wald_df, 'diag': diag
    }


def main():
    df = pd.read_csv(DATA)
    d, Xdf = build_design_matrix(df, y1_col='q9_binary', y2_col='q14_binary',
                                 iv_cols=IV, ctrl_cols=CTRL, turnover_mode='cat', sector_col='nace_b')
    print("=" * 76)
    print(f"02_bivariate_probit.py: Joint FIML Bivariate Probit (N = {len(d):,}, k = {Xdf.shape[1]}/eq)")
    print("=" * 76)

    fit = fit_biprobit_fiml(d, Xdf, y1_col='q9_binary', y2_col='q14_binary')
    th, se, V_rob, names, k, df_c = fit['theta'], fit['se'], fit['V_rob'], fit['names'], fit['k'], fit['df_c']
    diag = fit['diag']

    assert diag['grad_inf_norm'] < 1e-7, f"Gradient norm {diag['grad_inf_norm']:.2e} >= 1e-7!"
    assert diag['hess_min_eig'] > 0.0, f"Hessian not PD: min eig = {diag['hess_min_eig']}!"

    print(f"  Converged: {diag['converged']} | LL = {diag['ll']:.2f} | ||grad||_inf = {diag['grad_inf_norm']:.2e}")
    print(f"  Hessian eigenvalues: min = {diag['hess_min_eig']:.4f}, max = {diag['hess_max_eig']:.1f} (PD)")
    print(f"  Quadrature 24 vs 48 max |Delta log L_i| = {diag['quad_diff_24_vs_48']:.2e}")
    print(f"  Disturbance correlation rho = {diag['rho']:.4f} (cluster SE = {diag['se_rho']:.4f}, z = {diag['z_rho']:.2f})")

    # Build coefficient table (focal IVs + controls + turnover bracket FEs)
    turn_cols = [c for c in names if c.startswith('turn_')]
    report_vars = IV + CTRL + turn_cols
    rows = []
    for eq, off in [('Y1_Q9_green_offerings', 0), ('Y2_Q14_netzero_strategy', k)]:
        for v in report_vars:
            j = names.index(v) + off
            t_stat = float(th[j] / se[j])
            rows.append(dict(
                eq=eq, var=v, beta=float(th[j]), se_cluster=float(se[j]),
                se_robust=float(np.sqrt(V_rob[j, j])),
                z_cluster=t_stat,
                p_cluster=float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat)))),
                p_t26=float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_c)))
            ))
    coef = pd.DataFrame(rows)
    tdf = fit['tdf']
    am = fit['am_df']
    fe_wald_df = fit['fe_wald_df']
    diag_df = pd.DataFrame([diag])

    print("\n[Focal & Control Coefficients]")
    print(coef[coef['var'].isin(IV + CTRL)].round(4).to_string(index=False))
    print("\n[Cross-Equation Wald Tests (m=6 Focal Holm Primary; m=7 Also Stored)]")
    print(tdf.round(4).to_string(index=False))
    print("\n[Joint Discrete Average Marginal Effects (AMEs)]")
    print(am.round(4).to_string(index=False))
    print("\n[Joint Cluster-Robust Wald Chi-Squared Tests for Fixed-Effect Blocks]")
    print(fe_wald_df.round(4).to_string(index=False))

    coef.to_csv(OUT_CSV, index=False)
    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as w:
        coef.to_excel(w, sheet_name='coef', index=False)
        tdf.to_excel(w, sheet_name='cross_eq_tests', index=False)
        am.to_excel(w, sheet_name='AME', index=False)
        diag_df.to_excel(w, sheet_name='diagnostics', index=False)
        fe_wald_df.to_excel(w, sheet_name='fe_wald_tests', index=False)

    # Also write directly into Data/Results_Consolidated.xlsx
    mode = 'a' if os.path.exists(RESULTS_XLSX) else 'w'
    with pd.ExcelWriter(RESULTS_XLSX, engine='openpyxl', mode=mode,
                        if_sheet_exists='replace' if mode == 'a' else None) as w:
        coef.to_excel(w, sheet_name='BiProbit_Coefficients', index=False)
        coef.to_excel(w, sheet_name='BiProbit_Joint_FE', index=False)
        tdf.to_excel(w, sheet_name='BiProbit_CrossEq_Tests', index=False)
        tdf.to_excel(w, sheet_name='Cross_Eq_Tests', index=False)
        am.to_excel(w, sheet_name='BiProbit_AME', index=False)
        diag_df.to_excel(w, sheet_name='BiProbit_Diagnostics', index=False)
        fe_wald_df.to_excel(w, sheet_name='BiProbit_FE_Wald_Tests', index=False)

    print(f"\n  ✓ Saved BiProbit outputs to {OUT_XLSX} and {RESULTS_XLSX}")


if __name__ == '__main__':
    main()
