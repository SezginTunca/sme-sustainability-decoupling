#!/usr/bin/env python3
"""
05_robustness.py
Comprehensive robustness, multinomial configuration, and construct-validity suite
under the corrected specification (26 country FE + 3 sector FE + RECount, country-clustered SEs):
 1. Probability-scale moderation (public financial support x barriers), cluster-robust delta method + t_26
 2. Strict Q14 (excludes already climate-neutral firms, Q14=4)
 3. Survey-weighted PML probits (Binder 1983 cluster-robust sandwich):
    - Within-country SME weights (w1_sme)
    - EU-27 enterprise-population weights (w93_sme)
 4. Sector-stratified probits (country FE within each NACE macro-sector)
 5. Models without RECount (excluding potential post-treatment control)
 6. Strict 9-item RECount (Q1.1-Q1.9 only, excluding Q1.10 'None')
 7. Routed external-support seekers (Q5.3=1): both N=2,959 (complete Y1 & Y2) and N=3,055 (all valid Y2)
 8. Expanded covariate specification (+ Q7.5 cost barrier, Q7.9 label barrier, Q5.1 own finance,
    Q5.2 own expertise, Q4 environmental priority, SCR16 product seller)
 9. 4-Quadrant Multinomial Logit (Neither [0,0], Green Offerings Only [1,0],
    Net-Zero Strategy Only [0,1], Both Activities [1,1]): log-odds + discrete AMEs with clustered SEs
10. Point 9 Within-Survey Construct Validity Checks (SCR16, Q10, Q15, DX5 across configurations)
11. Master consolidation into Data/Results_Consolidated.xlsx
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.special import expit, softmax
from scipy.optimize import minimize
from patsy import dmatrices, build_design_matrices

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "Data", "Processed", "SME_Analytical_Final.csv"))
OUT = os.path.join(ROOT, "Data", "Processed", "Robustness_FE.xlsx")

IV = ['B2B_private', 'B2G_public', 'B2C', 'barrier_admin', 'barrier_expertise', 'grant_receipt', 'advice_public']
BASE = ' + '.join(IV) + ' + scr10 + scr14 + re_action_count + C(country_iso2) + C(nace_b)'
BASE_R9 = ' + '.join(IV) + ' + scr10 + scr14 + re_action_count_9 + C(country_iso2) + C(nace_b)'
NOREC = ' + '.join(IV) + ' + scr10 + scr14 + C(country_iso2) + C(nace_b)'
EXPANDED_VARS = ['barrier_cost', 'barrier_label', 'own_finance', 'own_expertise', 'q4_clean', 'sells_products']
EXPANDED = BASE + ' + ' + ' + '.join(EXPANDED_VARS)
need = IV + ['scr10', 'scr14', 're_action_count', 'country_iso2', 'nace_b', 'q9_binary', 'q14_binary']


def fit_weighted_probit_pml(formula, data, weight_col, cluster_col='country_iso2'):
    """
    Exact Binder (1983) survey-weighted pseudo-maximum likelihood (PML) probit
    with cluster-robust sandwich covariance and finite-cluster (G/(G-1))*((N-1)/(N-k)) correction.
    """
    y_mat, X_mat = dmatrices(formula, data, return_type='dataframe')
    y = y_mat.values.ravel().astype(float)
    X = X_mat.values.astype(float)
    col_names = list(X_mat.columns)
    w_raw = data.loc[y_mat.index, weight_col].values.astype(float)
    # Normalize weights to sum to N so scale matches sample size
    N, k = X.shape
    w = w_raw * (N / np.sum(w_raw))
    clusters = pd.factorize(data.loc[y_mat.index, cluster_col])[0]
    G = len(np.unique(clusters))

    # Initialize from unweighted probit
    init_res = sm.Probit(y, X).fit(disp=0)
    beta = init_res.params.copy()

    # Newton-Raphson for weighted log-likelihood
    for _ in range(100):
        xb = X @ beta
        phi = stats.norm.pdf(xb)
        Phi = np.clip(stats.norm.cdf(xb), 1e-12, 1.0 - 1e-12)
        lam = np.where(y == 1.0, phi / Phi, -phi / (1.0 - Phi))
        grad = X.T @ (w * lam)
        if np.max(np.abs(grad)) < 1e-10:
            break
        weight_hess = w * lam * (lam + xb)
        H = -(X.T * weight_hess) @ X
        step = np.linalg.solve(-H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break

    xb = X @ beta
    phi = stats.norm.pdf(xb)
    Phi = np.clip(stats.norm.cdf(xb), 1e-12, 1.0 - 1e-12)
    lam = np.where(y == 1.0, phi / Phi, -phi / (1.0 - Phi))
    scores = (w * lam)[:, None] * X
    H_inv = np.linalg.inv((X.T * (w * lam * (lam + xb))) @ X)

    meat = np.zeros((k, k))
    cluster_if = np.zeros((G, k))
    for g in range(G):
        sg = scores[clusters == g].sum(axis=0)
        meat += np.outer(sg, sg)
        cluster_if[g] = H_inv @ sg
    dof_scale = (G / (G - 1.0)) * ((N - 1.0) / (N - k))
    cov_cl = dof_scale * (H_inv @ meat @ H_inv)
    se_cl = np.sqrt(np.diag(cov_cl))
    z_vals = beta / se_cl
    p_vals = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_vals)))
    p_t26 = 2.0 * (1.0 - stats.t.cdf(np.abs(z_vals), df=G - 1))

    return {
        'params': pd.Series(beta, index=col_names),
        'bse': pd.Series(se_cl, index=col_names),
        'pvalues': pd.Series(p_vals, index=col_names),
        'p_t26': pd.Series(p_t26, index=col_names),
        'nobs': N,
        'n_clusters': G,
        'col_names': col_names,
        'cluster_if': cluster_if,
        'dof_scale': dof_scale
    }


def probit(y, formula_rhs, d, weights=None):
    f = f"{y} ~ {formula_rhs}"
    if weights is None:
        return fit_weighted_probit_pml_unweighted(f, d)
    else:
        return fit_weighted_probit_pml(f, d, weights)


def fit_weighted_probit_pml_unweighted(formula, data, cluster_col='country_iso2'):
    """Exact unweighted probit with cluster-robust sandwich covariance and cluster influence functions."""
    y_mat, X_mat = dmatrices(formula, data, return_type='dataframe')
    y = y_mat.values.ravel().astype(float)
    X = X_mat.values.astype(float)
    col_names = list(X_mat.columns)
    N, k = X.shape
    clusters = pd.factorize(data.loc[y_mat.index, cluster_col])[0]
    G = len(np.unique(clusters))

    m = sm.Probit(y, X).fit(
        disp=0, maxiter=200, cov_type='cluster',
        cov_kwds={'groups': clusters}
    )
    beta = np.asarray(m.params)
    se_cl = np.asarray(m.bse)
    p_vals = np.asarray(m.pvalues)
    z_vals = beta / se_cl
    p_t26 = 2.0 * (1.0 - stats.t.cdf(np.abs(z_vals), df=G - 1))

    xb = X @ beta
    phi = stats.norm.pdf(xb)
    Phi = np.clip(stats.norm.cdf(xb), 1e-12, 1.0 - 1e-12)
    lam = np.where(y == 1.0, phi / Phi, -phi / (1.0 - Phi))
    scores = lam[:, None] * X
    H_inv = np.linalg.inv((X.T * (lam * (lam + xb))) @ X)
    cluster_if = np.zeros((G, k))
    for g in range(G):
        cluster_if[g] = H_inv @ scores[clusters == g].sum(axis=0)
    dof_scale = (G / (G - 1.0)) * ((N - 1.0) / (N - k))

    return {
        'params': pd.Series(beta, index=col_names),
        'bse': pd.Series(se_cl, index=col_names),
        'pvalues': pd.Series(p_vals, index=col_names),
        'p_t26': pd.Series(p_t26, index=col_names),
        'nobs': N,
        'n_clusters': G,
        'col_names': col_names,
        'cluster_if': cluster_if,
        'dof_scale': dof_scale
    }


def tidy(m, label, vars_=IV):
    rows = []
    for v in vars_:
        if v in m['params'].index:
            rows.append(dict(
                spec=label, var=v,
                beta=float(m['params'][v]),
                se=float(m['bse'][v]),
                z=float(m['params'][v] / m['bse'][v]),
                p=float(m['pvalues'][v]),
                p_t26=float(m['p_t26'][v]),
                N=int(m['nobs'])
            ))
    return rows


def crosseq_diff_tidy(m1, m2, label, vars_=IV):
    """Compute cluster-robust cross-equation difference beta1 - beta2 using stacked cluster influence functions."""
    diff_rows = []
    G = m1['n_clusters']
    scale = m1['dof_scale']
    for v in vars_:
        if v in m1['params'].index and v in m2['params'].index:
            i1 = m1['col_names'].index(v)
            i2 = m2['col_names'].index(v)
            b1, b2 = float(m1['params'][v]), float(m2['params'][v])
            db = b1 - b2
            if_diff = m1['cluster_if'][:, i1] - m2['cluster_if'][:, i2]
            var_db = float(scale * np.sum(if_diff ** 2))
            se_db = float(np.sqrt(var_db))
            z_db = db / se_db
            p_db = float(2.0 * (1.0 - stats.norm.cdf(abs(z_db))))
            pt26_db = float(2.0 * (1.0 - stats.t.cdf(abs(z_db), df=G - 1)))
            diff_rows.append(dict(
                spec=label, var=v,
                beta_y1=b1, se_y1=float(m1['bse'][v]), p_y1=float(m1['pvalues'][v]),
                beta_y2=b2, se_y2=float(m2['bse'][v]), p_y2=float(m2['pvalues'][v]), p_t26_y2=float(m2['p_t26'][v]),
                diff_b1_minus_b2=db, se_diff=se_db, z_diff=z_db, p_diff=p_db, p_t26_diff=pt26_db,
                N=int(m1['nobs'])
            ))
    return diff_rows


print("=" * 80)
print("RUNNING 05_robustness.py: ROBUSTNESS, MULTINOMIAL LOGIT, AND CONSTRUCT VALIDITY")
print("=" * 80)

rows = []
diff_rows = []
d0 = df.dropna(subset=need).copy()

paired_specs = [
    ('Reference (sep. probit, FE, clustered)', BASE, d0, None, IV),
    ('No RECount', NOREC, df.dropna(subset=[c for c in need if c != 're_action_count']).copy(), None, IV),
    ('Strict 9-Item RECount (Q1.1-Q1.9)', BASE_R9, d0, None, IV),
    ('Survey-weighted PML (w1_sme)', BASE, d0.dropna(subset=['w1_sme']).copy(), 'w1_sme', IV),
    ('Population-weighted PML (w93_sme)', BASE, d0.dropna(subset=['w93_sme']).copy(), 'w93_sme', IV),
    ('Routed only (Q5.3=1, Joint Sample N=2,959)', BASE, d0[d0['q5.3'] == 1].copy(), None, IV),
    ('Expanded covariates (+Cost/Label/OwnRes/Q4/SCR16)', EXPANDED, d0.dropna(subset=EXPANDED_VARS).copy(), None, IV + EXPANDED_VARS),
    ('Product-selling SMEs only (SCR16=1)', BASE, d0[d0['sells_products'] == 1].copy(), None, IV),
]

for spec_label, formula_rhs, data_sub, w_col, v_list in paired_specs:
    m1 = probit('q9_binary', formula_rhs, data_sub, w_col)
    m2 = probit('q14_binary', formula_rhs, data_sub, w_col)
    rows += [dict(r, outcome='q9_binary') for r in tidy(m1, spec_label, vars_=v_list)]
    rows += [dict(r, outcome='q14_binary') for r in tidy(m2, spec_label, vars_=v_list)]
    diff_rows += crosseq_diff_tidy(m1, m2, spec_label, vars_=IV)

# Routed Q5.3=1 on all valid Q14 observations (N=3,055)
if 'q5.3' in df.columns:
    need_q14_only = [c for c in need if c != 'q9_binary']
    dr_q14 = df[df['q5.3'] == 1].dropna(subset=need_q14_only).copy()
    rows += [dict(r, outcome='q14_binary') for r in tidy(probit('q14_binary', BASE, dr_q14), 'Routed only (Q5.3=1, All Valid Q14 N=3,055)')]

# Strict Q14 (excludes Q14=4 already climate-neutral firms, N=10,624)
if 'q14_strict' in d0.columns:
    ds = d0.dropna(subset=['q14_strict']).copy()
    m1_s = probit('q9_binary', BASE, ds)
    m2_s = probit('q14_strict', BASE, ds)
    rows += [dict(r, outcome='q9_binary') for r in tidy(m1_s, 'Strict Q14 (Excl. Already Neutral)')]
    rows += [dict(r, outcome='q14_strict') for r in tidy(m2_s, 'Strict Q14 (Excl. Already Neutral)')]
    diff_rows += crosseq_diff_tidy(m1_s, m2_s, 'Strict Q14 (Excl. Already Neutral)', vars_=IV)

# Sector-stratified probits
for s, g in d0.groupby('nace_b'):
    rhs = ' + '.join(IV) + ' + scr10 + scr14 + re_action_count + C(country_iso2)'
    for y in ['q9_binary', 'q14_binary']:
        rows += [dict(r, outcome=y) for r in tidy(probit(y, rhs, g), f'Sector nace_b={int(s)}')]

rob = pd.DataFrame(rows)
rob_diffs = pd.DataFrame(diff_rows)

# ==============================================================================
# PROBABILITY-SCALE MODERATION (H3c): CLUSTER-ROBUST DELTA METHOD + t_26
# ==============================================================================
sub = d0.copy()
f_mod = ('q14_binary ~ grant_receipt * barrier_admin + grant_receipt * barrier_expertise + B2B_private + B2G_public + B2C '
         '+ advice_public + scr10 + scr14 + re_action_count + C(country_iso2) + C(nace_b)')
# Note: Also check original specification without advice_public vs with advice_public
f_mod_orig = ('q14_binary ~ grant_receipt * barrier_admin + grant_receipt * barrier_expertise + B2B_private + B2G_public + B2C '
              '+ scr10 + scr14 + re_action_count + C(country_iso2) + C(nace_b)')
m_mod = smf.logit(f_mod_orig, sub).fit(
    disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(sub['country_iso2'])[0]}
)
di, cov, beta = m_mod.model.data.design_info, m_mod.cov_params().values, m_mod.params.values
G_mod = sub['country_iso2'].nunique()


def pg(g, a=None, e=None):
    d = sub.copy()
    d['grant_receipt'] = g
    if a is not None:
        d['barrier_admin'] = a
    if e is not None:
        d['barrier_expertise'] = e
    X = np.asarray(build_design_matrices([di], d)[0])
    p = expit(X @ beta)
    return p.mean(), ((p * (1.0 - p))[:, None] * X).mean(0)


mod = []
for dim, kw, inter_term in [
    ('Admin', 'a', 'grant_receipt:barrier_admin'),
    ('Expertise', 'e', 'grant_receipt:barrier_expertise')
]:
    P = {(g, b): pg(g, **{kw: b}) for g in (0, 1) for b in (0, 1)}
    for lab, est, gr in [
        (f'AME(support | {dim}=0)', P[1, 0][0] - P[0, 0][0], P[1, 0][1] - P[0, 0][1]),
        (f'AME(support | {dim}=1)', P[1, 1][0] - P[0, 1][0], P[1, 1][1] - P[0, 1][1]),
        (f'Delta-AME ({dim}=1 vs 0)', (P[1, 1][0] - P[0, 1][0]) - (P[1, 0][0] - P[0, 0][0]),
         (P[1, 1][1] - P[0, 1][1]) - (P[1, 0][1] - P[0, 0][1]))
    ]:
        se = float(np.sqrt(gr @ cov @ gr))
        z_stat = float(est / se)
        p_norm = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
        p_t26 = float(2.0 * (1.0 - stats.t.cdf(abs(z_stat), df=G_mod - 1)))
        mod.append(dict(
            quantity=lab, est=float(est), se=se, z=z_stat,
            lo=float(est - 1.96 * se), hi=float(est + 1.96 * se),
            p=p_norm, p_t26=p_t26, N=int(m_mod.nobs)
        ))
    # Also record the logit interaction term itself
    b_int = float(m_mod.params[inter_term])
    se_int = float(m_mod.bse[inter_term])
    z_int = b_int / se_int
    mod.append(dict(
        quantity=f'Logit interaction ({inter_term})',
        est=b_int, se=se_int, z=z_int,
        lo=b_int - 1.96 * se_int, hi=b_int + 1.96 * se_int,
        p=float(2.0 * (1.0 - stats.norm.cdf(abs(z_int)))),
        p_t26=float(2.0 * (1.0 - stats.t.cdf(abs(z_int), df=G_mod - 1))),
        N=int(m_mod.nobs)
    ))

mod = pd.DataFrame(mod)

# ==============================================================================
# 4-QUADRANT MULTINOMIAL LOGIT (Point 4: Four Sustainability Configurations)
# ==============================================================================
print("\n[Estimating 4-Quadrant Multinomial Logit (N = 11,128, 26 country FE, 3 sector FE, RECount)]")
d_mnl = d0.copy()
# Quadrant coding:
# 0: Neither (Y1=0, Y2=0) [reference category]
# 1: Green Offerings Only (Y1=1, Y2=0)
# 2: Net-Zero Strategy Only (Y1=0, Y2=1)
# 3: Both Activities (Y1=1, Y2=1)
d_mnl['quad_4'] = np.select(
    [(d_mnl['q9_binary'] == 0) & (d_mnl['q14_binary'] == 0),
     (d_mnl['q9_binary'] == 1) & (d_mnl['q14_binary'] == 0),
     (d_mnl['q9_binary'] == 0) & (d_mnl['q14_binary'] == 1),
     (d_mnl['q9_binary'] == 1) & (d_mnl['q14_binary'] == 1)],
    [0, 1, 2, 3]
)
f_mnl = f"quad_4 ~ {BASE}"
mnl_fit = smf.mnlogit(f_mnl, d_mnl).fit(
    disp=0, maxiter=200, cov_type='cluster',
    cov_kwds={'groups': pd.factorize(d_mnl['country_iso2'])[0]}
)
G_mnl = d_mnl['country_iso2'].nunique()

# Extract log-odds coefficients for focal variables + controls across categories 1, 2, 3
mnl_vars = IV + ['scr10', 'scr14', 're_action_count']
quad_labels = {
    0: 'Neither (0,0)',
    1: 'Green Offerings Only (1,0)',
    2: 'Net-Zero Strategy Only (0,1)',
    3: 'Both Activities (1,1)'
}

mnl_rows = []
for v in mnl_vars:
    row = {'variable': v}
    for j in [1, 2, 3]:
        col_idx = j - 1
        b = float(mnl_fit.params.loc[v, col_idx])
        se = float(mnl_fit.bse.loc[v, col_idx])
        z = b / se
        p_n = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
        p_t = float(2.0 * (1.0 - stats.t.cdf(abs(z), df=G_mnl - 1)))
        row[f'beta_q{j}'] = b
        row[f'se_q{j}'] = se
        row[f'z_q{j}'] = z
        row[f'p_q{j}'] = p_n
        row[f'p_t26_q{j}'] = p_t
    mnl_rows.append(row)

# Compute sample-average discrete marginal effects (AMEs) for all 4 quadrants with cluster-robust delta method
di_mnl = mnl_fit.model.data.design_info
X_mnl = np.asarray(build_design_matrices([di_mnl], d_mnl)[0])
N_mnl, K_mnl = X_mnl.shape
# Beta matrix shape (K, 3) for j=1,2,3; cov matrix shape (3*K, 3*K) in statsmodels MNLogit
# Note: statsmodels orders MNLogit params column-major by equation (j=1 all K, then j=2 all K, then j=3 all K) or row-major?
# Verify parameter ordering in mnl_fit.cov_params()
cov_mnl_df = mnl_fit.cov_params()
param_names_mnl = list(mnl_fit.params.index)
B_mnl = mnl_fit.params.values  # shape (K, 3)

# Construct covariance matrix in (3*K, 3*K) ordered as [eq1_k0..eq1_kK, eq2_k0..eq2_kK, eq3_k0..eq3_kK]
# In statsmodels MNLogit, MultiIndex on cov_params() is (eq_label, param_name) where eq_label in ['1', '2', '3'] or [0, 1, 2]
cov_mnl = cov_mnl_df.values


def mnl_probs_and_grad(X_eval, B_mat):
    """
    Returns mean probability vector (4,) and gradient wrt vec_F(B_mat) of shape (4, 3*K)
    where vec_F stacks column 0 (eq 1), column 1 (eq 2), column 2 (eq 3).
    """
    logits = np.column_stack([np.zeros(len(X_eval)), X_eval @ B_mat])  # (N, 4)
    P = softmax(logits, axis=1)  # (N, 4)
    p_mean = P.mean(axis=0)
    # Derivative of P_{i,m} wrt B_{k,r} (where r in {1,2,3}):
    # d P_{i,m} / d eta_{i,r} = P_{i,m} * (1(m==r) - P_{i,r})
    grad = np.zeros((4, 3 * K_mnl))
    for m in range(4):
        for r_idx, r in enumerate([1, 2, 3]):
            w_mr = P[:, m] * ((1.0 if m == r else 0.0) - P[:, r])  # (N,)
            grad[m, r_idx * K_mnl:(r_idx + 1) * K_mnl] = (w_mr[:, None] * X_eval).mean(axis=0)
    return p_mean, grad


for i, v in enumerate(mnl_vars):
    is_binary = set(np.unique(d_mnl[v].dropna())).issubset({0, 1, 0.0, 1.0})
    if is_binary:
        d1 = d_mnl.copy(); d1[v] = 1
        d0_v = d_mnl.copy(); d0_v[v] = 0
        X1 = np.asarray(build_design_matrices([di_mnl], d1)[0])
        X0 = np.asarray(build_design_matrices([di_mnl], d0_v)[0])
        p1, g1 = mnl_probs_and_grad(X1, B_mnl)
        p0, g0 = mnl_probs_and_grad(X0, B_mnl)
        ame_vec = p1 - p0
        grad_ame = g1 - g0
    else:
        # Continuous variable: analytical average marginal effect
        eps = 1e-5
        d_plus = d_mnl.copy(); d_plus[v] = d_mnl[v] + eps
        d_minus = d_mnl.copy(); d_minus[v] = d_mnl[v] - eps
        Xp = np.asarray(build_design_matrices([di_mnl], d_plus)[0])
        Xm = np.asarray(build_design_matrices([di_mnl], d_minus)[0])
        pp, gp = mnl_probs_and_grad(Xp, B_mnl)
        pm, gm = mnl_probs_and_grad(Xm, B_mnl)
        ame_vec = (pp - pm) / (2.0 * eps)
        grad_ame = (gp - gm) / (2.0 * eps)

    for m in range(4):
        est_m = float(ame_vec[m])
        se_m = float(np.sqrt(grad_ame[m] @ cov_mnl @ grad_ame[m]))
        z_m = est_m / se_m
        p_m = float(2.0 * (1.0 - stats.norm.cdf(abs(z_m))))
        p_t26_m = float(2.0 * (1.0 - stats.t.cdf(abs(z_m), df=G_mnl - 1)))
        mnl_rows[i][f'ame_q{m}'] = est_m
        mnl_rows[i][f'ame_se_q{m}'] = se_m
        mnl_rows[i][f'ame_p_q{m}'] = p_m
        mnl_rows[i][f'ame_p_t26_q{m}'] = p_t26_m

mnl_df = pd.DataFrame(mnl_rows)

# ==============================================================================
# POINT 9: WITHIN-SURVEY CONSTRUCT VALIDITY & CONFIGURATION PROFILE TABLE
# ==============================================================================
print("\n[Building Point 9 Construct Validity & Configuration Profile Summary]")
p9_rows = []
for q_val, q_name in quad_labels.items():
    sub_q = d_mnl[d_mnl['quad_4'] == q_val]
    n_q = len(sub_q)
    share_q = 100.0 * n_q / len(d_mnl)
    # Product seller share (SCR16)
    scr16_pct = 100.0 * sub_q['sells_products'].mean() if 'sells_products' in sub_q.columns else np.nan
    # Green jobs > 0 (DX5)
    dx5_pct = 100.0 * sub_q['green_jobs_any'].mean() if 'green_jobs_any' in sub_q.columns else np.nan
    dx5_mean = sub_q['green_jobs_fte'].mean() if 'green_jobs_fte' in sub_q.columns else np.nan
    # RE action count (9 items)
    re9_mean = sub_q['re_action_count_9'].mean() if 're_action_count_9' in sub_q.columns else sub_q['re_action_count'].mean()
    # Public financial support (Q6.1)
    grant_pct = 100.0 * sub_q['grant_receipt'].mean()
    # B2C, B2G, B2B shares
    b2c_pct = 100.0 * sub_q['B2C'].mean()
    b2g_pct = 100.0 * sub_q['B2G_public'].mean()
    b2b_pct = 100.0 * sub_q['B2B_private'].mean()
    # Admin & Expertise barriers
    admin_pct = 100.0 * sub_q['barrier_admin'].mean()
    exp_pct = 100.0 * sub_q['barrier_expertise'].mean()
    # Planned resource-efficiency investment intensity (Q4: 1=Nothing, 2=<1%, 3=1-5%, 4=6-10%, 5=11-30%, 6=>30%)
    q4_valid = sub_q['q4_clean'].dropna() if 'q4_clean' in sub_q.columns else pd.Series(dtype=float)
    q4_any_inv = 100.0 * (sub_q['q4_clean'] >= 2).mean() if 'q4_clean' in sub_q.columns else np.nan
    q4_high_inv = 100.0 * (sub_q['q4_clean'] >= 4).mean() if 'q4_clean' in sub_q.columns else np.nan
    q4_high_valid = 100.0 * (q4_valid >= 4).mean() if len(q4_valid) > 0 else np.nan
    # Q15 concrete carbon actions among net-zero adopters (routed to Q14=1)
    q15_valid = sub_q['q15_count'].dropna() if 'q15_count' in sub_q.columns else pd.Series(dtype=float)
    q15_mean = float(q15_valid.mean()) if len(q15_valid) > 0 else np.nan
    # Q10 green revenue > 5% (Q10 in {2..6}) and > 30% (Q10 in {4..6}) among green product sellers
    q10_valid = sub_q['q10_clean'].dropna() if 'q10_clean' in sub_q.columns else pd.Series(dtype=float)
    q10_gt5_pct = float(100.0 * (q10_valid >= 2).mean()) if len(q10_valid) > 0 else np.nan
    q10_gt30_pct = float(100.0 * (q10_valid >= 4).mean()) if len(q10_valid) > 0 else np.nan

    p9_rows.append({
        'Quadrant_ID': q_val,
        'Configuration': q_name,
        'N': n_q,
        'Sample_Share_Pct': round(share_q, 2),
        'Sells_Products_SCR16_Pct': round(scr16_pct, 2),
        'B2C_Pct': round(b2c_pct, 2),
        'B2G_Pct': round(b2g_pct, 2),
        'B2B_Pct': round(b2b_pct, 2),
        'Barrier_Admin_Pct': round(admin_pct, 2),
        'Barrier_Expertise_Pct': round(exp_pct, 2),
        'Public_Support_Q6_1_Pct': round(grant_pct, 2),
        'RE_Actions_9_Mean': round(re9_mean, 2),
        'Green_Jobs_Any_DX5_Pct': round(dx5_pct, 2),
        'Green_Jobs_FTE_Mean': round(dx5_mean, 2) if not np.isnan(dx5_mean) else np.nan,
        'Plan_Invest_Any_Q4_Pct': round(q4_any_inv, 2),
        'Plan_Invest_GT5Pct_Q4_Pct': round(q4_high_inv, 2),
        'Plan_Invest_GT5Pct_Q4_Valid_Pct': round(q4_high_valid, 2) if not np.isnan(q4_high_valid) else np.nan,
        'Q10_Green_Rev_GT5Pct_Pct': round(q10_gt5_pct, 2) if not np.isnan(q10_gt5_pct) else np.nan,
        'Q10_Green_Rev_GT30Pct_Pct': round(q10_gt30_pct, 2) if not np.isnan(q10_gt30_pct) else np.nan,
        'Q15_Carbon_Action_Mean': round(q15_mean, 2) if not np.isnan(q15_mean) else np.nan
    })

p9_df = pd.DataFrame(p9_rows)

with pd.ExcelWriter(OUT) as w:
    rob.to_excel(w, sheet_name='robustness', index=False)
    rob_diffs.to_excel(w, sheet_name='robustness_crosseq_diffs', index=False)
    mod.to_excel(w, sheet_name='moderation_clustered', index=False)
    mnl_df.to_excel(w, sheet_name='mnlogit_4quadrants', index=False)
    p9_df.to_excel(w, sheet_name='point9_construct_validity', index=False)

# Rebuild Data/Results_Consolidated.xlsx cleanly (removing any obsolete pre-revision sheets)
RES_CONSOL = os.path.join(ROOT, "Data", "Results_Consolidated.xlsx")
BP_XLSX = os.path.join(ROOT, "Data", "Processed", "BiProbit_Joint_FE.xlsx")
INP_XLSX = os.path.join(ROOT, "Data", "Input_Data_Consolidated.xlsx")

preserved_sheets = {}
keep_from_existing = [
    'CausalForest_ATE', 'CausalForest_CATEs',
    'PSM_ATT', 'PSM_Balance', 'PSM_Sensitivity_Specs', 'Rosenbaum_Bounds',
    'DI_Moderation_OLS', 'DI_Country_Level',
    'GreenPremium_CrossTab', 'GreenPremium_OrderedProbit'
]
if os.path.exists(RES_CONSOL):
    existing_xl = pd.ExcelFile(RES_CONSOL)
    for sh in keep_from_existing:
        if sh in existing_xl.sheet_names:
            preserved_sheets[sh] = pd.read_excel(RES_CONSOL, sheet_name=sh)

readme_df = pd.DataFrame([
    {'Field': 'Manuscript Title', 'Details': 'Green Products, Net-Zero Plans: Market Channels and Finance in European SMEs'},
    {'Field': 'Authors', 'Details': 'Sezgin Tunca, Mausam Budhathoki, and Hari Gopal Risal'},
    {'Field': 'Target Journal', 'Details': 'Business Strategy and the Environment'},
    {'Field': 'Data Source', 'Details': 'Flash Eurobarometer 498 (ZA7806, 2021) & Eurostat Digital Intensity Index (isoc_e_dii, 2021)'},
    {'Field': 'Analytical Samples', 'Details': 'N=13,322 EU-27 SMEs; N=12,205 4-Quadrant Typology; N=11,521 IPW/CATE; N=11,128 Primary Joint Econometric Sample'},
    {'Field': 'Primary Specification', 'Details': 'Joint ML Bivariate Probit with 26 country FE, 3 NACE sector FE, RECount, and 27 country-clustered SEs + finite-cluster t_26'},
    {'Field': 'Disturbance Correlation (rho)', 'Details': 'rho = 0.2566 (cluster SE = 0.0257, z = 9.97, p < 0.001)'},
    {'Field': 'Verification Script', 'Details': 'Scripts/06_verify_all_results.py'}
])

with pd.ExcelWriter(RES_CONSOL, engine='openpyxl', mode='w') as writer:
    readme_df.to_excel(writer, sheet_name='Readme_Metadata', index=False)
    if os.path.exists(INP_XLSX):
        inp_xl = pd.ExcelFile(INP_XLSX)
        for sh in ['Sample_Waterfall', 'Table1_Typology_12205', 'Missing_Data_Attrition']:
            if sh in inp_xl.sheet_names:
                pd.read_excel(INP_XLSX, sheet_name=sh).to_excel(writer, sheet_name=sh, index=False)
    if os.path.exists(BP_XLSX):
        pd.read_excel(BP_XLSX, sheet_name='coef').to_excel(writer, sheet_name='BiProbit_Joint_FE', index=False)
        pd.read_excel(BP_XLSX, sheet_name='cross_eq_tests').to_excel(writer, sheet_name='Cross_Eq_Tests', index=False)
        pd.read_excel(BP_XLSX, sheet_name='AME').to_excel(writer, sheet_name='BiProbit_AME', index=False)
        pd.read_excel(BP_XLSX, sheet_name='diagnostics').to_excel(writer, sheet_name='BiProbit_Diagnostics', index=False)
    mnl_df.to_excel(writer, sheet_name='MNLogit_4Quadrants', index=False)
    for sh in keep_from_existing:
        if sh in preserved_sheets:
            preserved_sheets[sh].to_excel(writer, sheet_name=sh, index=False)
    rob.to_excel(writer, sheet_name='Robustness_All', index=False)
    rob_diffs.to_excel(writer, sheet_name='Robustness_CrossEq_Diffs', index=False)
    mod.to_excel(writer, sheet_name='Moderation_Clustered', index=False)
    p9_df.to_excel(writer, sheet_name='Point9_Construct_Validity', index=False)

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)
print("\n[Moderation (Clustered SEs + t_26)]")
print(mod.round(4).to_string(index=False))

print("\n[4-Quadrant Multinomial Logit AMEs (Percentage Points)]")
ame_cols = ['variable', 'ame_q0', 'ame_p_q0', 'ame_q1', 'ame_p_q1', 'ame_q2', 'ame_p_q2', 'ame_q3', 'ame_p_q3']
print(mnl_df[ame_cols].round(4).to_string(index=False))

print("\n[Point 9 Configuration Profile & Construct Validity Table]")
print(p9_df.to_string(index=False))

print("\n[Key Robustness Cross-Equation Differences]")
print(rob_diffs.round(4).to_string(index=False))
print(f"\n✓ Consolidated all sheets cleanly into: {RES_CONSOL}")
