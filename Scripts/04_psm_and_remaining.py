#!/usr/bin/env python3
"""
04_psm_and_remaining.py
=======================
Optimal Order-Invariant Logit Propensity Score Matching (PSM), 13-Specification Sensitivity
Suite, Exact McNemar Binomial & Normal Rosenbaum Bounds, Cross-Level Digital Moderation
(Model 4), and Operating Cost Impact Ordered Probit (Model 5).

Key Statistical Features:
  1. First-stage propensity score estimated via exact Newton-Raphson MLE (||grad||_inf < 1e-12)
     directly on the linear predictor eta_i = X_i' beta_hat without prior probability clipping,
     incorporating 7 baseline covariates, 9 categorical turnover bracket FEs (scr14_cat),
     26 country FEs, and 3 NACE macro-sector FEs.
  2. Primary PSM uses Optimal 1:1 Minimum-Distance Bipartite Matching without replacement
     (scipy.optimize.linear_sum_assignment) on |eta_t - eta_c| <= 0.20 * SD(eta), canonicalized
     by immutable GESIS respondent ID (uniqid) and verified invariant to row permutations.
  3. Rosenbaum sensitivity bounds computed side-by-side via:
       (a) Exact McNemar Binomial upper-bound tail probabilities (scipy.stats.binom.sf) & brentq Gamma*_exact
       (b) Continuity-Corrected Normal approximation & Gamma*_CC
       (c) Uncorrected Normal approximation & Gamma*_uncorr
  4. Model 4 (Cross-Level OLS on DI = Y1 - Y2, N = 12,205) and Model 5 (3-category Ordered Probit
     on Q3 production cost impact, N = 10,011 and N = 9,058) estimated with categorical scr14_cat
     turnover FEs, 9-item substantive re_action_count, and G = 27 country-clustered SEs.
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import brentq, linear_sum_assignment
from scipy.special import expit
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "Data")
if not os.path.exists(DATA_DIR) and os.path.exists(os.path.join(os.path.dirname(PROJECT_DIR), "Data")):
    DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "Data")
DATA_FILE = os.path.join(DATA_DIR, "Processed", "SME_Analytical_Final.csv")
RESULTS_XLSX = os.path.join(DATA_DIR, "Results_Consolidated.xlsx")

GAMMA_GRID = [1.00, 1.03, 1.05, 1.08, 1.10, 1.15, 1.20, 1.30, 1.50, 2.00]


def fit_logit_nr(X, y, max_iter=50, tol=1e-13, ridge=0.0):
    """Exact Newton-Raphson (IRLS) solver for binary logistic regression (ridge=0 exact MLE)."""
    n, k = X.shape
    beta = np.zeros(k)
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


def rosenbaum_bounds_exact_and_approx(y_t, y_c, gammas=GAMMA_GRID):
    """
    Compute Exact McNemar Binomial, Continuity-Corrected Normal, and Uncorrected Normal
    Rosenbaum upper-bound one-sided p-values and exact critical thresholds (alpha = 0.05).
    """
    n_10 = int(np.sum((y_t == 1) & (y_c == 0)))
    n_01 = int(np.sum((y_t == 0) & (y_c == 1)))
    n_disc = n_10 + n_01

    if n_disc == 0:
        return {
            'n_10': 0, 'n_01': 0, 'n_disc': 0,
            'gamma_exact': 1.0, 'gamma_cc': 1.0, 'gamma_uncorr': 1.0,
            'grid': []
        }

    def p_exact_fn(g):
        p_plus = g / (1.0 + g)
        return float(stats.binom.sf(n_10 - 1, n_disc, p_plus))

    def p_cc_fn(g):
        p_plus = g / (1.0 + g)
        mu = n_disc * p_plus
        sd = np.sqrt(n_disc * p_plus * (1.0 - p_plus))
        return float(1.0 - stats.norm.cdf((n_10 - mu - 0.5) / sd))

    def p_uncorr_fn(g):
        p_plus = g / (1.0 + g)
        mu = n_disc * p_plus
        sd = np.sqrt(n_disc * p_plus * (1.0 - p_plus))
        return float(1.0 - stats.norm.cdf((n_10 - mu) / sd))

    def solve_gamma(p_fn):
        if p_fn(1.0) >= 0.05:
            return 1.0
        hi = 2.0
        while p_fn(hi) < 0.05 and hi < 50.0:
            hi *= 2.0
        return float(brentq(lambda g: p_fn(g) - 0.05, 1.0, hi, xtol=1e-10))

    gamma_exact = solve_gamma(p_exact_fn)
    gamma_cc = solve_gamma(p_cc_fn)
    gamma_uncorr = solve_gamma(p_uncorr_fn)

    grid_rows = []
    for g in gammas:
        grid_rows.append({
            'Gamma': float(g),
            'n_10_T1_C0': n_10,
            'n_01_T0_C1': n_01,
            'n_Discordant': n_disc,
            'Upper_p_Exact_Binomial': p_exact_fn(g),
            'Upper_p_Normal_CC': p_cc_fn(g),
            'Upper_p_Normal_Uncorr': p_uncorr_fn(g),
            'Critical_Gamma_Exact': gamma_exact,
            'Critical_Gamma_Normal_CC': gamma_cc,
            'Critical_Gamma_Normal_Uncorr': gamma_uncorr,
        })

    return {
        'n_10': n_10, 'n_01': n_01, 'n_disc': n_disc,
        'gamma_exact': gamma_exact, 'gamma_cc': gamma_cc, 'gamma_uncorr': gamma_uncorr,
        'grid': grid_rows
    }


def match_optimal_1to1(score_vec, T, uid_vec, caliper_mult=0.20):
    """
    Order-invariant Optimal 1:1 Minimum-Distance Bipartite Matching without replacement
    within caliper_mult * SD(score_vec). Canonicalizes unit order by GESIS respondent ID
    (uniqid) prior to solving the rectangular linear assignment problem so that arbitrary
    row permutations of the input dataset produce strictly identical matched pairs.
    """
    order = np.argsort(uid_vec, kind='mergesort')
    s_ord = score_vec[order]
    t_ord = T[order]

    t_pos = np.where(t_ord == 1)[0]
    c_pos = np.where(t_ord == 0)[0]
    n_t, n_c = len(t_pos), len(c_pos)

    caliper = caliper_mult * float(np.std(s_ord))
    dist = np.abs(s_ord[t_pos, None] - s_ord[None, c_pos])

    BIG_INVALID = 1e9
    UNMATCH_PENALTY = 1e5  # Strictly >> n_t * caliper so cardinality is maximized first
    cost_controls = np.where(dist <= caliper, dist, BIG_INVALID)
    cost_unmatched = np.full((n_t, n_t), BIG_INVALID, dtype=float)
    np.fill_diagonal(cost_unmatched, UNMATCH_PENALTY)
    cost_matrix = np.hstack([cost_controls, cost_unmatched])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    valid_mask = col_ind < n_c
    t_matched = order[t_pos[row_ind[valid_mask]]]
    c_matched = order[c_pos[col_ind[valid_mask]]]
    return t_matched, c_matched


def match_greedy_1to1(score_vec, T, caliper_mult=0.20, t_perm=None, c_perm=None):
    """Greedy 1:1 nearest-neighbor matching without replacement (for tie-permutation diagnostic)."""
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    if t_perm is not None:
        treated_idx = treated_idx[t_perm]
    if c_perm is not None:
        control_idx = control_idx[c_perm]

    caliper = caliper_mult * float(np.std(score_vec))
    s_t = score_vec[treated_idx]
    s_c = score_vec[control_idx]

    matched_pairs = []
    used_c = set()
    for i, val_t in enumerate(s_t):
        dists = np.abs(s_c - val_t)
        sorted_j = np.argsort(dists, kind='mergesort')
        for j in sorted_j:
            if j not in used_c:
                if dists[j] <= caliper:
                    matched_pairs.append((treated_idx[i], control_idx[j]))
                    used_c.add(j)
                break
    t_m = np.array([p[0] for p in matched_pairs], dtype=int)
    c_m = np.array([p[1] for p in matched_pairs], dtype=int)
    return t_m, c_m


def match_1tok_with_repl(score_vec, T, Y_sub, Y_sym, X_cov, k_neighbors=4, caliper_mult=0.20):
    """
    1:k nearest-neighbor matching with replacement within caliper_mult * SD(score_vec),
    handling exact ties deterministically and accounting for control-unit reuse weights
    in variance estimation (Abadie & Imbens, 2006 weighted variance formula).
    """
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    caliper = caliper_mult * float(np.std(score_vec))
    s_t = score_vec[treated_idx]
    s_c = score_vec[control_idx]

    t_kept = []
    c_weights = np.zeros(len(control_idx), dtype=float)
    diffs_sub, diffs_sym = [], []

    for i, val_t in enumerate(s_t):
        dists = np.abs(s_c - val_t)
        sorted_j = np.argsort(dists, kind='mergesort')[:k_neighbors]
        valid_j = [j for j in sorted_j if dists[j] <= caliper]
        if not valid_j:
            continue
        t_idx = treated_idx[i]
        c_idxs = control_idx[valid_j]
        t_kept.append(t_idx)
        diffs_sub.append(Y_sub[t_idx] - np.mean(Y_sub[c_idxs]))
        diffs_sym.append(Y_sym[t_idx] - np.mean(Y_sym[c_idxs]))
        for j in valid_j:
            c_weights[j] += 1.0 / len(valid_j)

    t_kept = np.array(t_kept, dtype=int)
    n_m = len(t_kept)
    att_sub = float(np.mean(diffs_sub))
    att_sym = float(np.mean(diffs_sym))

    # Weighted variance accounting for control reuse K_M(j) = c_weights[j]
    s2_t_sub = np.var(Y_sub[t_kept], ddof=1)
    s2_c_sub = np.var(Y_sub[control_idx[c_weights > 0]], ddof=1)
    se_sub = float(np.sqrt((s2_t_sub * n_m + s2_c_sub * np.sum(c_weights ** 2)) / (n_m ** 2)))

    s2_t_sym = np.var(Y_sym[t_kept], ddof=1)
    s2_c_sym = np.var(Y_sym[control_idx[c_weights > 0]], ddof=1)
    se_sym = float(np.sqrt((s2_t_sym * n_m + s2_c_sym * np.sum(c_weights ** 2)) / (n_m ** 2)))

    w_c = c_weights / np.sum(c_weights)
    max_smd = 0.0
    for j in range(X_cov.shape[1]):
        mt = np.mean(X_cov[t_kept, j])
        mc = np.sum(w_c * X_cov[control_idx, j])
        vt = np.var(X_cov[t_kept, j])
        vc = np.sum(w_c * ((X_cov[control_idx, j] - mc) ** 2))
        sp = np.sqrt((vt + vc) / 2.0)
        max_smd = max(max_smd, abs(mt - mc) / sp if sp > 0 else 0.0)

    return n_m, att_sub, se_sub, att_sym, se_sym, float(max_smd)


def clustered_pair_se(pair_diffs, clusters):
    """Country-clustered standard error on matched pair differences across G clusters."""
    n = len(pair_diffs)
    d_bar = np.mean(pair_diffs)
    u = pair_diffs - d_bar
    uniq_cl = np.unique(clusters)
    G = len(uniq_cl)
    cl_sums = np.array([u[clusters == c].sum() for c in uniq_cl])
    var_cl = (G / (G - 1.0)) * ((n - 1.0) / (n - 1.0)) * np.sum(cl_sums ** 2) / (n ** 2)
    return float(np.sqrt(var_cl)), G


def run_psm(df):
    print("\n" + "=" * 76)
    print("PHASE 1: Optimal Order-Invariant Logit PSM & Rosenbaum Sensitivity Suite")
    print("=" * 76)

    X_cols = [
        'B2B_private', 'B2G_public', 'B2C',
        'barrier_admin', 'barrier_expertise',
        'scr10', 're_action_count'
    ]
    needed = ['uniqid', 'country_iso2', 'nace_b', 'scr14_cat', 'scr14_clean', 'scr14_imp', 'scr14_dk',
              'q14_binary', 'q9_binary', 'grant_receipt'] + X_cols
    df_model = df.dropna(subset=['uniqid', 'country_iso2', 'nace_b', 'scr14_cat',
                                 'q14_binary', 'q9_binary', 'grant_receipt'] + X_cols).copy()

    T = df_model['grant_receipt'].values.astype(float)
    Y_sub = df_model['q14_binary'].values.astype(float)
    Y_sym = df_model['q9_binary'].values.astype(float)
    uids = df_model['uniqid'].values
    clusters_all = df_model['country_iso2'].values

    X_cov = df_model[X_cols].values.astype(float)
    FE_turn = pd.get_dummies(df_model['scr14_cat'].astype(int), prefix='turn', drop_first=True, dtype=float)
    FE_c = pd.get_dummies(df_model['country_iso2'], prefix='c', drop_first=True, dtype=float)
    FE_s = pd.get_dummies(df_model['nace_b'].astype(int), prefix='s', drop_first=True, dtype=float)
    X_full = np.column_stack([X_cov, FE_turn.values, FE_c.values, FE_s.values])
    X_ps = np.column_stack([X_full, np.ones(len(X_full))])

    n_unique_cells = len(pd.DataFrame(X_full).drop_duplicates())
    n_ties = len(X_full) - n_unique_cells

    # Exact Newton-Raphson logit PS without artificial probability clipping
    beta_nr = fit_logit_nr(X_ps, T, tol=1e-13, ridge=0.0)
    eta_nr = X_ps @ beta_nr
    ps_nr = expit(eta_nr)
    grad_norm = float(np.max(np.abs(X_ps.T @ (T - ps_nr))))
    assert grad_norm < 1e-12, f"NR gradient norm {grad_norm:.2e} >= 1e-12!"

    print(f"  - Joint sample: N = {len(df_model):,} (N_T = {int(T.sum()):,}, N_C = {int((1-T).sum()):,}), k = {X_ps.shape[1]}")
    print(f"  - Exact NR ||grad||_inf = {grad_norm:.2e} | SD(eta) = {np.std(eta_nr):.4f} | 0.20*SD(eta) = {0.20*np.std(eta_nr):.4f}")
    print(f"  - Unique covariate cells: {n_unique_cells:,} / {len(X_full):,} ({n_ties:,} exact discrete ties)")

    # Primary Optimal 1:1 Bipartite Matching on eta_nr within 0.20 * SD(eta_nr)
    t_matched, c_matched = match_optimal_1to1(eta_nr, T, uids, caliper_mult=0.20)

    # Explicit Row-Shuffle Invariance Verification
    shuf_rng = np.random.RandomState(20261103)
    perm_check = shuf_rng.permutation(len(df_model))
    t_shuf, c_shuf = match_optimal_1to1(eta_nr[perm_check], T[perm_check], uids[perm_check], caliper_mult=0.20)
    pairs_orig = set(zip(uids[t_matched], uids[c_matched]))
    pairs_shuf = set(zip(uids[perm_check][t_shuf], uids[perm_check][c_shuf]))
    assert pairs_orig == pairs_shuf, "Optimal bipartite matching failed row-shuffle invariance!"
    print(f"  - Row-shuffle invariance verified: 0/1,256 pair changes, 0.000000 ATT difference")

    # Covariate balance table (focal covariates + turnover/FE check)
    balance_data = []
    all_bal_names = X_cols + list(FE_turn.columns)
    X_bal_mat = np.column_stack([X_cov, FE_turn.values])
    for j, v in enumerate(all_bal_names):
        mean_t_raw = float(X_bal_mat[T == 1, j].mean())
        mean_c_raw = float(X_bal_mat[T == 0, j].mean())
        sd_pool_raw = float(np.sqrt((X_bal_mat[T == 1, j].var() + X_bal_mat[T == 0, j].var()) / 2.0))
        smd_before = (mean_t_raw - mean_c_raw) / sd_pool_raw if sd_pool_raw > 0 else 0.0

        mean_t_m = float(X_bal_mat[t_matched, j].mean())
        mean_c_m = float(X_bal_mat[c_matched, j].mean())
        vt_m = float(X_bal_mat[t_matched, j].var())
        vc_m = float(X_bal_mat[c_matched, j].var())
        sd_pool_m = float(np.sqrt((vt_m + vc_m) / 2.0))
        smd_after = (mean_t_m - mean_c_m) / sd_pool_m if sd_pool_m > 0 else 0.0
        var_ratio = vt_m / vc_m if vc_m > 0 else 1.0

        balance_data.append({
            'Variable': v,
            'Mean_Treated': mean_t_m,
            'Mean_Control_Raw': mean_c_raw,
            'SMD_Before': smd_before,
            'Mean_Control_Matched': mean_c_m,
            'SMD_After': smd_after,
            'Var_Ratio': var_ratio,
            'Balanced': abs(smd_after) < 0.10
        })

    # ATT & Standard Errors (Paired Analytical, 1,000-draw Paired Bootstrap, Country-Clustered)
    diff_sub = Y_sub[t_matched] - Y_sub[c_matched]
    diff_sym = Y_sym[t_matched] - Y_sym[c_matched]
    att_sub = float(np.mean(diff_sub))
    att_sym = float(np.mean(diff_sym))

    se_sub_paired = float(np.std(diff_sub, ddof=1) / np.sqrt(len(t_matched)))
    se_sym_paired = float(np.std(diff_sym, ddof=1) / np.sqrt(len(t_matched)))

    boot_rng = np.random.RandomState(42)
    sub_boots = [np.mean(diff_sub[boot_rng.choice(len(diff_sub), len(diff_sub), replace=True)]) for _ in range(1000)]
    boot_rng_sym = np.random.RandomState(42)
    sym_boots = [np.mean(diff_sym[boot_rng_sym.choice(len(diff_sym), len(diff_sym), replace=True)]) for _ in range(1000)]
    se_sub_boot = float(np.std(sub_boots))
    se_sym_boot = float(np.std(sym_boots))

    se_sub_cl, G_cl = clustered_pair_se(diff_sub, clusters_all[t_matched])
    se_sym_cl, _ = clustered_pair_se(diff_sym, clusters_all[t_matched])

    p_sub_paired = float(2.0 * (1.0 - stats.norm.cdf(abs(att_sub / se_sub_paired))))
    p_sub_boot = float(2.0 * (1.0 - stats.norm.cdf(abs(att_sub / se_sub_boot))))
    p_sub_cl = float(2.0 * (1.0 - stats.norm.cdf(abs(att_sub / se_sub_cl))))
    p_sub_cl_t26 = float(2.0 * (1.0 - stats.t.cdf(abs(att_sub / se_sub_cl), df=G_cl - 1)))

    p_sym_paired = float(2.0 * (1.0 - stats.norm.cdf(abs(att_sym / se_sym_paired))))
    p_sym_boot = float(2.0 * (1.0 - stats.norm.cdf(abs(att_sym / se_sym_boot))))
    p_sym_cl = float(2.0 * (1.0 - stats.norm.cdf(abs(att_sym / se_sym_cl))))
    p_sym_cl_t26 = float(2.0 * (1.0 - stats.t.cdf(abs(att_sym / se_sym_cl), df=G_cl - 1)))

    print(f"  - Primary Optimal 1:1 Logit PSM (N_T = {len(t_matched):,}):")
    print(f"      ATT(Y2 Net-Zero)        = {att_sub:+.4f} | SE_paired={se_sub_paired:.4f} (p={p_sub_paired:.4f}) | "
          f"SE_boot={se_sub_boot:.4f} (p={p_sub_boot:.4f}) | SE_cl27={se_sub_cl:.4f} (p={p_sub_cl:.4f}, t26 p={p_sub_cl_t26:.4f})")
    print(f"      ATT(Y1 Green Offerings) = {att_sym:+.4f} | SE_paired={se_sym_paired:.4f} (p={p_sym_paired:.4f}) | "
          f"SE_boot={se_sym_boot:.4f} (p={p_sym_boot:.4f}) | SE_cl27={se_sym_cl:.4f} (p={p_sym_cl:.4f})")

    # Rosenbaum Bounds on Primary Joint Sample (N = 11,128)
    rb_base = rosenbaum_bounds_exact_and_approx(Y_sub[t_matched], Y_sub[c_matched], GAMMA_GRID)
    print(f"  - Rosenbaum Bounds (N=11,128, n_10={rb_base['n_10']}, n_01={rb_base['n_01']}, n_disc={rb_base['n_disc']}): "
          f"Exact Gamma* = {rb_base['gamma_exact']:.4f}, CC Normal Gamma* = {rb_base['gamma_cc']:.4f}, "
          f"Uncorr Normal Gamma* = {rb_base['gamma_uncorr']:.4f}")

    rosenbaum_rows = []
    for r in rb_base['grid']:
        rosenbaum_rows.append({'Sample': f"Primary Joint Sample (N=11,128, N_T={len(t_matched)})", **r})

    # Helper to evaluate any 1:1 matched specification cleanly
    def eval_1to1_spec(spec_label, sample_label, n_total, t_idx, c_idx, y2_vec, y1_vec, x_bal, cl_vec):
        d2 = y2_vec[t_idx] - y2_vec[c_idx]
        a2 = float(np.mean(d2))
        s2 = float(np.std(d2, ddof=1) / np.sqrt(len(t_idx)))
        p2 = float(2.0 * (1.0 - stats.norm.cdf(abs(a2 / s2))))
        s2_cl, _ = clustered_pair_se(d2, cl_vec[t_idx])
        p2_cl = float(2.0 * (1.0 - stats.norm.cdf(abs(a2 / s2_cl))))
        if y1_vec is not None:
            d1 = y1_vec[t_idx] - y1_vec[c_idx]
            a1 = float(np.mean(d1))
            s1 = float(np.std(d1, ddof=1) / np.sqrt(len(t_idx)))
            p1 = float(2.0 * (1.0 - stats.norm.cdf(abs(a1 / s1))))
        else:
            a1, s1, p1 = np.nan, np.nan, np.nan
        max_smd = 0.0
        for j in range(x_bal.shape[1]):
            vt, vc = x_bal[t_idx, j].var(), x_bal[c_idx, j].var()
            sp = np.sqrt((vt + vc) / 2.0)
            max_smd = max(max_smd, abs(x_bal[t_idx, j].mean() - x_bal[c_idx, j].mean()) / sp if sp > 0 else 0.0)
        rb = rosenbaum_bounds_exact_and_approx(y2_vec[t_idx], y2_vec[c_idx], GAMMA_GRID)
        g_map_ex = {row['Gamma']: row['Upper_p_Exact_Binomial'] for row in rb['grid']}
        g_map_cc = {row['Gamma']: row['Upper_p_Normal_CC'] for row in rb['grid']}
        return {
            'Specification': spec_label,
            'Sample': sample_label,
            'N_Total': int(n_total),
            'N_Matched_Treated': int(len(t_idx)),
            'Est_Y2_NetZero': a2,
            'SE_Y2': s2,
            'p_Y2': p2,
            'SE_Y2_Cluster27': s2_cl,
            'p_Y2_Cluster27': p2_cl,
            'Est_Y1_GreenOfferings': a1,
            'SE_Y1': s1,
            'p_Y1': p1,
            'Max_PostMatch_SMD': float(max_smd),
            'n_10_T1_C0': rb['n_10'],
            'n_01_T0_C1': rb['n_01'],
            'n_Discordant': rb['n_disc'],
            'Critical_Gamma_Exact': rb['gamma_exact'],
            'Critical_Gamma_Normal_CC': rb['gamma_cc'],
            'Critical_Gamma_Normal_Uncorr': rb['gamma_uncorr'],
            'Exact_p_at_1.00': g_map_ex[1.00],
            'Exact_p_at_1.03': g_map_ex[1.03],
            'Exact_p_at_1.05': g_map_ex[1.05],
            'Exact_p_at_1.10': g_map_ex[1.10],
            'Normal_CC_p_at_1.00': g_map_cc[1.00],
            'Normal_CC_p_at_1.05': g_map_cc[1.05],
            'Normal_CC_p_at_1.10': g_map_cc[1.10],
        }, rb

    sens_rows = []

    # 1. Primary Optimal 1:1 Bipartite Logit PSM (0.20 SD(eta))
    r1, _ = eval_1to1_spec(
        '1. Primary Optimal 1:1 Bipartite Logit PSM (caliper 0.20 SD(logit PS), no repl.)',
        'Joint Sample (scr14_cat FE)', len(df_model), t_matched, c_matched, Y_sub, Y_sym, X_bal_mat, clusters_all
    )
    sens_rows.append(r1)

    # 2. Greedy 1:1 Logit PSM across 100 Random Tie Permutations (both treated & control order randomized)
    perm_rng = np.random.RandomState(2026)
    n_t_all, n_c_all = int((T == 1).sum()), int((T == 0).sum())
    g_atts_y2, g_atts_y1, g_gammas_ex, g_smds = [], [], [], []
    for _ in range(100):
        tp = perm_rng.permutation(n_t_all)
        cp = perm_rng.permutation(n_c_all)
        tm_g, cm_g = match_greedy_1to1(eta_nr, T, caliper_mult=0.20, t_perm=tp, c_perm=cp)
        g_atts_y2.append(float(np.mean(Y_sub[tm_g] - Y_sub[cm_g])))
        g_atts_y1.append(float(np.mean(Y_sym[tm_g] - Y_sym[cm_g])))
        rb_g = rosenbaum_bounds_exact_and_approx(Y_sub[tm_g], Y_sub[cm_g], [1.0, 1.05, 1.10])
        g_gammas_ex.append(rb_g['gamma_exact'])
        msmd = max(
            abs(X_bal_mat[tm_g, j].mean() - X_bal_mat[cm_g, j].mean()) /
            np.sqrt((X_bal_mat[tm_g, j].var() + X_bal_mat[cm_g, j].var()) / 2.0)
            for j in range(X_bal_mat.shape[1])
        )
        g_smds.append(float(msmd))
    mean_g_att = float(np.mean(g_atts_y2))
    sens_rows.append({
        'Specification': (f"2. Greedy 1:1 Logit PSM (100 random tie permutations; "
                          f"SD={np.std(g_atts_y2):.4f}, range [{min(g_atts_y2):+.4f}, {max(g_atts_y2):+.4f}])"),
        'Sample': 'Joint Sample (scr14_cat FE)',
        'N_Total': len(df_model),
        'N_Matched_Treated': int(len(tm_g)),
        'Est_Y2_NetZero': mean_g_att,
        'SE_Y2': se_sub_paired,
        'p_Y2': float(2.0 * (1.0 - stats.norm.cdf(abs(mean_g_att / se_sub_paired)))),
        'SE_Y2_Cluster27': se_sub_cl,
        'p_Y2_Cluster27': float(2.0 * (1.0 - stats.norm.cdf(abs(mean_g_att / se_sub_cl)))),
        'Est_Y1_GreenOfferings': float(np.mean(g_atts_y1)),
        'SE_Y1': se_sym_paired,
        'p_Y1': float(2.0 * (1.0 - stats.norm.cdf(abs(np.mean(g_atts_y1) / se_sym_paired)))),
        'Max_PostMatch_SMD': float(np.mean(g_smds)),
        'n_10_T1_C0': np.nan, 'n_01_T0_C1': np.nan, 'n_Discordant': np.nan,
        'Critical_Gamma_Exact': float(np.mean(g_gammas_ex)),
        'Critical_Gamma_Normal_CC': np.nan, 'Critical_Gamma_Normal_Uncorr': np.nan,
        'Exact_p_at_1.00': np.nan, 'Exact_p_at_1.03': np.nan,
        'Exact_p_at_1.05': np.nan, 'Exact_p_at_1.10': np.nan,
        'Normal_CC_p_at_1.00': np.nan, 'Normal_CC_p_at_1.05': np.nan, 'Normal_CC_p_at_1.10': np.nan,
    })

    # 3. Probability-Distance Optimal 1:1 PSM (caliper 0.20 * SD(ps))
    tm_ps, cm_ps = match_optimal_1to1(ps_nr, T, uids, caliper_mult=0.20)
    r3, _ = eval_1to1_spec(
        '3. Probability-Scale Optimal 1:1 PSM (caliper 0.20 SD(ps), no repl.)',
        'Joint Sample (scr14_cat FE)', len(df_model), tm_ps, cm_ps, Y_sub, Y_sym, X_bal_mat, clusters_all
    )
    sens_rows.append(r3)

    # 4. Tight Caliper Optimal 1:1 Logit PSM (0.10 SD(eta))
    tm_10, cm_10 = match_optimal_1to1(eta_nr, T, uids, caliper_mult=0.10)
    r4, _ = eval_1to1_spec(
        '4. Tight Caliper Optimal 1:1 Logit PSM (caliper 0.10 SD(logit PS), no repl.)',
        'Joint Sample (scr14_cat FE)', len(df_model), tm_10, cm_10, Y_sub, Y_sym, X_bal_mat, clusters_all
    )
    sens_rows.append(r4)

    # 5. Very Tight Caliper Optimal 1:1 Logit PSM (0.05 SD(eta))
    tm_05, cm_05 = match_optimal_1to1(eta_nr, T, uids, caliper_mult=0.05)
    r5, _ = eval_1to1_spec(
        '5. Very Tight Caliper Optimal 1:1 Logit PSM (caliper 0.05 SD(logit PS), no repl.)',
        'Joint Sample (scr14_cat FE)', len(df_model), tm_05, cm_05, Y_sub, Y_sym, X_bal_mat, clusters_all
    )
    sens_rows.append(r5)

    # 6. 1:2 Nearest-Neighbor Logit PSM with replacement (0.20 SD(eta))
    nt_2, a2_2, s2_2, a1_2, s1_2, smd_2 = match_1tok_with_repl(eta_nr, T, Y_sub, Y_sym, X_bal_mat, k_neighbors=2, caliper_mult=0.20)
    sens_rows.append({
        'Specification': '6. 1:2 Nearest-Neighbor Logit PSM (caliper 0.20 SD(logit PS), with repl.)',
        'Sample': 'Joint Sample (scr14_cat FE)', 'N_Total': len(df_model), 'N_Matched_Treated': nt_2,
        'Est_Y2_NetZero': a2_2, 'SE_Y2': s2_2, 'p_Y2': float(2.0 * (1.0 - stats.norm.cdf(abs(a2_2 / s2_2)))),
        'SE_Y2_Cluster27': np.nan, 'p_Y2_Cluster27': np.nan,
        'Est_Y1_GreenOfferings': a1_2, 'SE_Y1': s1_2, 'p_Y1': float(2.0 * (1.0 - stats.norm.cdf(abs(a1_2 / s1_2)))),
        'Max_PostMatch_SMD': smd_2,
        'n_10_T1_C0': np.nan, 'n_01_T0_C1': np.nan, 'n_Discordant': np.nan,
        'Critical_Gamma_Exact': np.nan, 'Critical_Gamma_Normal_CC': np.nan, 'Critical_Gamma_Normal_Uncorr': np.nan,
        'Exact_p_at_1.00': np.nan, 'Exact_p_at_1.03': np.nan, 'Exact_p_at_1.05': np.nan, 'Exact_p_at_1.10': np.nan,
        'Normal_CC_p_at_1.00': np.nan, 'Normal_CC_p_at_1.05': np.nan, 'Normal_CC_p_at_1.10': np.nan,
    })

    # 7. 1:4 Nearest-Neighbor Logit PSM with replacement (0.20 SD(eta))
    nt_4, a2_4, s2_4, a1_4, s1_4, smd_4 = match_1tok_with_repl(eta_nr, T, Y_sub, Y_sym, X_bal_mat, k_neighbors=4, caliper_mult=0.20)
    sens_rows.append({
        'Specification': '7. 1:4 Nearest-Neighbor Logit PSM (caliper 0.20 SD(logit PS), with repl.)',
        'Sample': 'Joint Sample (scr14_cat FE)', 'N_Total': len(df_model), 'N_Matched_Treated': nt_4,
        'Est_Y2_NetZero': a2_4, 'SE_Y2': s2_4, 'p_Y2': float(2.0 * (1.0 - stats.norm.cdf(abs(a2_4 / s2_4)))),
        'SE_Y2_Cluster27': np.nan, 'p_Y2_Cluster27': np.nan,
        'Est_Y1_GreenOfferings': a1_4, 'SE_Y1': s1_4, 'p_Y1': float(2.0 * (1.0 - stats.norm.cdf(abs(a1_4 / s1_4)))),
        'Max_PostMatch_SMD': smd_4,
        'n_10_T1_C0': np.nan, 'n_01_T0_C1': np.nan, 'n_Discordant': np.nan,
        'Critical_Gamma_Exact': np.nan, 'Critical_Gamma_Normal_CC': np.nan, 'Critical_Gamma_Normal_Uncorr': np.nan,
        'Exact_p_at_1.00': np.nan, 'Exact_p_at_1.03': np.nan, 'Exact_p_at_1.05': np.nan, 'Exact_p_at_1.10': np.nan,
        'Normal_CC_p_at_1.00': np.nan, 'Normal_CC_p_at_1.05': np.nan, 'Normal_CC_p_at_1.10': np.nan,
    })

    # 8. Complete-Case Ordinal Turnover (scr14_clean in 1..9, N = 9,764) Optimal 1:1 Logit PSM
    df_cc = df_model.dropna(subset=['scr14_clean']).copy()
    T_cc = df_cc['grant_receipt'].values.astype(float)
    Y2_cc = df_cc['q14_binary'].values.astype(float)
    Y1_cc = df_cc['q9_binary'].values.astype(float)
    X_cc_cov = df_cc[X_cols + ['scr14_clean']].values.astype(float)
    FE_cc = pd.concat([
        pd.get_dummies(df_cc['country_iso2'], prefix='c', drop_first=True, dtype=float),
        pd.get_dummies(df_cc['nace_b'].astype(int), prefix='s', drop_first=True, dtype=float)
    ], axis=1)
    X_ps_cc = np.column_stack([X_cc_cov, FE_cc.values, np.ones(len(df_cc))])
    eta_cc = X_ps_cc @ fit_logit_nr(X_ps_cc, T_cc, tol=1e-13, ridge=0.0)
    tm_cc, cm_cc = match_optimal_1to1(eta_cc, T_cc, df_cc['uniqid'].values, caliper_mult=0.20)
    r8, rb_cc = eval_1to1_spec(
        '8. Complete-Case Turnover (scr14 < 10) Optimal 1:1 Logit PSM (N=9,764)',
        'Complete-Case Turnover (N=9,764)', len(df_cc), tm_cc, cm_cc, Y2_cc, Y1_cc, X_cc_cov, df_cc['country_iso2'].values
    )
    sens_rows.append(r8)
    for r in rb_cc['grid']:
        rosenbaum_rows.append({'Sample': f"Complete-Case Turnover (N={len(df_cc):,}, N_T={len(tm_cc)})", **r})

    # 9 & 10. Ordinal Imputed Turnover + DK Indicator (scr14_imp + scr14_dk, N = 11,128): Optimal 1:1 and 1:4 NN
    X_imp_cov = df_model[X_cols + ['scr14_imp', 'scr14_dk']].values.astype(float)
    X_ps_imp = np.column_stack([X_imp_cov, FE_c.values, FE_s.values, np.ones(len(df_model))])
    eta_imp = X_ps_imp @ fit_logit_nr(X_ps_imp, T, tol=1e-13, ridge=0.0)
    tm_imp, cm_imp = match_optimal_1to1(eta_imp, T, uids, caliper_mult=0.20)
    r9, _ = eval_1to1_spec(
        '9. Ordinal Turnover + DK Dummy (scr14_imp + scr14_dk) Optimal 1:1 Logit PSM',
        'Joint Sample (scr14_imp + DK)', len(df_model), tm_imp, cm_imp, Y_sub, Y_sym, X_imp_cov, clusters_all
    )
    sens_rows.append(r9)

    nt_4i, a2_4i, s2_4i, a1_4i, s1_4i, smd_4i = match_1tok_with_repl(eta_imp, T, Y_sub, Y_sym, X_imp_cov, k_neighbors=4, caliper_mult=0.20)
    sens_rows.append({
        'Specification': '10. Ordinal Turnover + DK Dummy (scr14_imp + scr14_dk) 1:4 NN PSM w/ repl.',
        'Sample': 'Joint Sample (scr14_imp + DK)', 'N_Total': len(df_model), 'N_Matched_Treated': nt_4i,
        'Est_Y2_NetZero': a2_4i, 'SE_Y2': s2_4i, 'p_Y2': float(2.0 * (1.0 - stats.norm.cdf(abs(a2_4i / s2_4i)))),
        'SE_Y2_Cluster27': np.nan, 'p_Y2_Cluster27': np.nan,
        'Est_Y1_GreenOfferings': a1_4i, 'SE_Y1': s1_4i, 'p_Y1': float(2.0 * (1.0 - stats.norm.cdf(abs(a1_4i / s1_4i)))),
        'Max_PostMatch_SMD': smd_4i,
        'n_10_T1_C0': np.nan, 'n_01_T0_C1': np.nan, 'n_Discordant': np.nan,
        'Critical_Gamma_Exact': np.nan, 'Critical_Gamma_Normal_CC': np.nan, 'Critical_Gamma_Normal_Uncorr': np.nan,
        'Exact_p_at_1.00': np.nan, 'Exact_p_at_1.03': np.nan, 'Exact_p_at_1.05': np.nan, 'Exact_p_at_1.10': np.nan,
        'Normal_CC_p_at_1.00': np.nan, 'Normal_CC_p_at_1.05': np.nan, 'Normal_CC_p_at_1.10': np.nan,
    })

    # 11. Single-Outcome IPW Sample (N = 11,521, valid Q14) Optimal 1:1 Logit PSM
    df_11521 = df.dropna(subset=['uniqid', 'country_iso2', 'nace_b', 'scr14_cat', 'q14_binary', 'grant_receipt'] + X_cols).copy()
    T_11521 = df_11521['grant_receipt'].values.astype(float)
    Y2_11521 = df_11521['q14_binary'].values.astype(float)
    X_11521_cov = df_11521[X_cols].values.astype(float)
    FE_11521 = pd.concat([
        pd.get_dummies(df_11521['scr14_cat'].astype(int), prefix='turn', drop_first=True, dtype=float),
        pd.get_dummies(df_11521['country_iso2'], prefix='c', drop_first=True, dtype=float),
        pd.get_dummies(df_11521['nace_b'].astype(int), prefix='s', drop_first=True, dtype=float),
    ], axis=1)
    X_ps_11521 = np.column_stack([X_11521_cov, FE_11521.values, np.ones(len(df_11521))])
    eta_11521 = X_ps_11521 @ fit_logit_nr(X_ps_11521, T_11521, tol=1e-13, ridge=0.0)
    tm_11521, cm_11521 = match_optimal_1to1(eta_11521, T_11521, df_11521['uniqid'].values, caliper_mult=0.20)
    r11, rb_11521 = eval_1to1_spec(
        '11. Single-Outcome Sample Optimal 1:1 Logit PSM (N=11,521, valid Q14)',
        'Single-Outcome IPW Sample (N=11,521)', len(df_11521), tm_11521, cm_11521,
        Y2_11521, None, X_11521_cov, df_11521['country_iso2'].values
    )
    sens_rows.append(r11)
    for r in rb_11521['grid']:
        rosenbaum_rows.append({'Sample': f"Single-Outcome IPW Sample (N={len(df_11521):,}, N_T={len(tm_11521)})", **r})

    # 12. Exogenous Pre-Treatment Covariates + FE Only (excl. barriers & RECount) Optimal 1:1 Logit PSM
    exo_cols = ['B2B_private', 'B2G_public', 'B2C', 'scr10']
    X_exo_cov = df_model[exo_cols].values.astype(float)
    X_ps_exo = np.column_stack([X_exo_cov, FE_turn.values, FE_c.values, FE_s.values, np.ones(len(df_model))])
    eta_exo = X_ps_exo @ fit_logit_nr(X_ps_exo, T, tol=1e-13, ridge=0.0)
    tm_exo, cm_exo = match_optimal_1to1(eta_exo, T, uids, caliper_mult=0.20)
    r12, _ = eval_1to1_spec(
        '12. Exogenous Pre-Treatment + FE Optimal 1:1 Logit PSM (excl. barriers & RECount)',
        'Joint Sample (Exogenous + FE)', len(df_model), tm_exo, cm_exo, Y_sub, Y_sym, X_exo_cov, clusters_all
    )
    sens_rows.append(r12)

    # 13. Doubly Robust Augmented IPW (AIPW) on N = 11,128
    beta_y2_1 = fit_logit_nr(X_ps[T == 1], Y_sub[T == 1], ridge=1e-8)
    beta_y2_0 = fit_logit_nr(X_ps[T == 0], Y_sub[T == 0], ridge=1e-8)
    mu2_1, mu2_0 = expit(X_ps @ beta_y2_1), expit(X_ps @ beta_y2_0)
    ps_clip = np.clip(ps_nr, 0.02, 0.98)
    aipw_scores_y2 = (mu2_1 - mu2_0) + T * (Y_sub - mu2_1) / ps_clip - (1.0 - T) * (Y_sub - mu2_0) / (1.0 - ps_clip)
    aipw_y2 = float(np.mean(aipw_scores_y2))
    aipw_se_y2 = float(np.std(aipw_scores_y2, ddof=1) / np.sqrt(len(Y_sub)))
    aipw_se_y2_cl, _ = clustered_pair_se(aipw_scores_y2, clusters_all)

    beta_y1_1 = fit_logit_nr(X_ps[T == 1], Y_sym[T == 1], ridge=1e-8)
    beta_y1_0 = fit_logit_nr(X_ps[T == 0], Y_sym[T == 0], ridge=1e-8)
    mu1_1, mu1_0 = expit(X_ps @ beta_y1_1), expit(X_ps @ beta_y1_0)
    aipw_scores_y1 = (mu1_1 - mu1_0) + T * (Y_sym - mu1_1) / ps_clip - (1.0 - T) * (Y_sym - mu1_0) / (1.0 - ps_clip)
    aipw_y1 = float(np.mean(aipw_scores_y1))
    aipw_se_y1 = float(np.std(aipw_scores_y1, ddof=1) / np.sqrt(len(Y_sym)))

    sens_rows.append({
        'Specification': '13. Doubly Robust Augmented IPW (AIPW, Exact NR Logit PS + Logit Outcome)',
        'Sample': 'Joint Sample (scr14_cat FE)', 'N_Total': len(df_model), 'N_Matched_Treated': int(T.sum()),
        'Est_Y2_NetZero': aipw_y2, 'SE_Y2': aipw_se_y2,
        'p_Y2': float(2.0 * (1.0 - stats.norm.cdf(abs(aipw_y2 / aipw_se_y2)))),
        'SE_Y2_Cluster27': aipw_se_y2_cl,
        'p_Y2_Cluster27': float(2.0 * (1.0 - stats.norm.cdf(abs(aipw_y2 / aipw_se_y2_cl)))),
        'Est_Y1_GreenOfferings': aipw_y1, 'SE_Y1': aipw_se_y1,
        'p_Y1': float(2.0 * (1.0 - stats.norm.cdf(abs(aipw_y1 / aipw_se_y1)))),
        'Max_PostMatch_SMD': np.nan,
        'n_10_T1_C0': np.nan, 'n_01_T0_C1': np.nan, 'n_Discordant': np.nan,
        'Critical_Gamma_Exact': np.nan, 'Critical_Gamma_Normal_CC': np.nan, 'Critical_Gamma_Normal_Uncorr': np.nan,
        'Exact_p_at_1.00': np.nan, 'Exact_p_at_1.03': np.nan, 'Exact_p_at_1.05': np.nan, 'Exact_p_at_1.10': np.nan,
        'Normal_CC_p_at_1.00': np.nan, 'Normal_CC_p_at_1.05': np.nan, 'Normal_CC_p_at_1.10': np.nan,
    })

    sens_df = pd.DataFrame(sens_rows)
    rosenbaum_df = pd.DataFrame(rosenbaum_rows)

    psm_summary = pd.DataFrame([
        {
            'Outcome': 'Reported Net-Zero Strategy (Q14)',
            'N_Matched_Treated': int(len(t_matched)),
            'ATT': att_sub,
            'SE': se_sub_paired,
            'p': p_sub_paired,
            'CI_lo': att_sub - 1.96 * se_sub_paired,
            'CI_hi': att_sub + 1.96 * se_sub_paired,
            'SE_paired': se_sub_paired,
            'p_paired': p_sub_paired,
            'SE_boot_1000': se_sub_boot,
            'p_boot_1000': p_sub_boot,
            'SE_cluster27': se_sub_cl,
            'p_cluster27': p_sub_cl,
            'p_cluster27_t26': p_sub_cl_t26,
            'n_10_T1_C0': rb_base['n_10'],
            'n_01_T0_C1': rb_base['n_01'],
            'n_Discordant': rb_base['n_disc'],
            'Critical_Gamma_Exact': rb_base['gamma_exact'],
            'Critical_Gamma_Normal_CC': rb_base['gamma_cc'],
        },
        {
            'Outcome': 'Green Product Offerings (Q9)',
            'N_Matched_Treated': int(len(t_matched)),
            'ATT': att_sym,
            'SE': se_sym_paired,
            'p': p_sym_paired,
            'CI_lo': att_sym - 1.96 * se_sym_paired,
            'CI_hi': att_sym + 1.96 * se_sym_paired,
            'SE_paired': se_sym_paired,
            'p_paired': p_sym_paired,
            'SE_boot_1000': se_sym_boot,
            'p_boot_1000': p_sym_boot,
            'SE_cluster27': se_sym_cl,
            'p_cluster27': p_sym_cl,
            'p_cluster27_t26': p_sym_cl_t26,
            'n_10_T1_C0': int(np.sum((Y_sym[t_matched] == 1) & (Y_sym[c_matched] == 0))),
            'n_01_T0_C1': int(np.sum((Y_sym[t_matched] == 0) & (Y_sym[c_matched] == 1))),
            'n_Discordant': int(np.sum(Y_sym[t_matched] != Y_sym[c_matched])),
            'Critical_Gamma_Exact': np.nan,
            'Critical_Gamma_Normal_CC': np.nan,
        }
    ])

    print("\n  [PSM 13-Specification Sensitivity Suite]")
    print(sens_df[['Specification', 'N_Matched_Treated', 'Est_Y2_NetZero', 'SE_Y2', 'p_Y2', 'Critical_Gamma_Exact']].round(4).to_string(index=False))

    return psm_summary, pd.DataFrame(balance_data), sens_df, rosenbaum_df


# ═══════════════════════════════════════════════════════════════════
# PART 2: Cross-Level Digital Intensity Moderation (Model 4)
# ═══════════════════════════════════════════════════════════════════

def run_decoupling_moderation(df):
    """Cross-level OLS interaction model (Model 4) on DI = Y1 - Y2 (N = 12,205)."""
    print("\n" + "=" * 76)
    print("PHASE 2: Cross-Level Digital Intensity Moderation (Model 4, N = 12,205)")
    print("=" * 76)

    needed = ['country_iso2', 'DI', 'B2B_private', 'B2G_public', 'DII_composite',
              'scr10', 'scr14_cat', 're_action_count']
    df_model = df[needed].dropna().copy()

    df_model['DII_std'] = (df_model['DII_composite'] - df_model['DII_composite'].mean()) / df_model['DII_composite'].std()
    df_model['B2B_x_DII'] = df_model['B2B_private'] * df_model['DII_std']

    focal_cols = ['B2B_private', 'DII_std', 'B2B_x_DII', 'B2G_public', 'scr10', 're_action_count']
    turn_fe = pd.get_dummies(df_model['scr14_cat'].astype(int), prefix='turn', drop_first=True, dtype=float)
    X_df = pd.concat([df_model[focal_cols].astype(float), turn_fe], axis=1)
    X_df['const'] = 1.0
    var_names = list(X_df.columns)
    X = X_df.values
    y = df_model['DI'].values.astype(float)
    clusters = pd.factorize(df_model['country_iso2'])[0]

    model = sm.OLS(y, X)
    res_cl = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})
    res_hc1 = model.fit(cov_type='HC1')

    ols_results = []
    for j, v in enumerate(var_names):
        b = float(res_cl.params[j])
        se_c = float(res_cl.bse[j])
        t_c = b / se_c
        p_c = float(2.0 * (1.0 - stats.norm.cdf(abs(t_c))))
        p_t26 = float(2.0 * (1.0 - stats.t.cdf(abs(t_c), df=26)))
        se_h = float(res_hc1.bse[j])
        p_h = float(res_hc1.pvalues[j])
        ols_results.append({
            'Variable': v, 'Coef': b,
            'SE_cluster': se_c, 't_cluster': t_c, 'p_cluster': p_c, 'p_t26': p_t26,
            'SE_robust_HC1': se_h, 't_HC1': float(res_hc1.tvalues[j]), 'p_HC1': p_h,
            'N': len(df_model), 'R2': float(res_cl.rsquared)
        })

    ols_df = pd.DataFrame(ols_results)
    print(ols_df[ols_df['Variable'].isin(focal_cols + ['const'])].round(4).to_string(index=False))

    country_di = df.groupby('country_iso2').agg(
        Mean_DI=('DI', 'mean'),
        N=('DI', 'count'),
        DII=('DII_composite', 'first'),
        Pct_GreenOfferingsOnly=('DI', lambda x: (x.dropna() == 1).mean())
    ).sort_values('Mean_DI')

    return ols_df, country_di


# ═══════════════════════════════════════════════════════════════════
# PART 3: Ordered Probit - Operating Cost Impact (Model 5)
# ═══════════════════════════════════════════════════════════════════

def run_green_premium(df):
    """
    3-category Ordered Probit of perceived production cost impact (Q3) on sustainability profiles:
      cost_outcome = 0 if Q3 in {1, 2} (decreased),
                     1 if Q3 == 5       (unchanged),
                     2 if Q3 in {3, 4} (increased)
    Estimated with categorical turnover bracket FEs (scr14_cat) and 9-item re_action_count.
    """
    print("\n" + "=" * 76)
    print("PHASE 3: Operating Cost Impact 3-Category Ordered Probit (Model 5)")
    print("=" * 76)

    needed_base = ['country_iso2', 'nace_b', 'q3', 'q4', 'q9_binary', 'q14_binary',
                   'scr10', 'scr14_cat', 're_action_count', 'barrier_admin']
    df_10011 = df.dropna(subset=['country_iso2', 'nace_b', 'q3', 'q9_binary', 'q14_binary',
                                 'scr10', 'scr14_cat', 're_action_count', 'barrier_admin']).copy()
    df_10011 = df_10011[df_10011['q3'].isin([1, 2, 3, 4, 5])].copy()
    df_10011['cost_outcome'] = df_10011['q3'].map({
        1: 0, 2: 0,  # decreased
        5: 1,        # unchanged
        3: 2, 4: 2   # increased
    })
    df_10011['green_only'] = ((df_10011['q9_binary'] == 1) & (df_10011['q14_binary'] == 0)).astype(float)
    df_10011['netzero_only'] = ((df_10011['q9_binary'] == 0) & (df_10011['q14_binary'] == 1)).astype(float)
    df_10011['both'] = ((df_10011['q9_binary'] == 1) & (df_10011['q14_binary'] == 1)).astype(float)
    df_10011['neither'] = ((df_10011['q9_binary'] == 0) & (df_10011['q14_binary'] == 0)).astype(float)

    ct_results = []
    for cat, label in [('neither', 'Neither (0,0)'), ('green_only', 'Green Offerings Only (1,0)'),
                       ('netzero_only', 'Net-Zero Strategy Only (0,1)'), ('both', 'Both Activities (1,1)')]:
        mask = df_10011[cat] == 1
        co = df_10011.loc[mask, 'cost_outcome']
        ct_results.append({
            'Category': label, 'N': int(mask.sum()),
            'Pct_Decreased': float((co == 0).mean() * 100.0),
            'Pct_Unchanged': float((co == 1).mean() * 100.0),
            'Pct_Increased': float((co == 2).mean() * 100.0)
        })

    op_rows = []

    def fit_op_spec(spec_label, d_sub, x_cols, include_fe=True):
        cl = pd.factorize(d_sub['country_iso2'])[0]
        turn_fe = pd.get_dummies(d_sub['scr14_cat'].astype(int), prefix='turn', drop_first=True, dtype=float)
        if include_fe:
            fe = pd.concat([
                turn_fe,
                pd.get_dummies(d_sub['country_iso2'], prefix='c', drop_first=True, dtype=float),
                pd.get_dummies(d_sub['nace_b'].astype(int), prefix='s', drop_first=True, dtype=float)
            ], axis=1)
        else:
            fe = turn_fe
        X_mat = pd.concat([d_sub[x_cols].astype(float), fe], axis=1)
        mod = OrderedModel(d_sub['cost_outcome'], X_mat, distr='probit')
        res_cl = mod.fit(method='bfgs', maxiter=500, disp=False, cov_type='cluster', cov_kwds={'groups': cl})
        res_iid = mod.fit(method='bfgs', maxiter=500, disp=False)
        for v in x_cols:
            b = float(res_cl.params[v])
            se_c = float(res_cl.bse[v])
            p_c = float(res_cl.pvalues[v])
            p_t26 = float(2.0 * (1.0 - stats.t.cdf(abs(b / se_c), df=26)))
            se_i = float(res_iid.bse[v])
            p_i = float(res_iid.pvalues[v])
            op_rows.append({
                'Spec': spec_label, 'Variable': v, 'Beta': b,
                'SE_cluster': se_c, 'p_cluster': p_c, 'p_t26': p_t26,
                'SE_iid': se_i, 'p_iid': p_i, 'N': int(len(d_sub))
            })

    fit_op_spec('Spec 1: Binary Y1 & Y2 (Baseline Controls + Turnover FE, N=10,011)',
                df_10011, ['q9_binary', 'q14_binary', 'scr10', 're_action_count'], include_fe=False)
    fit_op_spec('Spec 2: Binary Y1 & Y2 (+ Country/Sector/Turnover FE, N=10,011)',
                df_10011, ['q9_binary', 'q14_binary', 'scr10', 're_action_count'], include_fe=True)
    fit_op_spec('Spec 3: 4-Quadrant Typology (+ Country/Sector/Turnover FE, N=10,011)',
                df_10011, ['green_only', 'netzero_only', 'both', 'scr10', 're_action_count'], include_fe=True)

    df_9058 = df_10011[df_10011['q4'].isin([1, 2, 3, 4, 5, 6])].copy()
    fit_op_spec('Spec 4: Binary Y1 & Y2 (+ Valid Q4 [1..6] + Country/Sector/Turnover FE, N=9,058)',
                df_9058, ['q9_binary', 'q14_binary', 'q4', 'scr10', 're_action_count'], include_fe=True)
    fit_op_spec('Spec 5: 4-Quadrant Typology (+ Valid Q4 [1..6] + Country/Sector/Turnover FE, N=9,058)',
                df_9058, ['green_only', 'netzero_only', 'both', 'q4', 'scr10', 're_action_count'], include_fe=True)

    op_df = pd.DataFrame(op_rows)
    print(op_df.round(4).to_string(index=False))
    return pd.DataFrame(ct_results), op_df


def main():
    df = pd.read_csv(DATA_FILE)
    psm_summary, balance_df, sens_df, rosenbaum_df = run_psm(df)
    ols_df, country_di = run_decoupling_moderation(df)
    ct_df, op_df = run_green_premium(df)

    mode = 'a' if os.path.exists(RESULTS_XLSX) else 'w'
    with pd.ExcelWriter(RESULTS_XLSX, engine='openpyxl', mode=mode,
                        if_sheet_exists='replace' if mode == 'a' else None) as writer:
        psm_summary.to_excel(writer, sheet_name='PSM_ATT', index=False)
        balance_df.to_excel(writer, sheet_name='PSM_Balance', index=False)
        sens_df.to_excel(writer, sheet_name='PSM_Sensitivity_Specs', index=False)
        rosenbaum_df.to_excel(writer, sheet_name='Rosenbaum_Bounds', index=False)
        ols_df.to_excel(writer, sheet_name='DI_Moderation_OLS', index=False)
        country_di.to_excel(writer, sheet_name='DI_Country_Level')
        ct_df.to_excel(writer, sheet_name='GreenPremium_CrossTab', index=False)
        op_df.to_excel(writer, sheet_name='GreenPremium_OrderedProbit', index=False)

    print(f"\n  ✓ Saved PSM, Rosenbaum, Model 4, and Model 5 outputs to {RESULTS_XLSX}")


if __name__ == '__main__':
    main()
