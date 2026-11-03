#!/usr/bin/env python3
"""
05_robustness.py
================
Comprehensive Robustness, 4-Quadrant Multinomial Logit, Probability-Scale Moderation (H3c),
Construct Validity, and Round-2 Empirical Extensions (Appendix M), consolidating all model
outputs into Data/Results_Consolidated.xlsx.

Specifications estimated:
  1. 4-Quadrant Multinomial Logit (N = 11,128; categorical scr14_cat FEs, 26 country FEs,
     3 NACE macro-sector FEs, 9-item re_action_count; country-clustered SEs + discrete AMEs)
  2. Table 6 Joint FIML Bivariate Probit Robustness Matrix (all 12 specifications + sector-stratified
     joint BiProbits + separate probit reference check; strict interior convergence ||grad||_inf < 1e-7
     and PD Hessian verified on every joint specification)
  3. H3c Probability-Scale Moderation (including advice_public control; cluster-robust delta method + t_26)
  4. Construct_Validity_Table1 across configurations (both N = 12,205 and N = 11,128)
  5. Extension 1 (Appendix M.1): Tightened Construct-Intensity Joint FIML BiProbit
     (Y1: Q9=1 & Q10>10% green turnover vs. Y2: Q14=1 & [Q15.2=1 | Q15.3=1] low-carbon tech;
     estimated on both N = 10,687 keeping Q14=4 as 0 and N = 10,203 excluding Q14=4)
  6. Extension 2 (Appendix M.2): Unrouted Q8 Desired-Support Mechanism Check (N = 13,322 & N = 11,128)
  7. Extension 3 (Appendix M.3): Out-of-Sample Temporal Replication on 2024 Flash Eurobarometer 549
     (ZA8869_v1-0-0.sav, all 27 EU member states with GR->EL, N = 11,246 Joint FIML BiProbit;
     gracefully skipped if ZA8869 raw file is absent)
"""

import importlib.util
import os
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices, dmatrices
from scipy import stats
from scipy.special import expit, softmax

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "Data")
if not os.path.exists(DATA_DIR) and os.path.exists(os.path.join(os.path.dirname(ROOT), "Data")):
    DATA_DIR = os.path.join(os.path.dirname(ROOT), "Data")

DATA_CSV = os.path.join(DATA_DIR, "Processed", "SME_Analytical_Final.csv")
OUT_ROB_XLSX = os.path.join(DATA_DIR, "Processed", "Robustness_FE.xlsx")
BP_XLSX = os.path.join(DATA_DIR, "Processed", "BiProbit_Joint_FE.xlsx")
INP_XLSX = os.path.join(DATA_DIR, "Input_Data_Consolidated.xlsx")
RES_CONSOL = os.path.join(DATA_DIR, "Results_Consolidated.xlsx")

# Load Joint FIML Bivariate Probit engine from 02_bivariate_probit.py
_spec02 = importlib.util.spec_from_file_location("biprobit_mod", os.path.join(SCRIPT_DIR, "02_bivariate_probit.py"))
bp_mod = importlib.util.module_from_spec(_spec02)
_spec02.loader.exec_module(bp_mod)

IV = bp_mod.IV
FOCAL_6 = bp_mod.FOCAL_6
CTRL = bp_mod.CTRL  # ['scr10', 're_action_count']
EXPANDED_VARS = ['barrier_cost', 'barrier_label', 'own_finance', 'own_expertise', 'q4_clean', 'sells_products']

BASE_FORMULA = ' + '.join(IV) + ' + scr10 + re_action_count + C(scr14_cat) + C(country_iso2) + C(nace_b)'


def fit_single_probit_clustered(formula, data, cluster_col='country_iso2'):
    """Single-equation probit with country-clustered sandwich SEs (used for N=3,055 Q5.3=1 Y2-only check)."""
    y_mat, X_mat = dmatrices(formula, data, return_type='dataframe')
    y = y_mat.values.ravel().astype(float)
    X = X_mat.values.astype(float)
    col_names = list(X_mat.columns)
    N, k = X.shape
    clusters = pd.factorize(data.loc[y_mat.index, cluster_col])[0]
    G = len(np.unique(clusters))

    m = sm.Probit(y, X).fit(disp=0, maxiter=300, cov_type='cluster', cov_kwds={'groups': clusters})
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
        'nobs': N, 'n_clusters': G, 'col_names': col_names,
        'cluster_if': cluster_if, 'dof_scale': dof_scale
    }


def extract_biprobit_rows(fit_res, spec_label, y1_name='q9_binary', y2_name='q14_binary', extra_vars=None):
    """Extract per-equation coefficients, cross-equation Wald differences, and convergence diagnostics."""
    th = fit_res['theta']
    se = fit_res['se']
    names = fit_res['names']
    k = fit_res['k']
    df_c = fit_res['df_c']
    N = fit_res['diag']['N']
    tdf = fit_res['tdf'].set_index('var')
    am_df = fit_res['am_df'].set_index('var')

    vars_to_pull = [v for v in (IV + CTRL + (extra_vars or [])) if v in names]
    coef_rows = []
    for outcome_lbl, off in [(y1_name, 0), (y2_name, k)]:
        for v in vars_to_pull:
            j = names.index(v) + off
            z = float(th[j] / se[j])
            coef_rows.append(dict(
                spec=spec_label, outcome=outcome_lbl, var=v,
                beta=float(th[j]), se=float(se[j]), z=z,
                p=float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
                p_t26=float(2.0 * (1.0 - stats.t.cdf(abs(z), df=df_c))),
                N=int(N)
            ))

    diff_rows = []
    for v in [x for x in IV if x in tdf.index]:
        j1, j2 = names.index(v), names.index(v) + k
        b1, s1 = float(th[j1]), float(se[j1])
        b2, s2 = float(th[j2]), float(se[j2])
        tr = tdf.loc[v]
        ar = am_df.loc[v]
        diff_rows.append(dict(
            spec=spec_label, var=v,
            beta_y1=b1, se_y1=s1, p_y1=float(2.0 * (1.0 - stats.norm.cdf(abs(b1 / s1)))),
            p_t26_y1=float(2.0 * (1.0 - stats.t.cdf(abs(b1 / s1), df=df_c))),
            beta_y2=b2, se_y2=s2, p_y2=float(2.0 * (1.0 - stats.norm.cdf(abs(b2 / s2)))),
            p_t26_y2=float(2.0 * (1.0 - stats.t.cdf(abs(b2 / s2), df=df_c))),
            diff_b1_minus_b2=float(tr['diff']), se_diff=float(tr['se']), z_diff=float(tr['z']),
            p_diff=float(tr['p']), p_t26_diff=float(tr['p_t26']),
            p_holm_m6=float(tr['p_holm_m6']) if pd.notna(tr['p_holm_m6']) else np.nan,
            p_t26_holm_m6=float(tr['p_t26_holm_m6']) if pd.notna(tr['p_t26_holm_m6']) else np.nan,
            AME_y1=float(ar['AME_Q9']), se_AME_y1=float(ar['se_Q9']),
            AME_y2=float(ar['AME_Q14']), se_AME_y2=float(ar['se_Q14']),
            AME_diff=float(ar['AME_diff']), se_AME_diff=float(ar['se_AME_diff']),
            p_AME_diff=float(ar['p_AME_diff']), p_AME_diff_t26=float(ar['p_AME_diff_t26']),
            rho=float(fit_res['diag']['rho']), se_rho=float(fit_res['diag']['se_rho']),
            N=int(N)
        ))

    diag_row = dict(spec=spec_label, **fit_res['diag'])
    return coef_rows, diff_rows, diag_row


def build_construct_validity_profile(d_sample, sample_tag):
    """Build construct-validity profile across the 4 configurations for a given sample."""
    quad_labels = {
        0: '0. Neither (0,0)',
        1: '1. Green Offerings Only (1,0)',
        2: '2. Net-Zero Strategy Only (0,1)',
        3: '3. Dual Adopters / Both (1,1)'
    }
    d_loc = d_sample.copy()
    d_loc['quad_4'] = np.select(
        [(d_loc['q9_binary'] == 0) & (d_loc['q14_binary'] == 0),
         (d_loc['q9_binary'] == 1) & (d_loc['q14_binary'] == 0),
         (d_loc['q9_binary'] == 0) & (d_loc['q14_binary'] == 1),
         (d_loc['q9_binary'] == 1) & (d_loc['q14_binary'] == 1)],
        [0, 1, 2, 3]
    )
    out = []
    for q_val, q_name in list(quad_labels.items()) + [('All', 'All SMEs in Sample')]:
        sub_q = d_loc if q_val == 'All' else d_loc[d_loc['quad_4'] == q_val]
        n_q = len(sub_q)
        q4_valid = sub_q['q4_clean'].dropna()
        q15_valid = sub_q['q15_count'].dropna()
        q15_tech_valid = sub_q['q15_tech'].dropna()
        q10_valid = sub_q['q10_clean'].dropna()
        out.append({
            'Sample': sample_tag,
            'Quadrant_ID': q_val,
            'Configuration': q_name,
            'N': n_q,
            'Sample_Share_Pct': round(100.0 * n_q / len(d_loc), 2),
            'Mean_DI': round(float(sub_q['DI'].mean()), 4),
            'Sells_Products_SCR16_Pct': round(100.0 * float(sub_q['sells_products'].dropna().mean()), 2),
            'B2C_Pct': round(100.0 * float(sub_q['B2C'].mean()), 2),
            'B2G_Pct': round(100.0 * float(sub_q['B2G_public'].mean()), 2),
            'B2B_Pct': round(100.0 * float(sub_q['B2B_private'].mean()), 2),
            'Barrier_Admin_Pct': round(100.0 * float(sub_q['barrier_admin'].dropna().mean()), 2) if sub_q['barrier_admin'].notna().any() else np.nan,
            'Barrier_Expertise_Pct': round(100.0 * float(sub_q['barrier_expertise'].dropna().mean()), 2) if sub_q['barrier_expertise'].notna().any() else np.nan,
            'Public_Support_Q6_1_Pct': round(100.0 * float(sub_q['grant_receipt'].mean()), 2),
            'RE_Actions_9_Mean': round(float(sub_q['re_action_count'].mean()), 2),
            'Mean_Firm_Size_1_3': round(float(sub_q['scr10'].mean()), 2),
            'Mean_National_DII': round(float(sub_q['DII_composite'].mean()), 2),
            'Green_Jobs_Any_DX5_Pct': round(100.0 * float(sub_q['green_jobs_any'].dropna().mean()), 2),
            'Green_Jobs_FTE_Mean': round(float(sub_q['green_jobs_fte'].dropna().mean()), 2),
            'Plan_Invest_Any_Q4_Valid_Pct': round(100.0 * float((q4_valid >= 2).mean()), 2) if len(q4_valid) > 0 else np.nan,
            'Plan_Invest_GT5Pct_Q4_Valid_Pct': round(100.0 * float((q4_valid >= 4).mean()), 2) if len(q4_valid) > 0 else np.nan,
            'Q10_Green_Rev_GT5Pct_Pct': round(100.0 * float((q10_valid >= 2).mean()), 2) if len(q10_valid) > 0 else np.nan,
            'Q10_Green_Rev_GT10Pct_Pct': round(100.0 * float((q10_valid >= 3).mean()), 2) if len(q10_valid) > 0 else np.nan,
            'Q10_Green_Rev_GT30Pct_Pct': round(100.0 * float((q10_valid >= 4).mean()), 2) if len(q10_valid) > 0 else np.nan,
            'Q15_Carbon_Action_4Item_Mean': round(float(q15_valid.mean()), 2) if len(q15_valid) > 0 else np.nan,
            'Q15_LowCarbon_Tech_Pct': round(100.0 * float(q15_tech_valid.mean()), 2) if len(q15_tech_valid) > 0 else np.nan,
        })
    return out


def run_ext2_q8_desired_support(df, d_joint):
    """
    Extension 2 (Appendix M.2): Unrouted Q8 Desired-Support Mechanism Check across all
    N = 13,322 EU-27 SMEs and within the N = 11,128 routed joint sample.
    """
    q8_labels = {
        'q8.1': 'Q8.1 Tool to self-assess resource efficiency',
        'q8.2': 'Q8.2 Consultancy on how to improve resource efficiency',
        'q8.3': 'Q8.3 Grants or subsidies',
        'q8.4': 'Q8.4 Advice on funding possibilities and financial planning',
        'q8.5': 'Q8.5 Demonstration of new technologies/processes',
        'q8.6': 'Q8.6 Database with case studies of sustainable practices',
        'q8.7': 'Q8.7 Stronger cooperation across sectors for reuse/recycling',
        'q8.8': 'Q8.8 Clearer rules on secondary raw materials',
    }
    rows = []
    routed_mask = df['re_action_count'] >= 1
    for col, lbl in q8_labels.items():
        if col not in df.columns:
            continue
        m_all = float(df[col].mean() * 100.0)
        m_unrouted = float(df.loc[~routed_mask, col].mean() * 100.0)
        m_routed = float(df.loc[routed_mask, col].mean() * 100.0)

        m_exp1 = float(d_joint.loc[d_joint['barrier_expertise'] == 1, col].mean() * 100.0)
        m_exp0 = float(d_joint.loc[d_joint['barrier_expertise'] == 0, col].mean() * 100.0)
        diff_exp_raw = m_exp1 - m_exp0

        m_adm1 = float(d_joint.loc[d_joint['barrier_admin'] == 1, col].mean() * 100.0)
        m_adm0 = float(d_joint.loc[d_joint['barrier_admin'] == 0, col].mean() * 100.0)
        diff_adm_raw = m_adm1 - m_adm0

        # Controlled Probit AME in N=11,128 (controlling for markets, size, turnover FE, country FE, sector FE, RECount)
        f_q8 = (f"Q('{col}') ~ B2B_private + B2G_public + B2C + barrier_admin + barrier_expertise "
                f"+ grant_receipt + advice_public + scr10 + re_action_count + C(scr14_cat) + C(country_iso2) + C(nace_b)")
        m_q8 = fit_single_probit_clustered(f_q8, d_joint)
        y_m, X_m = dmatrices(f_q8, d_joint, return_type='dataframe')
        X_arr = X_m.values
        b_arr = m_q8['params'].values
        cols_q8 = list(X_m.columns)

        j_exp = cols_q8.index('barrier_expertise')
        X1_e, X0_e = X_arr.copy(), X_arr.copy()
        X1_e[:, j_exp] = 1.0
        X0_e[:, j_exp] = 0.0
        ame_exp = float((stats.norm.cdf(X1_e @ b_arr) - stats.norm.cdf(X0_e @ b_arr)).mean() * 100.0)

        j_adm = cols_q8.index('barrier_admin')
        X1_a, X0_a = X_arr.copy(), X_arr.copy()
        X1_a[:, j_adm] = 1.0
        X0_a[:, j_adm] = 0.0
        ame_adm = float((stats.norm.cdf(X1_a @ b_arr) - stats.norm.cdf(X0_a @ b_arr)).mean() * 100.0)

        rows.append({
            'Item': col.upper(),
            'Description': lbl,
            'All_EU27_SMEs_13322_Pct': round(m_all, 2),
            'Unrouted_ZeroAction_1231_Pct': round(m_unrouted, 2),
            'Routed_Active_12091_Pct': round(m_routed, 2),
            'Joint_Expertise_Shortage_Pct': round(m_exp1, 2),
            'Joint_No_Expertise_Shortage_Pct': round(m_exp0, 2),
            'Raw_Diff_Expertise_pp': round(diff_exp_raw, 2),
            'Controlled_AME_Expertise_pp': round(ame_exp, 2),
            'Probit_beta_Expertise': float(m_q8['params']['barrier_expertise']),
            'Probit_se_Expertise': float(m_q8['bse']['barrier_expertise']),
            'Probit_p_Expertise': float(m_q8['pvalues']['barrier_expertise']),
            'Joint_Admin_Barrier_Pct': round(m_adm1, 2),
            'Joint_No_Admin_Barrier_Pct': round(m_adm0, 2),
            'Raw_Diff_Admin_pp': round(diff_adm_raw, 2),
            'Controlled_AME_Admin_pp': round(ame_adm, 2),
            'Probit_beta_Admin': float(m_q8['params']['barrier_admin']),
            'Probit_se_Admin': float(m_q8['bse']['barrier_admin']),
            'Probit_p_Admin': float(m_q8['pvalues']['barrier_admin']),
        })
    return pd.DataFrame(rows)


def run_ext3_wave6_replication():
    """
    Extension 3 (Appendix M.3): Out-of-Sample Temporal Replication on 2024 Flash Eurobarometer 549
    (ZA8869_v1-0-0.sav, Wave 6). Harmonizes GR -> EL across all 27 EU member states (N = 13,124 SMEs
    -> N = 11,246 Joint FIML BiProbit). Gracefully returns None if ZA8869 raw file is absent.
    """
    raw_dir = os.path.join(DATA_DIR, "Raw")
    candidates = [
        os.path.join(raw_dir, "ZA8869_v1-0-0.sav"),
        os.path.join(raw_dir, "ZA8869_v2-0-0.sav"),
        os.path.join(raw_dir, "All_Raw_datasets_SME_Res_Eff",
                     "Flash Eurobarometer 549 (SMEs, Resource Efficiency and Green Markets, wave 6)",
                     "ZA8869_v1-0-0.sav"),
    ]
    w6_path = next((p for p in candidates if os.path.exists(p)), None)
    if w6_path is None:
        print("  [Ext3 Wave 6] ZA8869 .sav not found on disk; skipping Wave 6 re-estimation gracefully.")
        return None

    eu27_isos = {
        'FR', 'BE', 'NL', 'DE', 'IT', 'LU', 'DK', 'IE', 'EL', 'ES', 'PT', 'FI', 'SE', 'AT',
        'CY', 'CZ', 'EE', 'HU', 'LV', 'LT', 'MT', 'PL', 'SK', 'SI', 'BG', 'RO', 'HR'
    }
    w6_raw, _ = pyreadstat.read_sav(w6_path)
    w6_raw['country_iso2'] = w6_raw['isocntry'].str.strip().replace({'GR': 'EL'})
    w6_eu = w6_raw[w6_raw['country_iso2'].isin(eu27_isos)].copy()
    w6 = w6_eu[w6_eu['scr10'].isin([1.0, 2.0, 3.0])].copy()

    w6['q9_binary'] = np.where(w6['q9'] == 1.0, 1.0, np.where(w6['q9'].isin([2.0, 3.0]), 0.0, np.nan))
    w6['q14_binary'] = np.where(w6['q14'].isin([1.0, 4.0]), 1.0, np.where(w6['q14'].isin([2.0, 3.0]), 0.0, np.nan))
    w6['barrier_admin'] = np.where(w6['q7.1'].isin([0.0, 1.0]), w6['q7.1'], np.nan)
    w6['barrier_expertise'] = np.where(w6['q7.6'].isin([0.0, 1.0]), w6['q7.6'], np.nan)
    w6['grant_receipt'] = w6['q6.1'].fillna(0.0)
    w6['advice_public'] = w6['q6.4'].fillna(0.0)
    w6['re_action_count'] = (w6[[f'q1.{i}' for i in range(1, 10)]] == 1.0).sum(axis=1).astype(int)
    w6['scr14_cat'] = w6['scr14'].astype(int)

    w6_ivs = ['barrier_admin', 'barrier_expertise', 'grant_receipt', 'advice_public']
    w6_focal = ['barrier_admin', 'barrier_expertise', 'grant_receipt']
    d_w6, Xdf_w6 = bp_mod.build_design_matrix(
        w6, y1_col='q9_binary', y2_col='q14_binary',
        iv_cols=w6_ivs, ctrl_cols=CTRL, turnover_mode='cat', sector_col='nace_b'
    )
    fit_w6 = bp_mod.fit_biprobit_fiml(
        d_w6, Xdf_w6, y1_col='q9_binary', y2_col='q14_binary',
        compute_ames_for=w6_ivs, focal_6=w6_focal
    )
    assert fit_w6['diag']['grad_inf_norm'] < 1e-7 and fit_w6['diag']['hess_min_eig'] > 0
    _, diff_w6, diag_w6 = extract_biprobit_rows(
        fit_w6, 'Wave 6 (2024 ZA8869, N=11,246; SCR15 discontinued in Wave 6)'
    )
    w6_df = pd.DataFrame(diff_w6)
    w6_df['N_EU27_SMEs_Wave6'] = len(w6)
    w6_df['grad_inf_norm'] = diag_w6['grad_inf_norm']
    w6_df['hess_min_eig'] = diag_w6['hess_min_eig']
    w6_df['ll'] = diag_w6['ll']
    return w6_df


def main():
    print("=" * 80)
    print("05_robustness.py: JOINT BIPROBIT ROBUSTNESS, MNLOGIT, H3c MODERATION & EXTENSIONS")
    print("=" * 80)

    df = pd.read_csv(DATA_CSV)
    need_base = IV + CTRL + ['scr14_cat', 'country_iso2', 'nace_b', 'q9_binary', 'q14_binary']
    d0 = df.dropna(subset=need_base).copy()

    # ==========================================================================
    # 1. TABLE 6 JOINT FIML BIPROBIT ROBUSTNESS SUITE
    # ==========================================================================
    all_coef_rows = []
    all_diff_rows = []
    all_diag_rows = []

    def run_joint_spec(label, data_in, y1_c='q9_binary', y2_c='q14_binary',
                       ctrl_c=CTRL, turn_mode='cat', sec_col='nace_b',
                       extra_c=None, w_col=None):
        keep_w = [w_col] if w_col else None
        d_s, X_s = bp_mod.build_design_matrix(
            data_in, y1_col=y1_c, y2_col=y2_c,
            iv_cols=IV, ctrl_cols=ctrl_c,
            turnover_mode=turn_mode, sector_col=sec_col,
            extra_cols=extra_c, merge_top_turnover=False, keep_cols=keep_w
        )
        fit_s = bp_mod.fit_biprobit_fiml(d_s, X_s, y1_col=y1_c, y2_col=y2_c, weight_col=w_col)
        if fit_s['diag']['hess_min_eig'] <= 0 or fit_s['diag']['grad_inf_norm'] >= 1e-6:
            # Fallback merging top turnover brackets 8 & 9 (>10M) if tiny subsample has near-empty top cell
            d_s, X_s = bp_mod.build_design_matrix(
                data_in, y1_col=y1_c, y2_col=y2_c,
                iv_cols=IV, ctrl_cols=ctrl_c,
                turnover_mode=turn_mode, sector_col=sec_col,
                extra_cols=extra_c, merge_top_turnover=True, keep_cols=keep_w
            )
            fit_s = bp_mod.fit_biprobit_fiml(d_s, X_s, y1_col=y1_c, y2_col=y2_c, weight_col=w_col)
        assert fit_s['diag']['grad_inf_norm'] < 1e-7, f"{label} grad={fit_s['diag']['grad_inf_norm']:.2e}"
        assert fit_s['diag']['hess_min_eig'] > 0, f"{label} hess_min_eig={fit_s['diag']['hess_min_eig']}"
        extra_pull = list(extra_c or [])
        if turn_mode == 'clean':
            extra_pull.append('scr14_clean')
        elif turn_mode == 'imp_dk':
            extra_pull.extend(['scr14_imp', 'scr14_dk'])
        c_r, d_r, dg_r = extract_biprobit_rows(fit_s, label, y1_name=y1_c, y2_name=y2_c, extra_vars=extra_pull)
        all_coef_rows.extend(c_r)
        all_diff_rows.extend(d_r)
        all_diag_rows.append(dg_r)
        print(f"  ✓ {label:<62} | N={dg_r['N']:>6,} | ||g||={dg_r['grad_inf_norm']:.1e} | eig_min={dg_r['hess_min_eig']:>6.2f} | rho={dg_r['rho']:.4f}")

    # Row 1: Baseline Joint FIML BiProbit (N = 11,128)
    run_joint_spec('1. Baseline Joint FIML BiProbit (N=11,128)', d0)
    # Row 2: Omitting RECount (N = 11,128)
    run_joint_spec('2. Omitting RECount (Exogenous Pre-Treatment Controls, N=11,128)', d0, ctrl_c=['scr10'])
    # Row 3: 10-Item RECount Including 'Other' (Q1.1-Q1.10, N = 11,128)
    run_joint_spec('3. 10-Item RECount Incl. Other (Q1.1-Q1.10, N=11,128)', d0, ctrl_c=['scr10', 're_action_count_10'])
    # Row 4: Within-Country SME Weights PML (w1_sme, N = 11,128)
    run_joint_spec('4. Within-Country SME Weights PML (w1_sme, N=11,128)', d0.dropna(subset=['w1_sme']), w_col='w1_sme')
    # Row 5: EU-27 Population Weights PML (w93_sme, N = 11,128)
    run_joint_spec('5. EU-27 Population Weights PML (w93_sme, N=11,128)', d0.dropna(subset=['w93_sme']), w_col='w93_sme')
    # Row 6: Strict Active 2030 Planners (Excl. Q14=4 Already Neutral, N = 10,624)
    run_joint_spec('6. Strict Active 2030 Planners (Excl. Q14=4, N=10,624)', d0.dropna(subset=['q14_strict']), y2_c='q14_strict')
    # Row 7: External Support Seekers (Q5.3=1, Joint Sample N = 2,959)
    run_joint_spec('7. External Support Seekers (Q5.3=1, Joint N=2,959)', d0[d0['q5.3'] == 1].copy())
    # Row 8: Product-selling SMEs (SCR16 in {1, 3}) (N = 7,299)
    run_joint_spec('8. Product-selling SMEs (SCR16 in {1, 3}) (N=7,299)', d0[d0['sells_products'] == 1].copy())
    # Row 9: Expanded Covariates (+Q7.5, Q7.9, Q5.1, Q5.2, Q4, SCR16; N = 9,857)
    run_joint_spec('9. Expanded Covariates (+Q7.5/Q7.9/Q5.1/Q5.2/Q4/SCR16, N=9,857)',
                   d0.dropna(subset=EXPANDED_VARS).copy(), extra_c=EXPANDED_VARS)
    # Row 10: Granular 12-Section NACE Rev. 2 Fixed Effects (nace_a, 11 indicators; N = 11,128)
    run_joint_spec('10. 12-Section NACE Rev. 2 Fixed Effects (nace_a, N=11,128)', d0, sec_col='nace_a')
    # Row 11: Complete-Case Ordinal Turnover (scr14 < 10, N = 9,764)
    run_joint_spec('11. Complete-Case Ordinal Turnover (scr14 < 10, N=9,764)', d0, turn_mode='clean')
    # Row 12: Mean-Imputed Ordinal Turnover + DK Indicator (scr14_imp + scr14_dk, N = 11,128)
    run_joint_spec('12. Mean-Imputed Ordinal Turnover + DK Dummy (N=11,128)', d0, turn_mode='imp_dk')

    # Single-outcome Q5.3=1 check on all valid Q14 (N = 3,055)
    dr_3055 = df[df['q5.3'] == 1].dropna(subset=[c for c in need_base if c != 'q9_binary']).copy()
    m_3055 = fit_single_probit_clustered(f"q14_binary ~ {BASE_FORMULA}", dr_3055)
    for v in IV + CTRL:
        if v in m_3055['params'].index:
            all_coef_rows.append(dict(
                spec='7b. External Support Seekers (Q5.3=1, All Valid Q14 Single-Eq N=3,055)',
                outcome='q14_binary', var=v,
                beta=float(m_3055['params'][v]), se=float(m_3055['bse'][v]),
                z=float(m_3055['params'][v] / m_3055['bse'][v]),
                p=float(m_3055['pvalues'][v]), p_t26=float(m_3055['p_t26'][v]),
                N=int(m_3055['nobs'])
            ))

    # Sector-stratified Joint BiProbits (Appendix D: nace_b in {1, 2, 3, 4})
    sector_names = {
        1: 'Sector 1: Manufacturing (NACE C)',
        2: 'Sector 2: Retail (NACE G)',
        3: 'Sector 3: Services (NACE H-M)',
        4: 'Sector 4: Industry (NACE B/D/E/F)'
    }
    for s_val, s_lbl in sector_names.items():
        g_sub = d0[d0['nace_b'] == s_val].copy()
        # Within a single sector, sector FE has 1 category -> automatically omitted
        parts_s = [
            g_sub[IV + CTRL].astype(float),
            pd.get_dummies(g_sub['scr14_cat'].astype(int), prefix='turn', drop_first=True, dtype=float),
            pd.get_dummies(g_sub['country_iso2'], prefix='c', drop_first=True, dtype=float)
        ]
        X_sec = pd.concat(parts_s, axis=1)
        X_sec = X_sec.loc[:, X_sec.nunique() > 1]
        X_sec.insert(0, 'const', 1.0)
        fit_sec = bp_mod.fit_biprobit_fiml(g_sub, X_sec, y1_col='q9_binary', y2_col='q14_binary')
        if fit_sec['diag']['hess_min_eig'] <= 0 or fit_sec['diag']['grad_inf_norm'] >= 1e-6:
            t_m = g_sub['scr14_cat'].astype(int).replace({9: 8})
            X_sec = pd.concat([
                g_sub[IV + CTRL].astype(float),
                pd.get_dummies(t_m, prefix='turn', drop_first=True, dtype=float),
                pd.get_dummies(g_sub['country_iso2'], prefix='c', drop_first=True, dtype=float)
            ], axis=1)
            X_sec = X_sec.loc[:, X_sec.nunique() > 1]
            X_sec.insert(0, 'const', 1.0)
            fit_sec = bp_mod.fit_biprobit_fiml(g_sub, X_sec, y1_col='q9_binary', y2_col='q14_binary')
        c_r, d_r, dg_r = extract_biprobit_rows(fit_sec, s_lbl)
        all_coef_rows.extend(c_r)
        all_diff_rows.extend(d_r)
        all_diag_rows.append(dg_r)
        print(f"  ✓ {s_lbl:<62} | N={dg_r['N']:>6,} | ||g||={dg_r['grad_inf_norm']:.1e} | eig_min={dg_r['hess_min_eig']:>6.2f}")

    rob_df = pd.DataFrame(all_coef_rows)
    rob_diffs_df = pd.DataFrame(all_diff_rows)
    rob_diag_df = pd.DataFrame(all_diag_rows)

    # ==========================================================================
    # 2. PROBABILITY-SCALE MODERATION (H3c) WITH advice_public INCLUDED
    # ==========================================================================
    sub = d0.copy()
    f_mod = (
        'q14_binary ~ grant_receipt * barrier_admin + grant_receipt * barrier_expertise '
        '+ B2B_private + B2G_public + B2C + advice_public + scr10 + re_action_count '
        '+ C(scr14_cat) + C(country_iso2) + C(nace_b)'
    )
    m_mod = smf.logit(f_mod, sub).fit(
        disp=0, maxiter=300, cov_type='cluster',
        cov_kwds={'groups': pd.factorize(sub['country_iso2'])[0]}
    )
    di_mod, cov_mod, beta_mod = m_mod.model.data.design_info, m_mod.cov_params().values, m_mod.params.values
    G_mod = sub['country_iso2'].nunique()

    def pg(g, a=None, e=None):
        d_cf = sub.copy()
        d_cf['grant_receipt'] = g
        if a is not None:
            d_cf['barrier_admin'] = a
        if e is not None:
            d_cf['barrier_expertise'] = e
        X_cf = np.asarray(build_design_matrices([di_mod], d_cf)[0])
        p_cf = expit(X_cf @ beta_mod)
        return p_cf.mean(), ((p_cf * (1.0 - p_cf))[:, None] * X_cf).mean(axis=0)

    mod_rows = []
    for dim, kw, inter_term in [
        ('Admin', 'a', 'grant_receipt:barrier_admin'),
        ('Expertise', 'e', 'grant_receipt:barrier_expertise')
    ]:
        P = {(g, b): pg(g, **{kw: b}) for g in (0, 1) for b in (0, 1)}
        for lab, est, gr in [
            (f'AME(support | {dim}=0)', P[1, 0][0] - P[0, 0][0], P[1, 0][1] - P[0, 0][1]),
            (f'AME(support | {dim}=1)', P[1, 1][0] - P[0, 1][0], P[1, 1][1] - P[0, 1][1]),
            (f'Delta-AME ({dim}=1 vs 0)',
             (P[1, 1][0] - P[0, 1][0]) - (P[1, 0][0] - P[0, 0][0]),
             (P[1, 1][1] - P[0, 1][1]) - (P[1, 0][1] - P[0, 0][1]))
        ]:
            se_m = float(np.sqrt(gr @ cov_mod @ gr))
            z_m = float(est / se_m)
            mod_rows.append(dict(
                quantity=lab, est=float(est), se=se_m, z=z_m,
                lo=float(est - 1.96 * se_m), hi=float(est + 1.96 * se_m),
                p=float(2.0 * (1.0 - stats.norm.cdf(abs(z_m)))),
                p_t26=float(2.0 * (1.0 - stats.t.cdf(abs(z_m), df=G_mod - 1))),
                N=int(m_mod.nobs)
            ))
        b_int = float(m_mod.params[inter_term])
        se_int = float(m_mod.bse[inter_term])
        z_int = b_int / se_int
        mod_rows.append(dict(
            quantity=f'Logit interaction ({inter_term})',
            est=b_int, se=se_int, z=z_int,
            lo=b_int - 1.96 * se_int, hi=b_int + 1.96 * se_int,
            p=float(2.0 * (1.0 - stats.norm.cdf(abs(z_int)))),
            p_t26=float(2.0 * (1.0 - stats.t.cdf(abs(z_int), df=G_mod - 1))),
            N=int(m_mod.nobs)
        ))
    mod_df = pd.DataFrame(mod_rows)

    # ==========================================================================
    # 3. 4-QUADRANT MULTINOMIAL LOGIT (N = 11,128)
    # ==========================================================================
    d_mnl = d0.copy()
    d_mnl['quad_4'] = np.select(
        [(d_mnl['q9_binary'] == 0) & (d_mnl['q14_binary'] == 0),
         (d_mnl['q9_binary'] == 1) & (d_mnl['q14_binary'] == 0),
         (d_mnl['q9_binary'] == 0) & (d_mnl['q14_binary'] == 1),
         (d_mnl['q9_binary'] == 1) & (d_mnl['q14_binary'] == 1)],
        [0, 1, 2, 3]
    )
    mnl_fit = smf.mnlogit(f"quad_4 ~ {BASE_FORMULA}", d_mnl).fit(
        disp=0, maxiter=300, cov_type='cluster',
        cov_kwds={'groups': pd.factorize(d_mnl['country_iso2'])[0]}
    )
    G_mnl = d_mnl['country_iso2'].nunique()
    mnl_vars = IV + CTRL

    di_mnl = mnl_fit.model.data.design_info
    K_mnl = len(mnl_fit.params.index)
    B_mnl = mnl_fit.params.values  # shape (K, 3)

    # Explicitly reorder cov_params() to match column-stacked [eq1_k0..K, eq2_k0..K, eq3_k0..K]
    cov_raw = mnl_fit.cov_params()
    eq_keys = list(cov_raw.index.levels[0])
    # Map equation labels for j=1,2,3 in sorted order
    eq_sorted = sorted(eq_keys, key=lambda x: int(x))
    ordered_multi_idx = [(eq_lbl, pname) for eq_lbl in eq_sorted for pname in mnl_fit.params.index]
    cov_mnl = cov_raw.loc[ordered_multi_idx, ordered_multi_idx].values

    def mnl_probs_and_grad(X_eval, B_mat):
        logits = np.column_stack([np.zeros(len(X_eval)), X_eval @ B_mat])
        P = softmax(logits, axis=1)
        p_mean = P.mean(axis=0)
        grad = np.zeros((4, 3 * K_mnl))
        for m in range(4):
            for r_idx, r in enumerate([1, 2, 3]):
                w_mr = P[:, m] * ((1.0 if m == r else 0.0) - P[:, r])
                grad[m, r_idx * K_mnl:(r_idx + 1) * K_mnl] = (w_mr[:, None] * X_eval).mean(axis=0)
        return p_mean, grad

    mnl_rows = []
    for v in mnl_vars:
        row = {'variable': v}
        for j in [1, 2, 3]:
            col_idx = j - 1
            b = float(mnl_fit.params.loc[v, col_idx])
            se = float(mnl_fit.bse.loc[v, col_idx])
            z = b / se
            row[f'beta_q{j}'] = b
            row[f'se_q{j}'] = se
            row[f'z_q{j}'] = z
            row[f'p_q{j}'] = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
            row[f'p_t26_q{j}'] = float(2.0 * (1.0 - stats.t.cdf(abs(z), df=G_mnl - 1)))

        is_binary = set(np.unique(d_mnl[v].dropna())).issubset({0, 1, 0.0, 1.0})
        if is_binary:
            d1 = d_mnl.copy(); d1[v] = 1
            d0_v = d_mnl.copy(); d0_v[v] = 0
            X1 = np.asarray(build_design_matrices([di_mnl], d1)[0])
            X0 = np.asarray(build_design_matrices([di_mnl], d0_v)[0])
            p1, g1 = mnl_probs_and_grad(X1, B_mnl)
            p0, g0 = mnl_probs_and_grad(X0, B_mnl)
            ame_vec, grad_ame = p1 - p0, g1 - g0
        else:
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
            row[f'ame_q{m}'] = est_m
            row[f'ame_se_q{m}'] = se_m
            row[f'ame_p_q{m}'] = float(2.0 * (1.0 - stats.norm.cdf(abs(z_m))))
            row[f'ame_p_t26_q{m}'] = float(2.0 * (1.0 - stats.t.cdf(abs(z_m), df=G_mnl - 1)))
        mnl_rows.append(row)
    mnl_df = pd.DataFrame(mnl_rows)

    # ==========================================================================
    # 4. CONSTRUCT VALIDITY TABLE (BOTH N = 12,205 AND N = 11,128)
    # ==========================================================================
    d_typ_12205 = df.dropna(subset=['q9_binary', 'q14_binary']).copy()
    cv_12205 = build_construct_validity_profile(d_typ_12205, 'Full_Typology_N12205')
    cv_11128 = build_construct_validity_profile(d0, 'Primary_Joint_Sample_N11128')
    cv_both_df = pd.DataFrame(cv_12205 + cv_11128)
    p9_df = pd.DataFrame(cv_11128[:-1])  # 4 quadrants in N=11,128 for legacy compatibility

    # ==========================================================================
    # 5. ROUND-2 EMPIRICAL EXTENSIONS (APPENDIX M.1, M.2, M.3)
    # ==========================================================================
    print("\n[Running Appendix M Empirical Extensions]")
    ext1_rows = []
    for ext_lbl, y2_tight_col in [
        ('Ext1a: Tightened Construct BiProbit (Q10>10% vs Q15.2/Q15.3 Tech; Keep Q14=4 as 0, N=10,687)', 'y2_tight_q15_tech'),
        ('Ext1b: Tightened Construct BiProbit (Q10>10% vs Q15.2/Q15.3 Tech; Excl. Q14=4, N=10,203)', 'y2_tight_q15_tech_excl4'),
    ]:
        d_t, X_t = bp_mod.build_design_matrix(
            df[df['barrier_admin'].notna()].copy(),
            y1_col='y1_tight_q10_gt10', y2_col=y2_tight_col,
            iv_cols=IV, ctrl_cols=CTRL, turnover_mode='cat', sector_col='nace_b'
        )
        fit_t = bp_mod.fit_biprobit_fiml(d_t, X_t, y1_col='y1_tight_q10_gt10', y2_col=y2_tight_col)
        assert fit_t['diag']['grad_inf_norm'] < 1e-7 and fit_t['diag']['hess_min_eig'] > 0
        _, diff_t, dg_t = extract_biprobit_rows(fit_t, ext_lbl, y1_name='y1_tight_q10_gt10', y2_name=y2_tight_col)
        for r in diff_t:
            r['ll'] = dg_t['ll']
            r['grad_inf_norm'] = dg_t['grad_inf_norm']
            r['hess_min_eig'] = dg_t['hess_min_eig']
            ext1_rows.append(r)
        print(f"  ✓ {ext_lbl} | N={dg_t['N']:,} | rho={dg_t['rho']:.4f} | ||g||={dg_t['grad_inf_norm']:.1e}")
    ext1_df = pd.DataFrame(ext1_rows)

    ext2_df = run_ext2_q8_desired_support(df, d0)
    print(f"  ✓ Ext2 Q8 Desired Support Mechanism Check completed (N=13,322 & N=11,128)")

    ext3_df = run_ext3_wave6_replication()
    if ext3_df is not None:
        print(f"  ✓ Ext3 Wave 6 (2024 ZA8869) Joint FIML BiProbit completed (N={int(ext3_df['N'].iloc[0]):,})")

    # Save standalone Robustness_FE.xlsx
    with pd.ExcelWriter(OUT_ROB_XLSX, engine='openpyxl') as w:
        rob_df.to_excel(w, sheet_name='robustness', index=False)
        rob_diffs_df.to_excel(w, sheet_name='robustness_crosseq_diffs', index=False)
        rob_diag_df.to_excel(w, sheet_name='robustness_diagnostics', index=False)
        mod_df.to_excel(w, sheet_name='moderation_clustered', index=False)
        mnl_df.to_excel(w, sheet_name='mnlogit_4quadrants', index=False)
        cv_both_df.to_excel(w, sheet_name='construct_validity', index=False)
        ext1_df.to_excel(w, sheet_name='ext1_tightened_biprobit', index=False)
        ext2_df.to_excel(w, sheet_name='ext2_q8_desired_support', index=False)
        if ext3_df is not None:
            ext3_df.to_excel(w, sheet_name='ext3_wave6_za8869', index=False)

    # ==========================================================================
    # 6. MASTER CONSOLIDATION INTO Data/Results_Consolidated.xlsx
    # ==========================================================================
    preserved_sheets = {}
    keep_from_existing = [
        'CausalForest_ATE', 'CausalForest_CATEs', 'IPW_Diagnostics',
        'PSM_ATT', 'PSM_Balance', 'PSM_Sensitivity_Specs', 'Rosenbaum_Bounds',
        'DI_Moderation_OLS', 'DI_Country_Level',
        'GreenPremium_CrossTab', 'GreenPremium_OrderedProbit',
        'Ext3_Wave6_ZA8869_Replication'
    ]
    if os.path.exists(RES_CONSOL):
        existing_xl = pd.ExcelFile(RES_CONSOL)
        for sh in keep_from_existing:
            if sh in existing_xl.sheet_names:
                preserved_sheets[sh] = pd.read_excel(RES_CONSOL, sheet_name=sh)

    bp_diag = pd.read_excel(BP_XLSX, sheet_name='diagnostics')
    rho_v = float(bp_diag['rho'].iloc[0])
    se_rho_v = float(bp_diag['se_rho'].iloc[0])
    z_rho_v = float(bp_diag['z_rho'].iloc[0])

    readme_df = pd.DataFrame([
        {'Field': 'Manuscript Title', 'Details': 'Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs'},
        {'Field': 'Running Head', 'Details': 'Green Offerings and Net-Zero Plans in SMEs'},
        {'Field': 'Authors', 'Details': 'Sezgin Tunca, Mausam Budhathoki, and Hari Gopal Risal'},
        {'Field': 'Target Journal', 'Details': 'Business Strategy and the Environment'},
        {'Field': 'Canonical Zenodo Concept DOI', 'Details': '10.5281/zenodo.23619994 (https://doi.org/10.5281/zenodo.23619994)'},
        {'Field': 'Data Sources', 'Details': 'Flash Eurobarometer 498 (ZA7806_v1-0-0.sav, 2021), Eurostat Digital Intensity Index (isoc_e_dii, 2021), and Wave 6 Flash Eurobarometer 549 (ZA8869_v1-0-0.sav, 2024)'},
        {'Field': 'Analytical Samples', 'Details': 'N=13,322 EU-27 SMEs; N=12,205 4-Quadrant Typology; N=11,521 IPW/CATE; N=11,128 Primary Joint Econometric Sample'},
        {'Field': 'Primary Specification', 'Details': 'Joint FIML Bivariate Probit with 9 categorical turnover bracket FEs (scr14_cat), 26 country FEs, 3 NACE macro-sector FEs, 9-item substantive RECount, and G=27 country-clustered SEs + finite-cluster t_26 + m=6 focal Holm adjustment'},
        {'Field': 'Disturbance Correlation (rho)', 'Details': f'rho = {rho_v:.4f} (cluster SE = {se_rho_v:.4f}, z = {z_rho_v:.2f}, p < 0.001)'},
        {'Field': 'One-Click Reproduction', 'Details': 'python run_all_analysis.py && python generate_all_figures.py'}
    ])

    with pd.ExcelWriter(RES_CONSOL, engine='openpyxl', mode='w') as writer:
        readme_df.to_excel(writer, sheet_name='Readme_Metadata', index=False)
        if os.path.exists(INP_XLSX):
            inp_xl = pd.ExcelFile(INP_XLSX)
            for sh in ['Sample_Waterfall', 'Table1_Typology_12205', 'Missing_Data_Attrition']:
                if sh in inp_xl.sheet_names:
                    pd.read_excel(INP_XLSX, sheet_name=sh).to_excel(writer, sheet_name=sh, index=False)
        if os.path.exists(BP_XLSX):
            bp_coef = pd.read_excel(BP_XLSX, sheet_name='coef')
            bp_cet = pd.read_excel(BP_XLSX, sheet_name='cross_eq_tests')
            bp_ame = pd.read_excel(BP_XLSX, sheet_name='AME')
            bp_fe_w = pd.read_excel(BP_XLSX, sheet_name='fe_wald_tests')
            bp_coef.to_excel(writer, sheet_name='BiProbit_Coefficients', index=False)
            bp_coef.to_excel(writer, sheet_name='BiProbit_Joint_FE', index=False)
            bp_cet.to_excel(writer, sheet_name='BiProbit_CrossEq_Tests', index=False)
            bp_cet.to_excel(writer, sheet_name='Cross_Eq_Tests', index=False)
            bp_ame.to_excel(writer, sheet_name='BiProbit_AME', index=False)
            bp_diag.to_excel(writer, sheet_name='BiProbit_Diagnostics', index=False)
            bp_fe_w.to_excel(writer, sheet_name='BiProbit_FE_Wald_Tests', index=False)
        mnl_df.to_excel(writer, sheet_name='MNLogit_4Quadrants', index=False)
        for sh in [s for s in keep_from_existing if s != 'Ext3_Wave6_ZA8869_Replication']:
            if sh in preserved_sheets:
                preserved_sheets[sh].to_excel(writer, sheet_name=sh, index=False)
        rob_df.to_excel(writer, sheet_name='Robustness_All', index=False)
        rob_diffs_df.to_excel(writer, sheet_name='Robustness_CrossEq_Diffs', index=False)
        rob_diag_df.to_excel(writer, sheet_name='Robustness_Diagnostics', index=False)
        mod_df.to_excel(writer, sheet_name='AppendixJ_ProbScale_Moderation', index=False)
        mod_df.to_excel(writer, sheet_name='Moderation_Clustered', index=False)
        cv_both_df.to_excel(writer, sheet_name='Construct_Validity_Table1', index=False)
        p9_df.to_excel(writer, sheet_name='Point9_Construct_Validity', index=False)
        ext1_df.to_excel(writer, sheet_name='Ext1_Tightened_Construct_BiProbit', index=False)
        ext2_df.to_excel(writer, sheet_name='Ext2_Q8_Desired_Support', index=False)
        if ext3_df is not None:
            ext3_df.to_excel(writer, sheet_name='Ext3_Wave6_ZA8869_Replication', index=False)
        elif 'Ext3_Wave6_ZA8869_Replication' in preserved_sheets:
            preserved_sheets['Ext3_Wave6_ZA8869_Replication'].to_excel(writer, sheet_name='Ext3_Wave6_ZA8869_Replication', index=False)

    print("\n[H3c Probability-Scale Moderation (incl. advice_public)]")
    print(mod_df.round(4).to_string(index=False))
    print("\n[4-Quadrant Multinomial Logit Discrete AMEs]")
    print(mnl_df[['variable', 'ame_q0', 'ame_p_q0', 'ame_q1', 'ame_p_q1', 'ame_q2', 'ame_p_q2', 'ame_q3', 'ame_p_q3']].round(4).to_string(index=False))
    print("\n[Ext1 Tightened Construct-Intensity BiProbit]")
    print(ext1_df[['spec', 'var', 'beta_y1', 'p_y1', 'beta_y2', 'p_y2', 'diff_b1_minus_b2', 'p_diff']].round(4).to_string(index=False))
    if ext3_df is not None:
        print("\n[Ext3 Wave 6 (2024 ZA8869) Out-of-Sample Replication]")
        print(ext3_df[['var', 'beta_y1', 'p_y1', 'beta_y2', 'p_y2', 'diff_b1_minus_b2', 'p_diff', 'rho']].round(4).to_string(index=False))
    print(f"\n  ✓ Consolidated all sheets into: {RES_CONSOL}")


if __name__ == '__main__':
    main()
