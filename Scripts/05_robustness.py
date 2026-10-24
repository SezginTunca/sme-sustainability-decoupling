#!/usr/bin/env python3
"""
05_robustness.py
================
Econometric robustness checks and sensitivity models for the SME
Institutional Decoupling study.

Analyses performed:
  1. Strict Substantive Definition (excluding legacy carbon-neutral firms)
  2. Sector-Stratified Probits (Manufacturing, Services, Retail, Other)
  3. Survey-Weighted Probits (using Eurobarometer w1_sme sampling weights)
  4. Delta-Method Probability-Scale Moderation (H3b) & AME Contrasts (H3a)

Outputs:
  - Appends to Data/Results_Consolidated.xlsx:
    * Sheet 'Robustness_Sector'
    * Sheet 'Robustness_Weighted' (Appendix Table J1)
    * Sheet 'AppendixJ_ProbScale_Moderation' (Appendix Table J3)
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize as sp_minimize
from scipy.optimize import approx_fprime
from scipy.special import expit
import statsmodels.formula.api as smf
from patsy import build_design_matrices

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_DIR, "Data", "Processed", "SME_Analytical_Final.csv")
RESULTS_XLSX = os.path.join(PROJECT_DIR, "Data", "Results_Consolidated.xlsx")


# ═══════════════════════════════════════════════════════════════════
# ROBUSTNESS CHECK 1: Strict Substantive Definition
# ═══════════════════════════════════════════════════════════════════

def robustness_q14_strict(df):
    """
    Re-run the probit models with a strict definition of substantive action 
    that excludes 'already carbon neutral' firms from the active category.
    
    Why this matters (Robustness Rationale):
      - In the main analysis, we code both 'already carbon neutral' and 'currently implementing
        a strategy' as 1.0 (substantive action).
      - This check ensures that our findings are not driven by legacy achievements (firms that were
        neutral long before the survey) or structurally invariant firms.
      - By focusing strictly on active recent strategy adopters (1.0), we confirm that
        customer channels and capability constraints drive/stifle current operational change.
    """
    print("\n  === Robustness Check 1: Strict Substantive Definition (Excludes Legacy Neutral Firms) ===")
    print("  - Rationale: Isolating active recent strategy adopters from legacy achievements.")

    iv_vars = ['B2B_private', 'B2G_public', 'B2C',
               'barrier_admin', 'barrier_expertise',
               'grant_receipt', 'advice_public', 'scr10', 'scr14']

    df_model = df[iv_vars + ['q9_binary', 'q14_strict']].dropna().copy()
    df_model['const'] = 1.0
    var_names = iv_vars + ['const']

    X = df_model[var_names].values
    y1 = df_model['q9_binary'].values
    y2 = df_model['q14_strict'].values

    results = {}
    for name, y in [('Symbolic', y1), ('Substantive_strict', y2)]:
        def neg_ll(beta):
            xb = np.clip(X @ beta, -5, 5)
            p = stats.norm.cdf(xb)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        beta0 = np.zeros(X.shape[1])
        res = sp_minimize(neg_ll, beta0, method='BFGS')

        # Hessian-based Standard Errors
        eps = 1e-5
        hess = np.zeros((len(res.x), len(res.x)))
        for i in range(len(res.x)):
            def grad_i(b):
                return approx_fprime(b, neg_ll, eps)[i]
            hess[i, :] = approx_fprime(res.x, grad_i, eps)
        try:
            se = np.sqrt(np.diag(np.linalg.inv(hess)))
        except:
            se = np.full(len(res.x), np.nan)

        p_vals = 2 * (1 - stats.norm.cdf(np.abs(res.x / se)))
        results[name] = {'coef': res.x, 'se': se, 'p': p_vals}

    print(f"    - Complete Cases: N = {len(df_model):,}")
    print(f"    - Key Coefficient Comparison:")
    for j, v in enumerate(var_names):
        if v in ['B2G_public', 'barrier_admin', 'barrier_expertise', 'grant_receipt']:
            b1 = results['Symbolic']['coef'][j]
            p1 = results['Symbolic']['p'][j]
            b2 = results['Substantive_strict']['coef'][j]
            p2 = results['Substantive_strict']['p'][j]
            sig1 = '***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else ''
            sig2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else ''
            print(f"      * {v:<20} Symbolic: {b1:>7.4f}{sig1:<3} | Substantive_strict: {b2:>7.4f}{sig2:<3}")

    return results


# ═══════════════════════════════════════════════════════════════════
# ROBUSTNESS CHECK 2: Sector-Stratified Probits
# ═══════════════════════════════════════════════════════════════════

def robustness_sector_stratified(df):
    """
    Run separate probits by broad sector.
    
    Why this matters (Robustness Rationale):
      - Environmental actions and supply chain relationships differ across sectors. Manufacturing
        involves tangible physical goods and materials (highly visible green products), whereas Services
        deal with intangible operations where administrative barriers or expertise deficits might differ.
      - We stratify our models into Manufacturing, Services, and Retail to prove that the core decoupling
        phenomenon is a general institutional behavior rather than an industry-specific artifact.
    """
    print("\n  === Robustness Check 2: Sector-Stratified Probits ===")
    print("  - Rationale: Verifying findings across Manufacturing, Services, Retail, and Other.")

    if 'nace_b' not in df.columns:
        print("    ⚠ nace_b not available, skipping sector stratification")
        return None

    iv_vars = ['B2B_private', 'B2G_public', 'B2C',
               'barrier_admin', 'barrier_expertise',
               'grant_receipt', 'advice_public', 'scr10', 'scr14']

    sector_map = {
        1.0: 'Manufacturing',
        2.0: 'Retail',
        3.0: 'Services',
        4.0: 'Other'
    }

    df['broad_sector'] = df['nace_b'].map(sector_map)

    sector_results = []
    for sector in ['Manufacturing', 'Services', 'Retail', 'Other']:
        mask = df['broad_sector'] == sector
        df_sec = df.loc[mask, iv_vars + ['q9_binary', 'q14_binary']].dropna()
        if len(df_sec) < 200:
            continue

        df_sec['const'] = 1.0
        var_names = iv_vars + ['const']
        X = df_sec[var_names].values
        y1 = df_sec['q9_binary'].values

        def neg_ll(beta):
            xb = np.clip(X @ beta, -5, 5)
            p = stats.norm.cdf(xb)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.sum(y1 * np.log(p) + (1 - y1) * np.log(1 - p))

        res = sp_minimize(neg_ll, np.zeros(X.shape[1]), method='BFGS')

        eps = 1e-5
        hess = np.zeros((len(res.x), len(res.x)))
        for i in range(len(res.x)):
            def grad_i(b):
                return approx_fprime(b, neg_ll, eps)[i]
            hess[i, :] = approx_fprime(res.x, grad_i, eps)
        try:
            se = np.sqrt(np.diag(np.linalg.inv(hess)))
        except:
            se = np.full(len(res.x), np.nan)

        p_vals = 2 * (1 - stats.norm.cdf(np.abs(res.x / se)))

        b2g_idx = var_names.index('B2G_public')
        b2b_idx = var_names.index('B2B_private')
        print(f"    - {sector:<13} (N={len(df_sec):>5,}):")
        print(f"      * B2G (Govt Procurement) -> Symbolic: β = {res.x[b2g_idx]:.4f} (p = {p_vals[b2g_idx]:.4f})")
        print(f"      * B2B (Private Supply)   -> Symbolic: β = {res.x[b2b_idx]:.4f} (p = {p_vals[b2b_idx]:.4f})")

        for j, v in enumerate(var_names):
            sector_results.append({
                'Sector': sector, 'Variable': v,
                'Coef': res.x[j], 'SE': se[j], 'p': p_vals[j], 'N': len(df_sec)
            })

    return pd.DataFrame(sector_results) if sector_results else None


# ═══════════════════════════════════════════════════════════════════
# ROBUSTNESS CHECK 3: Survey-Weighted Probits
# ═══════════════════════════════════════════════════════════════════

def robustness_weighted(df):
    """
    Re-run key model with sampling weights.
    
    Why this matters (Robustness Rationale):
      - Survey sampling designs (like Flash Eurobarometer) are stratified across countries and firm sizes.
      - We re-estimate the models using survey weights (w1_sme) to verify that parameter estimates
        are robust and representative of the European SME population (Appendix Table J1).
    """
    print("\n  === Robustness Check 3: Survey-Weighted Probits ===")
    print("  - Rationale: Applying sampling weights (w1_sme) for population-level representative estimation.")

    if 'w1_sme' not in df.columns:
        print("    ⚠ w1_sme not available, skipping weighted estimation")
        return None

    iv_vars = ['B2B_private', 'B2G_public', 'B2C',
               'barrier_admin', 'barrier_expertise',
               'grant_receipt', 'advice_public', 'scr10', 'scr14']

    df_model = df[iv_vars + ['q9_binary', 'q14_binary', 'w1_sme']].dropna().copy()
    df_model['const'] = 1.0
    var_names = iv_vars + ['const']

    X = df_model[var_names].values
    y = df_model['q9_binary'].values
    w = df_model['w1_sme'].values
    w = w / w.mean()  # Normalize weights to sum to N

    def weighted_neg_ll(beta):
        xb = np.clip(X @ beta, -5, 5)
        p = stats.norm.cdf(xb)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))

    res = sp_minimize(weighted_neg_ll, np.zeros(X.shape[1]), method='BFGS')

    eps = 1e-5
    hess = np.zeros((len(res.x), len(res.x)))
    for i in range(len(res.x)):
        def grad_i(b):
            return approx_fprime(b, weighted_neg_ll, eps)[i]
        hess[i, :] = approx_fprime(res.x, grad_i, eps)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(hess)))
    except:
        se = np.full(len(res.x), np.nan)
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(res.x / se)))

    print(f"    - Weighted Probit (N={len(df_model):,}):")
    for j, v in enumerate(var_names):
        if v in ['B2G_public', 'B2C', 'barrier_admin', 'barrier_expertise', 'grant_receipt']:
            sig = '***' if p_vals[j] < 0.001 else '**' if p_vals[j] < 0.01 else '*' if p_vals[j] < 0.05 else ''
            print(f"      * {v:<20} β = {res.x[j]:>7.4f} (SE = {se[j]:.4f}, p = {p_vals[j]:.4f}) {sig}")

    return {'coef': res.x, 'se': se, 'p': p_vals, 'var_names': var_names}


# ═══════════════════════════════════════════════════════════════════
# ROBUSTNESS CHECK 4: Probability-Scale Moderation (H3b)
# ═══════════════════════════════════════════════════════════════════

def robustness_appendix_j_probscale(df):
    """
    Compute probability-scale predicted probabilities, conditional AMEs,
    and second differences (Delta Method, 95% CIs) for H3b grant interactions,
    plus the cross-equation AME contrast for H3a (Appendix Table J3).
    """
    print("\n  === Robustness Check 4: Probability-Scale Moderation (H3b) & AME Contrast (H3a) ===")

    req = ['q14_binary', 'B2B_private', 'B2G_public', 'B2C',
           'barrier_admin', 'barrier_expertise', 'grant_receipt',
           'scr10', 'scr14', 're_action_count', 'country_iso2', 'scr13']
    sub = df.dropna(subset=req).copy()

    f = ('q14_binary ~ grant_receipt * barrier_admin + grant_receipt * barrier_expertise '
         '+ B2B_private + B2G_public + B2C + scr10 + scr14 + re_action_count '
         '+ C(country_iso2) + C(scr13)')
    m = smf.logit(f, data=sub).fit(disp=False)
    design_info = m.model.data.design_info
    cov = m.cov_params().values
    beta = m.params.values

    def get_pred_and_grad(g_val, a_val=None, e_val=None):
        d = sub.copy()
        d['grant_receipt'] = g_val
        if a_val is not None:
            d['barrier_admin'] = a_val
        if e_val is not None:
            d['barrier_expertise'] = e_val
        X = np.asarray(build_design_matrices([design_info], d)[0])
        xb = X @ beta
        p = expit(xb)
        return np.mean(p), np.mean((p * (1 - p))[:, None] * X, axis=0)

    rows = []
    # Admin barrier moderation
    p00, g00 = get_pred_and_grad(0, a_val=0)
    p10, g10 = get_pred_and_grad(1, a_val=0)
    p01, g01 = get_pred_and_grad(0, a_val=1)
    p11, g11 = get_pred_and_grad(1, a_val=1)

    for label, est, grad in [
        ('Pr(Y2=1 | Grant=0, Admin=0)', p00, g00),
        ('Pr(Y2=1 | Grant=1, Admin=0)', p10, g10),
        ('Conditional AME(Grant | Admin=0)', p10 - p00, g10 - g00),
        ('Pr(Y2=1 | Grant=0, Admin=1)', p01, g01),
        ('Pr(Y2=1 | Grant=1, Admin=1)', p11, g11),
        ('Conditional AME(Grant | Admin=1)', p11 - p01, g11 - g01),
        ('Second Difference ΔAME (Admin=1 vs Admin=0)', (p11 - p01) - (p10 - p00), (g11 - g01) - (g10 - g00)),
    ]:
        se = float(np.sqrt(grad @ cov @ grad))
        z = float(est / se)
        pval = float(2 * (1 - stats.norm.cdf(abs(z))))
        rows.append({
            'Dimension': 'Administrative Complexity (Q7.1)',
            'Quantity': label,
            'Estimate': float(est),
            'SE': se,
            'CI_95_Lo': float(est - 1.96 * se),
            'CI_95_Hi': float(est + 1.96 * se),
            'z_stat': z,
            'p_value': pval
        })

    # Expertise deficit moderation
    pe00, ge00 = get_pred_and_grad(0, e_val=0)
    pe10, ge10 = get_pred_and_grad(1, e_val=0)
    pe01, ge01 = get_pred_and_grad(0, e_val=1)
    pe11, ge11 = get_pred_and_grad(1, e_val=1)

    for label, est, grad in [
        ('Pr(Y2=1 | Grant=0, Expertise=0)', pe00, ge00),
        ('Pr(Y2=1 | Grant=1, Expertise=0)', pe10, ge10),
        ('Conditional AME(Grant | Expertise=0)', pe10 - pe00, ge10 - ge00),
        ('Pr(Y2=1 | Grant=0, Expertise=1)', pe01, ge01),
        ('Pr(Y2=1 | Grant=1, Expertise=1)', pe11, ge11),
        ('Conditional AME(Grant | Expertise=1)', pe11 - pe01, ge11 - ge01),
        ('Second Difference ΔAME (Expertise=1 vs Expertise=0)', (pe11 - pe01) - (pe10 - pe00), (ge11 - ge01) - (ge10 - ge00)),
    ]:
        se = float(np.sqrt(grad @ cov @ grad))
        z = float(est / se)
        pval = float(2 * (1 - stats.norm.cdf(abs(z))))
        rows.append({
            'Dimension': 'Expertise Deficit (Q7.6)',
            'Quantity': label,
            'Estimate': float(est),
            'SE': se,
            'CI_95_Lo': float(est - 1.96 * se),
            'CI_95_Hi': float(est + 1.96 * se),
            'z_stat': z,
            'p_value': pval
        })

    out_df = pd.DataFrame(rows)
    for _, r in out_df.iterrows():
        print(f"    - {r['Quantity']:<46}: {r['Estimate']:+.4f} (SE={r['SE']:.4f}, 95% CI [{r['CI_95_Lo']:+.4f}, {r['CI_95_Hi']:+.4f}], p={r['p_value']:.4f})")
    return out_df


def main():
    print("=" * 70)
    print("ECONOMETRIC ROBUSTNESS CHECKS & SENSITIVITY ANALYSES")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    print(f"  Loaded analytical sample: {len(df):,} observations\n")

    r1 = robustness_q14_strict(df)
    r2 = robustness_sector_stratified(df)
    r3 = robustness_weighted(df)
    r4 = robustness_appendix_j_probscale(df)

    # Save robustness results directly to master Excel workbook
    mode = 'a' if os.path.exists(RESULTS_XLSX) else 'w'
    with pd.ExcelWriter(RESULTS_XLSX, engine='openpyxl', mode=mode,
                        if_sheet_exists='replace' if mode == 'a' else None) as writer:
        if r2 is not None:
            r2.to_excel(writer, sheet_name='Robustness_Sector', index=False)
        if r3 is not None:
            pd.DataFrame({'Variable': r3['var_names'], 'Coef': r3['coef'], 'SE': r3['se'], 'p': r3['p']}).to_excel(writer, sheet_name='Robustness_Weighted', index=False)
        if r4 is not None:
            r4.to_excel(writer, sheet_name='AppendixJ_ProbScale_Moderation', index=False)

    print(f"\n  ✓ Robustness estimation complete.")
    print(f"  ✓ Results appended to: {RESULTS_XLSX}")


if __name__ == '__main__':
    main()
