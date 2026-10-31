#!/usr/bin/env python3
"""
01_build_analytical_sample.py
=============================
Master data pipeline for the SME Institutional Decoupling manuscript.

Reads raw Flash Eurobarometer 498 (ZA7806) SPSS file, applies all
filters, recodes variables with CORRECT labels (verified against SPSS
metadata), merges Eurostat DESI macro data, and outputs:
  - Data/Processed/SME_Analytical_Final.csv
  - Data/Input_Data_Consolidated.xlsx  (multi-sheet)

CRITICAL FIXES applied in this script:
  1. DESI indicator codes: E_DI3_HI / E_DI3_VHI (not E_DI_HI)
  2. scr15 labels: scr15.1=B2C, scr15.2=B2B, scr15.3=B2G
  3. Q6 labels:   Q6.1=grants, Q6.4=advice
  4. EU-27 + SME-only sample filter
  5. Binary recoding of Q9, Q14
  6. Handling of Inap (9.0) codes as NaN

Author: Research Pipeline
Date: 2026-04-16
"""

import pandas as pd
import numpy as np
import pyreadstat
import re
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ──────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "Data")
if not os.path.exists(DATA_DIR) and os.path.exists(os.path.join(os.path.dirname(PROJECT_DIR), "Data")):
    DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "Data")
RAW_DIR = os.path.join(DATA_DIR, "Raw")

# 1. Search for SAV file in different potential locations
SAV_FILE = os.path.join(RAW_DIR, "FL498_2021_ZA7806.sav")
if not os.path.exists(SAV_FILE):
    SAV_FILE = os.path.join(RAW_DIR, "ZA7806_v1-0-0.sav")
if not os.path.exists(SAV_FILE):
    # Fallback to renamed FE 498 subfolder
    SAV_FILE = os.path.join(RAW_DIR, "Data_Flash_Eurobarometer_498_SMEs_resource_efficiency_and_green_markets", "FL498_2021_ZA7806.sav")
if not os.path.exists(SAV_FILE):
    # Legacy fallback to FE 342 subfolder
    SAV_FILE = os.path.join(RAW_DIR, "Data_Flash_Eurobarometer_342_SMEs_resource_efficiency_and_green_markets", "FL498_2021_ZA7806.sav")

# 2. Search for Eurostat TSV file
EUROSTAT_TSV = os.path.join(DATA_DIR, "Eurostat_Digital_Intensity.tsv")
if not os.path.exists(EUROSTAT_TSV):
    EUROSTAT_TSV = os.path.join(RAW_DIR, "Eurostat_Digital_Intensity.tsv")

OUTPUT_CSV = os.path.join(DATA_DIR, "Processed", "SME_Analytical_Final.csv")
OUTPUT_XLSX = os.path.join(DATA_DIR, "Input_Data_Consolidated.xlsx")

# EU-27 country codes (from SPSS metadata, verified)
EU27_COUNTRY_CODES = {
    1: 'FR', 2: 'BE', 3: 'NL', 4: 'DE', 5: 'IT', 6: 'LU',
    7: 'DK', 8: 'IE', 11: 'EL', 12: 'ES', 13: 'PT', 16: 'FI',
    17: 'SE', 18: 'AT', 19: 'CY', 20: 'CZ', 21: 'EE', 22: 'HU',
    23: 'LV', 24: 'LT', 25: 'MT', 26: 'PL', 27: 'SK', 28: 'SI',
    29: 'BG', 30: 'RO', 32: 'HR'
}
# Excluded: 9=UK, 31=TR, 34=MK, 35=ME, 36=RS, 37=AL, 39=MD, 41=NO, 43=IS, 61=US

# ─── CORRECTED Variable Mapping ────────────────────────────────────
# scr15 labels (VERIFIED against SPSS metadata):
#   scr15.1 = "Directly to consumers" (B2C)
#   scr15.2 = "To other companies" (B2B private)
#   scr15.3 = "To public administration" (B2G)
#
# Q6 labels (VERIFIED against SPSS metadata):
#   Q6.1 = "Public funding such as grants, guarantees or loans" (FINANCIAL)
#   Q6.4 = "Advice or other non-financial assistance from public admin" (NON-FINANCIAL)
#   Q6.5 = "Advice from private consulting and audit companies"

VARS_TO_EXTRACT = [
    # IDs and weights
    'uniqid', 'country', 'w1_sme', 'w93_sme',
    # Firm characteristics
    'scr10', 'scr12', 'scr12a', 'scr12b', 'scr13', 'scr14', 'nace_b',
    # Market orientation (scr15) - CORRECT LABELS
    'scr15.1', 'scr15.2', 'scr15.3', 'scr15.4',
    # Domestic/foreign sales
    'scr16',
    # Resource efficiency actions (Q1.1-Q1.9 for action count)
    'q1.1', 'q1.2', 'q1.3', 'q1.4', 'q1.5', 'q1.6',
    'q1.7', 'q1.8', 'q1.9', 'q1.10', 'q1.11', 'q1.12', 'q1t',
    # Planned actions
    'q2.3',
    # Production cost impact
    'q3',
    # Investment level
    'q4',
    # Support type
    'q5.1', 'q5.2', 'q5.3', 'q5.4', 'q5.5',
    # External support details (Q6) - CORRECT LABELS
    'q6.1', 'q6.2', 'q6.3', 'q6.4', 'q6.5', 'q6.6', 'q6.7',
    # Barriers (Q7)
    'q7.1', 'q7.2', 'q7.3', 'q7.4', 'q7.5', 'q7.6',
    'q7.7', 'q7.8', 'q7.9', 'q7.10', 'q7.11', 'q7.12',
    # Desired support
    'q8.1', 'q8.2', 'q8.3', 'q8.4', 'q8.5', 'q8.6',
    # DVs
    'q9',  # Green products (Market-facing outcome)
    'q10', # Green product share of annual turnover (if Q9=1)
    'q14', # Net-zero strategy (Internal strategic outcome)
    # Climate actions (only if Q14 in {1,4})
    'q15.1', 'q15.2', 'q15.3', 'q15.4', 'q15.5', 'q15.6',
    # Green jobs FTE count
    'dx5',
]


def clean_eurostat_value(val):
    """Parse Eurostat TSV cell values, handling flags and missing markers."""
    if pd.isna(val):
        return np.nan
    v = str(val).strip()
    if v == ':' or v == '':
        return np.nan
    v = re.sub(r'[a-zA-Z]', '', v).strip()
    try:
        return float(v)
    except ValueError:
        return np.nan


def load_and_filter_fl498():
    """
    Load FL498, apply EU-27 + SME filter, extract required variables.
    
    Conceptual Rationale for Filters:
      - EU-27: The study focuses on firms operating under the harmonized EU regulatory
        framework and single market. Non-EU countries (e.g., UK, Turkey) are excluded
        to prevent confounding from differing environmental/trade policy regimes.
      - SME-only: Small and Medium-sized Enterprises (1-249 employees) are structurally
        different from large multinationals. They face severe capability and resource
        constraints, making them the ideal testing ground for "capability traps" and
        institutional pressures. Large firms (250+ employees) are excluded.
    """
    print("=" * 70)
    print("PHASE 1: Loading Flash Eurobarometer 498 (ZA7806) SPSS dataset")
    print("=" * 70)

    df, meta = pyreadstat.read_sav(SAV_FILE)
    print(f"  [Data Loading] Raw file successfully loaded.")
    print(f"  - Total Observations (Firms): {len(df):,}")
    print(f"  - Total Variables: {len(df.columns)}")

    # Keep only variables that exist in the dataset
    valid_vars = [v for v in VARS_TO_EXTRACT if v in df.columns]
    missing_vars = [v for v in VARS_TO_EXTRACT if v not in df.columns]
    if missing_vars:
        print(f"  ⚠ Warning: Variables not found in dataset: {missing_vars}")
    df = df[valid_vars].copy()
    print(f"  - Retained {len(valid_vars)} relevant variables for analysis")

    # ── Filter 1: EU-27 countries only ──────────────────────────────
    # Restricts sample to member states operating under standard EU environmental policy guidelines
    eu27_codes = list(EU27_COUNTRY_CODES.keys())
    n_before = len(df)
    df = df[df['country'].isin(eu27_codes)].copy()
    print(f"  [Filter 1] Excluded non-EU countries: {n_before:,} → {len(df):,} firms (dropped {n_before - len(df):,} firms)")

    # ── Filter 2: SME-only (exclude 250+ and DK/NA) ────────────────
    # Focuses on micro (1-9), small (10-49), and medium (50-249) enterprises
    n_before = len(df)
    df = df[df['scr10'].isin([1.0, 2.0, 3.0])].copy()
    print(f"  [Filter 2] Excluded large firms (250+ emp) and missing sizes: {n_before:,} → {len(df):,} firms (dropped {n_before - len(df):,} firms)")

    # ── Add ISO2 country code ───────────────────────────────────────
    df['country_iso2'] = df['country'].map(EU27_COUNTRY_CODES)

    print(f"\n  ✓ Phase 1 Complete: Sample is locked at N = {len(df):,} across {df['country_iso2'].nunique()} countries")
    return df, meta


def recode_variables(df):
    """
    Apply variable recodings, handle inapplicable Flash Eurobarometer codes, and 
    construct the Decoupling Index.
    
    Survey Routing & Inapplicable Codes:
      - Flash Eurobarometer utilizes branching logic. If a firm does not pass a filter question
        (e.g., they didn't implement any resource efficiency action), the sub-questions
        associated with it are marked as 9.0 ("Inapplicable") in the SPSS raw data.
      - If left as 9.0, standard numerical summaries or regressions would interpret this
        as a high numeric value. We map all 9.0 values in sub-question blocks (q3, q4, q5,
        q6, q7, q8) to np.nan (Missing) to prevent severe statistical bias.

    Dependent Variables (DVs) Setup:
      1. Symbolic DV (q9_binary): Offers green products/services. Maps 1.0 (Yes) -> 1,
         and 2.0/3.0 (Planning/No) -> 0. "Don't Know" (4.0) is treated as NaN.
      2. Substantive DV (q14_binary): Concrete carbon footprint reduction strategy.
         We treat Yes (1.0) and Already Carbon Neutral (4.0) as 1 (active operational
         commitment), and Planning (2.0) and No (3.0) as 0. "Don't Know" (5.0) -> NaN.
      
    Decoupling Index (DI) Construction:
      - Formulated as: DI = Symbolic (q9_binary) - Substantive (q14_binary).
      - Range: [-1, 0, +1].
        - +1 (Symbolic Excess / "Greenwashing"): Firm offers green products/services
          but does NOT have a net-zero operational carbon strategy.
        -  0 (Aligned): Firm is either consistently active (both 1) or disengaged (both 0).
        - -1 (Substantive Excess / "Quiet Green" / "Greenhushing"): Firm acts operationally
          but does not market green products/services.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Variable Recoding, Routing Fixes & Feature Engineering")
    print("=" * 70)

    # ── Handle Inapplicable codes (9.0) → NaN ──────────────────────
    inap_vars = [c for c in df.columns if c.startswith(('q3', 'q4', 'q5.', 'q6.', 'q7.', 'q8.', 'q10', 'q15.'))]
    total_repcodes = 0
    for v in inap_vars:
        if v in df.columns:
            n_inap = (df[v] == 9.0).sum()
            if n_inap > 0:
                df.loc[df[v] == 9.0, v] = np.nan
                total_repcodes += n_inap
    print(f"  [Routing Fix] Converted {total_repcodes:,} inapplicable (9.0) codes to NaN across {len(inap_vars)} variables.")

    # ── DV 1: Q9 (Market-Facing Green Offerings) - Binary recode ─────
    df['q9_binary'] = np.where(df['q9'] == 1.0, 1, np.where(df['q9'].isin([2.0, 3.0]), 0, np.nan))
    n_sym = df['q9_binary'].sum()
    print(f"  [DV Recode] Q9 (Market-Facing - Green Products/Services):")
    print(f"    - Yes (active offering) -> q9_binary=1 (n={int(n_sym):,})")
    print(f"    - No / Planning -> q9_binary=0")

    # ── DV 2: Q14 (Reported Net-Zero Strategy) - Binary recode ───────
    df['q14_binary'] = np.where(df['q14'].isin([1.0, 4.0]), 1,
                        np.where(df['q14'].isin([2.0, 3.0]), 0, np.nan))
    n_sub = df['q14_binary'].sum()
    print(f"  [DV Recode] Q14 (Internal Strategic - Net-Zero Strategy):")
    print(f"    - Yes / Already Neutral -> q14_binary=1 (n={int(n_sub):,})")
    print(f"    - No / Planning -> q14_binary=0")

    # Robustness Strategy: Strict recode that excludes "already neutral" (4.0) firms,
    # targeting only active 2030 transition planners (1.0).
    df['q14_strict'] = np.where(df['q14'] == 1.0, 1,
                       np.where(df['q14'].isin([2.0, 3.0]), 0, np.nan))

    # ── Cross-Domain Sustainability Asymmetry Index (DI) & 4-Quadrant Typology ──
    mask_both = df['q9_binary'].notna() & df['q14_binary'].notna()
    df['DI'] = np.nan
    df.loc[mask_both, 'DI'] = df.loc[mask_both, 'q9_binary'] - df.loc[mask_both, 'q14_binary']
    di_dist = df['DI'].value_counts().sort_index()
    print(f"  [DI Construction] Directional Sustainability Asymmetry Index (Y1 - Y2) Distribution:")
    print(f"    - DI = -1 (Net-Zero Strategy Only): {di_dist.get(-1.0, 0):,} firms")
    print(f"    - DI =  0 (Zero Net Asymmetry):     {di_dist.get(0.0, 0):,} firms")
    print(f"    - DI = +1 (Green Offerings Only):   {di_dist.get(1.0, 0):,} firms")

    # Four-Quadrant categorical indicator (0=Neither [0,0], 1=Green Only [1,0], 2=Net-Zero Only [0,1], 3=Both [1,1])
    df['quadrant'] = np.nan
    df.loc[mask_both & (df['q9_binary'] == 0) & (df['q14_binary'] == 0), 'quadrant'] = 0
    df.loc[mask_both & (df['q9_binary'] == 1) & (df['q14_binary'] == 0), 'quadrant'] = 1
    df.loc[mask_both & (df['q9_binary'] == 0) & (df['q14_binary'] == 1), 'quadrant'] = 2
    df.loc[mask_both & (df['q9_binary'] == 1) & (df['q14_binary'] == 1), 'quadrant'] = 3

    # ── Market Orientation (scr15) - Human-Friendly Semantics ──────────
    df['B2C'] = df['scr15.1']
    df['B2B_private'] = df['scr15.2']
    df['B2G_public'] = df['scr15.3']
    print("  [Semantic Rename] Market channels renamed: scr15.1→B2C, scr15.2→B2B_private, scr15.3→B2G_public")

    # ── External Support Channels (Q6) - Human-Friendly Semantics ───────
    # Fill bypassed survey paths with 0 (since they didn't receive the support)
    for q6_var in ['q6.1', 'q6.2', 'q6.3', 'q6.4', 'q6.5', 'q6.6', 'q6.7']:
        if q6_var in df.columns:
            df[q6_var] = df[q6_var].fillna(0)  # NaN (Inap) → 0 (support not received)

    df['grant_receipt'] = df['q6.1']   # Public financial support: grants, guarantees, or subsidized loans
    df['public_financial_support'] = df['q6.1']
    df['advice_public'] = df['q6.4']   # Non-financial advice from public administrations
    df['advice_private'] = df.get('q6.5', 0) # Non-financial advice from private consultancies
    df['private_finance'] = df.get('q6.2', 0) # Private bank/VC finance
    df['supply_chain_advice'] = df.get('q6.7', 0) # Advice from supply chain partners
    print("  [Semantic Rename] Q6 support channels renamed: q6.1→grant_receipt/public_financial_support, q6.4→advice_public")

    # ── Internal Capability Barriers (Q7) & Additional Variables (Point 9) ──
    df['barrier_admin'] = df.get('q7.1', np.nan)
    df['barrier_finance'] = df.get('q7.2', np.nan)
    df['barrier_technical'] = df.get('q7.3', np.nan)
    df['barrier_cost'] = df.get('q7.5', np.nan)
    df['barrier_expertise'] = df.get('q7.6', np.nan)
    df['barrier_label'] = df.get('q7.9', np.nan)
    df['own_finance'] = df.get('q5.1', np.nan)
    df['own_expertise'] = df.get('q5.2', np.nan)
    df['sells_products'] = np.where(df['scr16'].isin([1.0, 3.0]), 1.0,
                           np.where(df['scr16'] == 2.0, 0.0, np.nan))
    df['q4_clean'] = np.where(df['q4'].isin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), df['q4'], np.nan)
    if 'q10' in df.columns:
        df['q10_clean'] = np.where(df['q10'].isin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), df['q10'], np.nan)
    if 'dx5' in df.columns:
        df['green_jobs_any'] = np.where(df['dx5'] == 999997.0, 0.0,
                               np.where((df['dx5'] >= 0) & (df['dx5'] < 999990), (df['dx5'] > 0).astype(float), np.nan))
        df['green_jobs_fte'] = np.where(df['dx5'] == 999997.0, 0.0,
                               np.where((df['dx5'] >= 0) & (df['dx5'] < 999990), df['dx5'].astype(float), np.nan))
    q15_items = [f'q15.{i}' for i in range(1, 6) if f'q15.{i}' in df.columns]
    if q15_items:
        df['q15_count'] = np.where(df[q15_items[0]].notna(), (df[q15_items] == 1.0).sum(axis=1).astype(float), np.nan)

    # ── Derived Feature: Resource Efficiency Action Count ───────────────
    # 1) Strict 9-item count (q1.1 to q1.9, excluding q1.10 Other, q1.11 None, q1.12 DK):
    #    Equals 0 for all 1,231 unrouted firms (matching the Q3-Q8 survey routing rule to the single firm).
    q1_9_vars = [f'q1.{i}' for i in range(1, 10) if f'q1.{i}' in df.columns]
    df['re_action_count_9'] = (df[q1_9_vars] == 1.0).sum(axis=1)
    # 2) Baseline re_action_count (preserves routed sample values while ensuring None/DK are not counted as actions)
    q1_all_vars = [c for c in df.columns if re.match(r'^q1\.\d+$', c)]
    df['re_action_count'] = (df[q1_all_vars] == 1.0).sum(axis=1)
    print(f"  [Feature Generation] Resource Efficiency Action Count (9-item Q1.1-Q1.9): mean={df['re_action_count_9'].mean():.2f}, zero-action unrouted firms={(df['re_action_count_9']==0).sum():,}")

    # ── Derived Feature: Total Barrier Count (Q7.1-Q7.9, excluding Other/None/DK) ──
    q7_9_vars = [f'q7.{i}' for i in range(1, 10) if f'q7.{i}' in df.columns]
    df['barrier_count'] = (df[q7_9_vars] == 1.0).sum(axis=1)

    # ── Derived Feature: Firm Age ───────────────────────────────────
    if 'scr12a' in df.columns:
        df['firm_age'] = 2021 - df['scr12a']
        df.loc[df['firm_age'] < 0, 'firm_age'] = np.nan
        df.loc[df['firm_age'] > 200, 'firm_age'] = np.nan
        print(f"  [Feature Generation] Firm age: mean={df['firm_age'].mean():.1f} years, median={df['firm_age'].median():.0f} years")
    elif 'scr12' in df.columns:
        age_map = {1.0: 40, 2.0: 26, 3.0: 16, 4.0: 8, 5.0: 3}
        df['firm_age'] = df['scr12'].map(age_map)
        print(f"  [Feature Generation] Firm age (approximated): mean={df['firm_age'].mean():.1f} years")

    # ── Firm size category labels ───────────────────────────────────
    size_map = {1.0: 'Micro (1-9)', 2.0: 'Small (10-49)', 3.0: 'Medium (50-249)'}
    df['firm_size_label'] = df['scr10'].map(size_map)

    return df


def merge_desi(df):
    """Merge Eurostat DESI data using CORRECT indicator codes."""
    print("\n" + "=" * 70)
    print("PHASE 3: Merging Eurostat DESI (E_DI3_HI, E_DI3_VHI for 2021)")
    print("=" * 70)

    df_euro = pd.read_csv(EUROSTAT_TSV, sep='\t')
    first_col = df_euro.columns[0]
    dims = df_euro[first_col].str.split(',', expand=True)
    geo_col_idx = len(dims.columns) - 1
    dims.iloc[:, geo_col_idx] = dims.iloc[:, geo_col_idx].str.split(r'\\').str[0]

    df_euro['size_emp'] = dims[1]
    df_euro['indic_is'] = dims[3]
    df_euro['geo'] = dims[geo_col_idx]

    year_col = [c for c in df_euro.columns if '2021' in c][0]

    # ── CORRECTED: Use E_DI3_HI and E_DI3_VHI ──────────────────────
    eu27_isos = list(EU27_COUNTRY_CODES.values())

    # High digital intensity
    mask_hi = ((df_euro['size_emp'] == '10-249') &
               (df_euro['indic_is'] == 'E_DI3_HI') &
               (df_euro['geo'].isin(eu27_isos)))
    desi_hi = df_euro.loc[mask_hi, ['geo', year_col]].copy()
    desi_hi['DESI_High'] = desi_hi[year_col].apply(clean_eurostat_value)
    desi_hi = desi_hi[['geo', 'DESI_High']].dropna()

    # Very high digital intensity
    mask_vhi = ((df_euro['size_emp'] == '10-249') &
                (df_euro['indic_is'] == 'E_DI3_VHI') &
                (df_euro['geo'].isin(eu27_isos)))
    desi_vhi = df_euro.loc[mask_vhi, ['geo', year_col]].copy()
    desi_vhi['DESI_VeryHigh'] = desi_vhi[year_col].apply(clean_eurostat_value)
    desi_vhi = desi_vhi[['geo', 'DESI_VeryHigh']].dropna()

    # Merge high + very high
    desi = desi_hi.merge(desi_vhi, on='geo', how='outer')
    desi['DII_composite'] = desi['DESI_High'].fillna(0) + desi['DESI_VeryHigh'].fillna(0)

    print(f"  DESI data: {len(desi)} countries with valid data")
    print(f"  Countries: {sorted(desi['geo'].tolist())}")

    # Check coverage
    missing_countries = set(eu27_isos) - set(desi['geo'].tolist())
    if missing_countries:
        print(f"  ⚠ Missing DESI data for: {missing_countries}")
    else:
        print(f"  ✓ Full 27/27 EU country coverage")

    # Merge onto firm-level data
    n_before = df['country_iso2'].notna().sum()
    df = df.merge(desi, left_on='country_iso2', right_on='geo', how='left')
    df.drop(columns=['geo'], inplace=True, errors='ignore')

    n_desi = df['DII_composite'].notna().sum()
    print(f"  Merge result: {n_desi:,}/{len(df):,} firms have DESI data ({n_desi/len(df)*100:.1f}%)")

    # Print DESI summary
    print(f"\n  DESI_High:     mean={df['DESI_High'].mean():.2f}, sd={df['DESI_High'].std():.2f}")
    print(f"  DESI_VeryHigh: mean={df['DESI_VeryHigh'].mean():.2f}, sd={df['DESI_VeryHigh'].std():.2f}")
    print(f"  DII_composite: mean={df['DII_composite'].mean():.2f}, sd={df['DII_composite'].std():.2f}")

    return df


def compute_descriptive_stats(df):
    """Generate descriptive statistics table."""
    print("\n" + "=" * 70)
    print("PHASE 4: Descriptive Statistics")
    print("=" * 70)

    stats = {}

    # Sample composition
    print(f"\n  Sample: N = {len(df):,}")
    print(f"  Countries: {df['country_iso2'].nunique()}")

    # Firm size
    size_dist = df['firm_size_label'].value_counts()
    print(f"\n  Firm Size Distribution:")
    for k, v in size_dist.items():
        print(f"    {k}: {v:,} ({v/len(df)*100:.1f}%)")

    # DV distributions
    print(f"\n  Q9 (Symbolic - green products):")
    q9_dist = df['q9_binary'].value_counts(dropna=False).sort_index()
    for k, v in q9_dist.items():
        print(f"    {k}: {v:,}")

    print(f"\n  Q14 (Substantive - net-zero strategy):")
    q14_dist = df['q14_binary'].value_counts(dropna=False).sort_index()
    for k, v in q14_dist.items():
        print(f"    {k}: {v:,}")

    print(f"\n  Decoupling Index:")
    di_dist = df['DI'].value_counts(dropna=False).sort_index()
    for k, v in di_dist.items():
        label = {-1: 'Substantive only', 0: 'Aligned', 1: 'Greenwashing (sym only)'}.get(k, 'Missing')
        print(f"    DI={k} ({label}): {v:,}")

    # IV distributions
    print(f"\n  Market Orientation:")
    print(f"    B2C (scr15.1): {(df['B2C']==1).sum():,} ({(df['B2C']==1).mean()*100:.1f}%)")
    print(f"    B2B (scr15.2): {(df['B2B_private']==1).sum():,} ({(df['B2B_private']==1).mean()*100:.1f}%)")
    print(f"    B2G (scr15.3): {(df['B2G_public']==1).sum():,} ({(df['B2G_public']==1).mean()*100:.1f}%)")

    print(f"\n  External Support (corrected labels):")
    print(f"    Grant receipt (Q6.1): {(df['grant_receipt']==1).sum():,} ({(df['grant_receipt']==1).mean()*100:.1f}%)")
    print(f"    Advice public (Q6.4): {(df['advice_public']==1).sum():,} ({(df['advice_public']==1).mean()*100:.1f}%)")

    print(f"\n  Barriers (among RE-active firms):")
    for bname, bvar in [('Admin complexity', 'barrier_admin'),
                        ('Financial', 'barrier_finance'),
                        ('Expertise deficit', 'barrier_expertise')]:
        valid = df[bvar].dropna()
        pct = (valid == 1).mean() * 100 if len(valid) > 0 else 0
        print(f"    {bname}: {(valid==1).sum():,} of {len(valid):,} ({pct:.1f}%)")

    # Country-level DI
    print(f"\n  Country-Level Decoupling (Top 5 highest / lowest):")
    country_di = df.groupby('country_iso2')['DI'].mean().dropna().sort_values()
    print(f"    LOWEST (most substantive):")
    for iso, val in country_di.head(5).items():
        print(f"      {iso}: {val:.3f}")
    print(f"    HIGHEST (most greenwashing):")
    for iso, val in country_di.tail(5).items():
        print(f"      {iso}: {val:.3f}")

    return df


def save_outputs(df):
    """Save analytical CSV and multi-sheet Input Excel workbook."""
    print("\n" + "=" * 70)
    print("PHASE 5: Saving Outputs")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ Analytical CSV: {OUTPUT_CSV}")
    print(f"    Shape: {df.shape}")

    # ── Multi-sheet Input Excel Workbook ────────────────────────────
    with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:

        # Sheet 1: Full analytical dataset
        df.to_excel(writer, sheet_name='Analytical_Sample', index=False)

        # Sheet 2: Country-level DESI data
        desi_cols = ['country_iso2', 'DESI_High', 'DESI_VeryHigh', 'DII_composite']
        desi_sheet = df[desi_cols].drop_duplicates().dropna(subset=['DESI_High'])
        desi_sheet = desi_sheet.sort_values('country_iso2')
        desi_sheet.to_excel(writer, sheet_name='DESI_Country_Data', index=False)

        # Sheet 3: Variable codebook
        codebook_data = {
            'Variable': ['q9_binary', 'q14_binary', 'DI', 'B2C', 'B2B_private', 'B2G_public',
                         'grant_receipt', 'advice_public', 'barrier_admin', 'barrier_expertise',
                         'scr10', 'scr14', 'firm_age', 're_action_count', 're_action_count_9', 'DII_composite'],
            'Source': ['Q9', 'Q14', 'Q9-Q14', 'scr15.1', 'scr15.2', 'scr15.3',
                       'Q6.1', 'Q6.4', 'Q7.1', 'Q7.6',
                       'scr10', 'scr14', 'scr12a', 'Q1', 'Q1.1-Q1.9', 'Eurostat'],
            'Description': [
                'Market-facing green offerings (1=Offers green products/services)',
                'Reported net-zero strategy (1=Yes concrete 2030 strategy or Already neutral)',
                'Directional Cross-Domain Sustainability Asymmetry Index (Y1 - Y2)',
                'Sells directly to consumers (B2C)',
                'Sells to other companies (B2B)',
                'Sells to public administration (B2G)',
                'Public financial support (grants, guarantees, loans)',
                'Public non-financial advice from public administration',
                'Barrier: complexity of administrative or legal procedures',
                'Barrier: lack of specific environmental expertise',
                'Firm size (1=micro, 2=small, 3=medium)',
                'Annual turnover category',
                'Years since establishment',
                'Baseline resource-efficiency action count',
                'Strict 9-item resource-efficiency action count (Q1.1-Q1.9, 0-9)',
                'National Digital Intensity Index (DESI High + Very High %)'
            ],
            'Type': ['Binary', 'Binary', 'Ternary(-1,0,+1)', 'Binary', 'Binary', 'Binary',
                     'Binary', 'Binary', 'Binary', 'Binary',
                     'Ordinal', 'Ordinal', 'Continuous', 'Count', 'Count', 'Continuous'],
            'Hypothesis': ['Y1(H1a,H1b,H2a,H2b,H3b)', 'Y2(H1a,H1b,H1c,H2a,H2b,H3a,H3b,H3c)', 'DV(Model 4)',
                          'X(H1a)', 'X(H1c)', 'X(H1b)',
                          'Treatment(H3a,H3b,H3c)', 'Control', 'X(H2a,H3c)', 'X(H2b)',
                          'Control', 'Control', 'Control', 'Control', 'Control(Robustness)', 'Moderator(Model 4)']
        }
        pd.DataFrame(codebook_data).to_excel(writer, sheet_name='Codebook', index=False)

        # Sheet 4: Country metadata
        country_meta = df.groupby('country_iso2').agg(
            N=('uniqid', 'count'),
            Pct_Green_Offerings=('q9_binary', 'mean'),
            Pct_NetZero_Strategy=('q14_binary', 'mean'),
            Mean_DI=('DI', 'mean'),
            Pct_B2B=('B2B_private', 'mean'),
            Pct_Public_Support=('grant_receipt', 'mean'),
            DII=('DII_composite', 'first')
        ).round(4)
        country_meta.to_excel(writer, sheet_name='Country_Summary')

        # Sheet 5: Descriptive statistics
        desc_vars = ['q9_binary', 'q14_binary', 'DI', 'B2C', 'B2B_private', 'B2G_public',
                     'grant_receipt', 'advice_public', 'barrier_admin', 'barrier_expertise',
                     'scr10', 'scr14', 're_action_count', 're_action_count_9', 'DII_composite']
        desc_vars = [v for v in desc_vars if v in df.columns]
        desc_stats = df[desc_vars].describe().T
        desc_stats['N_valid'] = df[desc_vars].notna().sum()
        desc_stats['N_missing'] = df[desc_vars].isna().sum()
        desc_stats['Pct_missing'] = (desc_stats['N_missing'] / len(df) * 100).round(1)
        desc_stats.to_excel(writer, sheet_name='Descriptive_Statistics')

        # Sheet 6: Sample Waterfall Reconciliation
        n_typology = int((df['q9_binary'].notna() & df['q14_binary'].notna()).sum())
        n_routed = int(df['barrier_admin'].notna().sum())
        n_ipw = int((df['barrier_admin'].notna() & df['q14_binary'].notna()).sum())
        n_joint = int((df['barrier_admin'].notna() & df['q9_binary'].notna() & df['q14_binary'].notna()).sum())
        n_strict = int((df['barrier_admin'].notna() & df['q9_binary'].notna() & df['q14_strict'].notna()).sum())
        n_cost = int((df['barrier_admin'].notna() & df['q9_binary'].notna() & df['q14_binary'].notna() &
                      df['q3'].isin([1, 2, 3, 4, 5]) & df['q4'].notna()).sum())
        n_q53_routed = int((df['q5.3'] == 1.0).sum())
        n_q53_y2 = int(((df['q5.3'] == 1.0) & df['q14_binary'].notna()).sum())
        n_q53_joint = int(((df['q5.3'] == 1.0) & df['q9_binary'].notna() & df['q14_binary'].notna()).sum())

        waterfall_df = pd.DataFrame([
            {'Step': '0. Raw GESIS ZA7806 Dataset', 'Sample_N': 17662,
             'Filter_Rule': 'All enterprise interviews across 36 countries in Flash Eurobarometer 498'},
            {'Step': '1. EU-27 Member States', 'Sample_N': 14215,
             'Filter_Rule': 'Excludes 3,447 non-EU enterprises'},
            {'Step': '2. Baseline EU-27 SME Population', 'Sample_N': len(df),
             'Filter_Rule': 'Retains active SMEs with 1-249 employees (scr10 in {1,2,3}); drops 893 large/missing firms'},
            {'Step': '3. Four-Quadrant Typology & Model 4 Sample', 'Sample_N': n_typology,
             'Filter_Rule': 'Excludes 1,117 SMEs with DK on Q9 (593) or Q14 (676; 152 missing on both)'},
            {'Step': '4. Resource-Efficiency Routed Sample (Q3-Q8)', 'Sample_N': n_routed,
             'Filter_Rule': 'Excludes 1,231 SMEs with zero actions in Q1.1-Q1.9 (re_action_count_9 == 0) that bypassed Q3-Q8'},
            {'Step': '5. Single-Outcome IPW & CATE Sample (Q14 valid)', 'Sample_N': n_ipw,
             'Filter_Rule': 'Excludes 570 routed SMEs with DK on Q14 (1,292 treated, 10,229 control)'},
            {'Step': '6. Primary Joint Econometric Sample (Q9 & Q14 valid)', 'Sample_N': n_joint,
             'Filter_Rule': 'Excludes 393 routed SMEs with DK on Q9 from 11,521 (1,256 treated, 9,872 control)'},
            {'Step': '7. Strict 2030 Transition Subsample', 'Sample_N': n_strict,
             'Filter_Rule': 'Excludes 504 SMEs reporting Already climate neutral (Q14 = 4) from N = 11,128'},
            {'Step': '8. Operating Cost Ordered Probit Subsample (Model 5)', 'Sample_N': n_cost,
             'Filter_Rule': 'Routed SMEs in N = 11,128 with valid Q3 (in {1..5}) and Q4'},
            {'Step': '9a. External Support Seekers Subsample (Q5.3=1, valid Q14)', 'Sample_N': n_q53_y2,
             'Filter_Rule': f'Subsample of {n_q53_routed:,} Q5.3=1 support-seeking SMEs with valid Q14 (1,292 treated, 1,763 control)'},
            {'Step': '9b. External Support Seekers Subsample (Q5.3=1, valid Q9 & Q14)', 'Sample_N': n_q53_joint,
             'Filter_Rule': f'Subsample of {n_q53_routed:,} Q5.3=1 support-seeking SMEs in N=11,128 (1,256 treated, 1,703 control)'},
        ])
        waterfall_df.to_excel(writer, sheet_name='Sample_Waterfall', index=False)

        # Sheet 7: Table 1 Typology & Within-Survey Construct Validation (N = 12,205)
        d_typ = df.dropna(subset=['q9_binary', 'q14_binary']).copy()
        d_typ['quad_4'] = np.select(
            [(d_typ['q9_binary'] == 0) & (d_typ['q14_binary'] == 0),
             (d_typ['q9_binary'] == 1) & (d_typ['q14_binary'] == 0),
             (d_typ['q9_binary'] == 0) & (d_typ['q14_binary'] == 1),
             (d_typ['q9_binary'] == 1) & (d_typ['q14_binary'] == 1)],
            [0, 1, 2, 3]
        )
        t1_rows = []
        for q_id, q_label in [(0, '0. Disengaged / Neither (0,0)'),
                              (1, '1. Green Offerings Only (1,0)'),
                              (2, '2. Net-Zero Strategy Only (0,1)'),
                              (3, '3. Dual Adopters / Both (1,1)'),
                              ('All', 'Full Typology Sample')]:
            sub_q = d_typ if q_id == 'All' else d_typ[d_typ['quad_4'] == q_id]
            t1_rows.append({
                'Configuration': q_label,
                'N': len(sub_q),
                'Share_Pct': round(100.0 * len(sub_q) / len(d_typ), 2),
                'Mean_DI': round(float(sub_q['DI'].mean()), 4),
                'Mean_RE_Actions': round(float(sub_q['re_action_count'].mean()), 2),
                'Mean_RE_Actions_9Item': round(float(sub_q['re_action_count_9'].mean()), 2),
                'Mean_Firm_Size_1_3': round(float(sub_q['scr10'].mean()), 2),
                'Public_Support_Pct': round(100.0 * float((sub_q['grant_receipt'] == 1.0).mean()), 2),
                'Mean_National_DII': round(float(sub_q['DII_composite'].mean()), 2),
                'Sells_Products_SCR16_Pct': round(100.0 * float(sub_q['sells_products'].dropna().mean()), 2),
                'Green_Jobs_Any_DX5_Pct': round(100.0 * float(sub_q['green_jobs_any'].dropna().mean()), 2),
                'RE_Invest_GT5Pct_Q4_Valid_Pct': round(100.0 * float((sub_q['q4_clean'].dropna() >= 4).mean()), 2),
                'Q10_Green_Rev_GT30Pct_Pct': round(100.0 * float((sub_q['q10_clean'].dropna() >= 4).mean()), 2) if sub_q['q10_clean'].notna().any() else np.nan,
                'Q10_Green_Rev_GT5Pct_Pct': round(100.0 * float((sub_q['q10_clean'].dropna() >= 2).mean()), 2) if sub_q['q10_clean'].notna().any() else np.nan,
                'Q15_Carbon_Action_Mean': round(float(sub_q['q15_count'].dropna().mean()), 2) if sub_q['q15_count'].notna().any() else np.nan
            })
        pd.DataFrame(t1_rows).to_excel(writer, sheet_name='Table1_Typology_12205', index=False)

        # Sheet 8: Table G2 Missing-Data & Routing Attrition Comparison
        routed_sub = df[df['barrier_admin'].notna()]
        joint_sub = routed_sub[routed_sub['q9_binary'].notna() & routed_sub['q14_binary'].notna()]
        unrouted_sub = df[df['barrier_admin'].isna()]
        routed_miss_sub = routed_sub[routed_sub['q9_binary'].isna() | routed_sub['q14_binary'].isna()]
        attr_rows = []
        for c_name, c_sub in [('Full EU-27 SME Population', df),
                              ('Primary Joint Sample (Routed, Valid Q9 & Q14)', joint_sub),
                              ('Unrouted Zero-Action SMEs (re_action_count_9 == 0)', unrouted_sub),
                              ('Routed SMEs Missing Q9 or Q14 (DK)', routed_miss_sub)]:
            attr_rows.append({
                'Subsample_Cohort': c_name,
                'N': len(c_sub),
                'Micro_1_9_Pct': round(100.0 * float((c_sub['scr10'] == 1).mean()), 2),
                'Small_10_49_Pct': round(100.0 * float((c_sub['scr10'] == 2).mean()), 2),
                'Medium_50_249_Pct': round(100.0 * float((c_sub['scr10'] == 3).mean()), 2),
                'B2C_Pct': round(100.0 * float(c_sub['B2C'].mean()), 2),
                'B2B_Pct': round(100.0 * float(c_sub['B2B_private'].mean()), 2),
                'B2G_Pct': round(100.0 * float(c_sub['B2G_public'].mean()), 2),
                'Green_Offerings_Y1_Valid_Pct': round(100.0 * float(c_sub['q9_binary'].dropna().mean()), 2),
                'NetZero_Strategy_Y2_Valid_Pct': round(100.0 * float(c_sub['q14_binary'].dropna().mean()), 2),
                'Mean_9Item_RE_Count': round(float(c_sub['re_action_count_9'].mean()), 2)
            })
        pd.DataFrame(attr_rows).to_excel(writer, sheet_name='Missing_Data_Attrition', index=False)

    print(f"  ✓ Input Excel Workbook: {OUTPUT_XLSX}")
    print(f"    Sheets: Analytical_Sample, DESI_Country_Data, Codebook, Country_Summary, Descriptive_Statistics, Sample_Waterfall, Table1_Typology_12205, Missing_Data_Attrition")


def main():
    print("╔" + "═" * 68 + "╗")
    print("║  SME Institutional Decoupling - Master Data Pipeline             ║")
    print("║  Corrected: 16 April 2026 (6 critical bug fixes applied)        ║")
    print("╚" + "═" * 68 + "╝\n")

    # Step 1: Load and filter
    df, meta = load_and_filter_fl498()

    # Step 2: Recode variables
    df = recode_variables(df)

    # Step 3: Merge DESI
    df = merge_desi(df)

    # Step 4: Descriptive statistics
    df = compute_descriptive_stats(df)

    # Step 5: Save outputs
    save_outputs(df)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Final N: {len(df):,}")
    print(f"  Variables: {len(df.columns)}")
    print(f"  Output CSV: {OUTPUT_CSV}")
    print(f"  Output XLSX: {OUTPUT_XLSX}")


if __name__ == '__main__':
    main()
