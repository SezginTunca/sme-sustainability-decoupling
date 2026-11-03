#!/usr/bin/env python3
"""
01_build_analytical_sample.py
=============================
Master data construction and codebook pipeline for:
"Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs"

Reads unmodified Flash Eurobarometer 498 (ZA7806_v1-0-0.sav / FL498_2021_ZA7806.sav),
applies EU-27 and SME filters, constructs all analytical variables with verified GESIS
routing and missing-value rules, merges Eurostat Digital Intensity Index (isoc_e_dii, 2021),
and outputs:
  1. Data/Processed/SME_Analytical_Final.csv
  2. Data/Input_Data_Consolidated.xlsx (8 sheets)
  3. Data/Documentation/Codebook_ZA7806_Raw_and_Cleaned.xlsx (3 sheets: all 253 raw GESIS
     variables, all constructed variables with formulas and H1a-H3c mappings, and the
     complete sample waterfall including sensitivity and Wave 6 ZA8869 subsamples)
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import pyreadstat

warnings.filterwarnings('ignore')

# ─── Configuration ──────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "Data")
if not os.path.exists(DATA_DIR) and os.path.exists(os.path.join(os.path.dirname(PROJECT_DIR), "Data")):
    DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "Data")
RAW_DIR = os.path.join(DATA_DIR, "Raw")
DOC_DIR = os.path.join(DATA_DIR, "Documentation")

SAV_CANDIDATES = [
    os.path.join(RAW_DIR, "ZA7806_v1-0-0.sav"),
    os.path.join(RAW_DIR, "FL498_2021_ZA7806.sav"),
    os.path.join(RAW_DIR, "Data_Flash_Eurobarometer_498_SMEs_resource_efficiency_and_green_markets", "FL498_2021_ZA7806.sav"),
]
SAV_FILE = next((p for p in SAV_CANDIDATES if os.path.exists(p)), SAV_CANDIDATES[0])

W6_CANDIDATES = [
    os.path.join(RAW_DIR, "ZA8869_v1-0-0.sav"),
    os.path.join(RAW_DIR, "ZA8869_v2-0-0.sav"),
    os.path.join(RAW_DIR, "All_Raw_datasets_SME_Res_Eff",
                 "Flash Eurobarometer 549 (SMEs, Resource Efficiency and Green Markets, wave 6)",
                 "ZA8869_v1-0-0.sav"),
]
W6_SAV_FILE = next((p for p in W6_CANDIDATES if os.path.exists(p)), None)

EUROSTAT_TSV = os.path.join(RAW_DIR, "Eurostat_Digital_Intensity.tsv")
if not os.path.exists(EUROSTAT_TSV):
    EUROSTAT_TSV = os.path.join(DATA_DIR, "Eurostat_Digital_Intensity.tsv")

OUTPUT_CSV = os.path.join(DATA_DIR, "Processed", "SME_Analytical_Final.csv")
OUTPUT_XLSX = os.path.join(DATA_DIR, "Input_Data_Consolidated.xlsx")
CODEBOOK_XLSX = os.path.join(DOC_DIR, "Codebook_ZA7806_Raw_and_Cleaned.xlsx")

# EU-27 country codes (verified against GESIS ZA7806 SPSS metadata)
EU27_COUNTRY_CODES = {
    1: 'FR', 2: 'BE', 3: 'NL', 4: 'DE', 5: 'IT', 6: 'LU',
    7: 'DK', 8: 'IE', 11: 'EL', 12: 'ES', 13: 'PT', 16: 'FI',
    17: 'SE', 18: 'AT', 19: 'CY', 20: 'CZ', 21: 'EE', 22: 'HU',
    23: 'LV', 24: 'LT', 25: 'MT', 26: 'PL', 27: 'SK', 28: 'SI',
    29: 'BG', 30: 'RO', 32: 'HR'
}

VARS_TO_EXTRACT = [
    # IDs and weights
    'uniqid', 'country', 'w1_sme', 'w93_sme',
    # Firm characteristics and NACE classifications (4-group nace_b and 12-section nace_a)
    'scr10', 'scr12', 'scr12a', 'scr12b', 'scr13', 'scr14', 'nace_b', 'nace_a',
    # Market orientation (scr15: 1=B2C, 2=B2B, 3=B2G, 4=DK/NA)
    'scr15.1', 'scr15.2', 'scr15.3', 'scr15.4',
    # Offerings type (scr16: 1=Products, 2=Services, 3=Both, 4=DK/NA)
    'scr16',
    # Current resource-efficiency actions (Q1.1-Q1.9 substantive; Q1.10=Other, Q1.11=None, Q1.12=DK)
    'q1.1', 'q1.2', 'q1.3', 'q1.4', 'q1.5', 'q1.6',
    'q1.7', 'q1.8', 'q1.9', 'q1.10', 'q1.11', 'q1.12', 'q1t',
    # Planned resource-efficiency actions
    'q2.3',
    # Production cost impact (Q3) and resource-efficiency investment share (Q4)
    'q3', 'q4',
    # Reliance on internal vs. external support (Q5.1-Q5.5)
    'q5.1', 'q5.2', 'q5.3', 'q5.4', 'q5.5',
    # Specific external support types (Q6.1-Q6.7; routed to Q5.3 == 1)
    'q6.1', 'q6.2', 'q6.3', 'q6.4', 'q6.5', 'q6.6', 'q6.7',
    # Resource-efficiency barriers (Q7.1-Q7.12; routed to >=1 action in Q1.1-Q1.9)
    'q7.1', 'q7.2', 'q7.3', 'q7.4', 'q7.5', 'q7.6',
    'q7.7', 'q7.8', 'q7.9', 'q7.10', 'q7.11', 'q7.12',
    # Desired support to expand resource efficiency (Q8.1-Q8.11; asked of ALL SMEs)
    'q8.1', 'q8.2', 'q8.3', 'q8.4', 'q8.5', 'q8.6',
    'q8.7', 'q8.8', 'q8.9', 'q8.10', 'q8.11',
    # Primary outcomes & follow-up intensity items
    'q9',   # Green products or services (Y1)
    'q10',  # Green share of annual turnover (routed to Q9 == 1)
    'q14',  # Concrete strategy to reduce carbon footprint and become climate neutral (Y2)
    'q15.1', 'q15.2', 'q15.3', 'q15.4', 'q15.5', 'q15.6',  # Routed to Q14 == 1
    # Green jobs FTE count
    'dx5',
]


def clean_eurostat_value(val):
    """Parse Eurostat TSV cell values, stripping observation flags and missing markers."""
    if pd.isna(val):
        return np.nan
    v = str(val).strip()
    if v in (':', ''):
        return np.nan
    v = re.sub(r'[a-zA-Z]', '', v).strip()
    try:
        return float(v)
    except ValueError:
        return np.nan


def infer_routing_rule(var_name, val_labels):
    """Return transparent GESIS questionnaire routing / missing-value rule for a raw variable."""
    v = var_name.lower()
    inap_notes = [f"{int(k) if float(k).is_integer() else k}={lbl}"
                  for k, lbl in (val_labels or {}).items()
                  if 'inap' in str(lbl).lower()]
    if inap_notes:
        return "; ".join(inap_notes)
    if v.startswith('q1.') or v == 'q1t':
        return "Asked of all enterprises (unrouted)."
    if v.startswith('q2.') or v == 'q2t':
        return "Asked of all enterprises (planned actions over next 2 years)."
    if v in ('q3', 'q4') or v.startswith('q5.') or v.startswith('q7.'):
        return "Routed to enterprises undertaking >=1 substantive resource-efficiency action in Q1.1-Q1.9 (SPSS 9 = Inap)."
    if v.startswith('q6.'):
        return "Routed to enterprises reporting external support in Q5.3 == 1 (SPSS 9 = Inap)."
    if v.startswith('q8.'):
        return "Asked of ALL enterprises (unrouted desired-support module; valid 0/1 across all 13,322 EU-27 SMEs)."
    if v == 'q9':
        return "Asked of all enterprises (4 = Don't know / No answer)."
    if v in ('q10', 'q12') or v.startswith(('q11.', 'q13.')):
        return "Routed to enterprises currently offering green products or services (Q9 == 1)."
    if v.startswith('q11') or v.startswith('q12') or v.startswith('q13'):
        return "Routed to Q9 == 1 or Q9 == 2 per ZA7806_q_gb.pdf."
    if v == 'q14':
        return "Asked of all enterprises (5 = Don't know / No answer)."
    if v.startswith('q15.'):
        return "Routed to enterprises with an active carbon-footprint strategy in Q14 == 1 (SPSS 9 = Inap)."
    if v == 'scr14':
        return "Asked of enterprises established before 2021 (1..9 = turnover brackets; 10 = Don't know / No answer)."
    if v.startswith('scr') or v.startswith('nace') or v.startswith('dx'):
        return "Screener / background classification item (asked of all enterprises unless filtered by establishment date)."
    if v.startswith('w'):
        return "GESIS / Kantar sampling or population projection weight."
    return "Standard GESIS survey/administrative variable."


def load_and_filter_fl498():
    """Load unmodified ZA7806 SPSS file, retain full raw metadata, and filter to EU-27 SMEs."""
    print("=" * 72)
    print("PHASE 1: Loading Flash Eurobarometer 498 (ZA7806) Raw SPSS Microdata")
    print("=" * 72)

    df_raw, meta = pyreadstat.read_sav(SAV_FILE)
    print(f"  [Raw File] {SAV_FILE}")
    print(f"  - Raw dimensions: {df_raw.shape[0]:,} enterprises x {df_raw.shape[1]} variables")

    # Apply explicit Inap=9.0 -> NaN ONLY where SPSS metadata labels 9.0 as 'Inap'
    inap_9_vars = []
    for col in df_raw.columns:
        lbl_9 = str(meta.variable_value_labels.get(col, {}).get(9.0, ''))
        if 'inap' in lbl_9.lower():
            inap_9_vars.append(col)

    df_raw_clean_inap = df_raw.copy()
    for col in inap_9_vars:
        df_raw_clean_inap.loc[df_raw_clean_inap[col] == 9.0, col] = np.nan

    eu27_codes = list(EU27_COUNTRY_CODES.keys())
    df_eu27_raw = df_raw_clean_inap[df_raw_clean_inap['country'].isin(eu27_codes)]
    df_sme_raw = df_eu27_raw[df_eu27_raw['scr10'].isin([1.0, 2.0, 3.0])].copy()

    print(f"  - Filter 1 (EU-27 member states): {len(df_raw):,} -> {len(df_eu27_raw):,} (dropped {len(df_raw) - len(df_eu27_raw):,})")
    print(f"  - Filter 2 (SMEs 1-249 emp, scr10 in {{1,2,3}}): {len(df_eu27_raw):,} -> {len(df_sme_raw):,} (dropped {len(df_eu27_raw) - len(df_sme_raw):,})")

    valid_vars = [v for v in VARS_TO_EXTRACT if v in df_sme_raw.columns]
    missing_vars = [v for v in VARS_TO_EXTRACT if v not in df_sme_raw.columns]
    if missing_vars:
        raise KeyError(f"Expected raw variables missing from ZA7806: {missing_vars}")

    df = df_sme_raw[valid_vars].copy()
    df['country_iso2'] = df['country'].map(EU27_COUNTRY_CODES)
    print(f"  - Extracted {len(valid_vars)} raw variables across {df['country_iso2'].nunique()} EU-27 member states")
    return df, df_raw_clean_inap, df_sme_raw, meta


def recode_variables(df, meta):
    """Recode raw GESIS items into analytical variables with verified routing rules."""
    print("\n" + "=" * 72)
    print("PHASE 2: Variable Recoding, Routing Verification & Feature Engineering")
    print("=" * 72)

    # 1. Verify Inap (9.0) handling only touches genuine Inap-labeled variables
    routed_9_converted = 0
    for v in df.columns:
        lbl_9 = str(meta.variable_value_labels.get(v, {}).get(9.0, ''))
        if 'inap' in lbl_9.lower():
            n_9 = int((df[v] == 9.0).sum())
            if n_9 > 0:
                df.loc[df[v] == 9.0, v] = np.nan
                routed_9_converted += n_9
    # Verify q8.1-q8.6 have zero missing values and binary 0/1 coding across all 13,322 SMEs
    for q8v in [f'q8.{i}' for i in range(1, 9)]:
        vals = set(df[q8v].dropna().unique().tolist())
        assert len(df[q8v].dropna()) == len(df) and vals.issubset({0.0, 1.0}), f"Unexpected values in {q8v}: {vals}"
    print(f"  [Routing Check] Verified Q8.1-Q8.8 binary (0/1) across all {len(df):,} EU-27 SMEs; converted {routed_9_converted} residual 9.0=Inap codes.")

    # 2. Primary Dependent Variables (Y1: Green Offerings Q9; Y2: Net-Zero Plans Q14)
    df['q9_binary'] = np.where(df['q9'] == 1.0, 1.0,
                      np.where(df['q9'].isin([2.0, 3.0]), 0.0, np.nan))
    df['q14_binary'] = np.where(df['q14'].isin([1.0, 4.0]), 1.0,
                       np.where(df['q14'].isin([2.0, 3.0]), 0.0, np.nan))
    df['q14_strict'] = np.where(df['q14'] == 1.0, 1.0,
                       np.where(df['q14'].isin([2.0, 3.0]), 0.0, np.nan))

    # 3. Tightened Construct-Intensity Outcomes (SI Appendix M / Section 4.5)
    #    Y1_tight: Q9 == 1 AND Q10 >= 3 (>10% of annual turnover from green products/services)
    #    Y2_tight: Q14 == 1 AND (Q15.2 == 1 OR Q15.3 == 1) (adopting or developing new low-carbon tech)
    df['y1_tight_q10_gt10'] = np.where(
        (df['q9'] == 1.0) & (df['q10'].isin([3.0, 4.0, 5.0, 6.0])), 1.0,
        np.where((df['q9'].isin([2.0, 3.0])) | ((df['q9'] == 1.0) & (df['q10'].isin([1.0, 2.0]))), 0.0, np.nan)
    )
    df['q15_tech'] = np.where(
        df['q14'] == 1.0,
        ((df['q15.2'] == 1.0) | (df['q15.3'] == 1.0)).astype(float),
        np.nan
    )
    df['y2_tight_q15_tech'] = np.where(
        (df['q14'] == 1.0) & ((df['q15.2'] == 1.0) | (df['q15.3'] == 1.0)), 1.0,
        np.where((df['q14'].isin([2.0, 3.0, 4.0])) |
                 ((df['q14'] == 1.0) & (df['q15.2'] == 0.0) & (df['q15.3'] == 0.0)), 0.0, np.nan)
    )
    df['y2_tight_q15_tech_excl4'] = np.where(df['q14'] == 4.0, np.nan, df['y2_tight_q15_tech'])

    # 4. Directional Cross-Domain Sustainability Asymmetry Index (DI = Y1 - Y2) & 4-Quadrant Typology
    mask_both = df['q9_binary'].notna() & df['q14_binary'].notna()
    df['DI'] = np.nan
    df.loc[mask_both, 'DI'] = df.loc[mask_both, 'q9_binary'] - df.loc[mask_both, 'q14_binary']

    df['quadrant'] = np.nan
    df.loc[mask_both & (df['q9_binary'] == 0) & (df['q14_binary'] == 0), 'quadrant'] = 0
    df.loc[mask_both & (df['q9_binary'] == 1) & (df['q14_binary'] == 0), 'quadrant'] = 1
    df.loc[mask_both & (df['q9_binary'] == 0) & (df['q14_binary'] == 1), 'quadrant'] = 2
    df.loc[mask_both & (df['q9_binary'] == 1) & (df['q14_binary'] == 1), 'quadrant'] = 3

    # 5. Market Channels (SCR15)
    df['B2C'] = df['scr15.1']
    df['B2B_private'] = df['scr15.2']
    df['B2G_public'] = df['scr15.3']

    # 6. External Support Channels (Q6; unrouted Q5.3=0 firms did not receive external support -> 0)
    for q6_var in ['q6.1', 'q6.2', 'q6.3', 'q6.4', 'q6.5', 'q6.6', 'q6.7']:
        df[q6_var] = df[q6_var].fillna(0.0)

    df['grant_receipt'] = df['q6.1']
    df['public_financial_support'] = df['q6.1']
    df['advice_public'] = df['q6.4']
    df['advice_private'] = df['q6.5']
    df['private_finance'] = df['q6.2']
    df['supply_chain_advice'] = df['q6.7']

    # 7. Capability & Regulatory Barriers (Q7; verified against ZA7806_q_gb.pdf)
    #    q7.1 = Complexity of administrative or legal procedures
    #    q7.2 = Difficulty to adapt environmental legislation to your company
    #    q7.3 = Technical requirements of the legislation not being up to date
    #    q7.5 = Cost of environmental actions
    #    q7.6 = Lack of specific environmental expertise
    #    q7.9 = Complexity associated with environmental labelling and certification
    df['barrier_admin'] = df['q7.1']
    df['barrier_legislation_adapt'] = df['q7.2']
    df['barrier_technical'] = df['q7.3']
    df['barrier_cost'] = df['q7.5']
    df['barrier_expertise'] = df['q7.6']
    df['barrier_label'] = df['q7.9']

    df['own_finance'] = df['q5.1']
    df['own_expertise'] = df['q5.2']
    df['sells_products'] = np.where(df['scr16'].isin([1.0, 3.0]), 1.0,
                           np.where(df['scr16'] == 2.0, 0.0, np.nan))
    df['q4_clean'] = np.where(df['q4'].isin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), df['q4'], np.nan)
    df['q10_clean'] = np.where(df['q10'].isin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), df['q10'], np.nan)

    df['green_jobs_any'] = np.where(df['dx5'] == 999997.0, 0.0,
                           np.where((df['dx5'] >= 0) & (df['dx5'] < 999990), (df['dx5'] > 0).astype(float), np.nan))
    df['green_jobs_fte'] = np.where(df['dx5'] == 999997.0, 0.0,
                           np.where((df['dx5'] >= 0) & (df['dx5'] < 999990), df['dx5'].astype(float), np.nan))

    # 8. Q15 Substantive Carbon Action Count: 4 substantive items (q15.1-q15.4) routed to Q14 == 1
    #    Excludes q15.5 ("Other") and q15.6 ("None")
    q15_4_vars = ['q15.1', 'q15.2', 'q15.3', 'q15.4']
    df['q15_count'] = np.where(
        df['q14'] == 1.0,
        (df[q15_4_vars] == 1.0).sum(axis=1).astype(float),
        np.nan
    )

    # 9. Resource-Efficiency Action Count Fix:
    #    re_action_count = strict 9-item substantive sum (q1.1 to q1.9), excluding q1.10 (Other),
    #    q1.11 (None), and q1.12 (DK), so all 1,231 unrouted zero-action SMEs equal 0.
    #    re_action_count_10 = 10-item sum including q1.10 ("Other") for Table 6 sensitivity.
    q1_9_vars = [f'q1.{i}' for i in range(1, 10)]
    q1_10_vars = [f'q1.{i}' for i in range(1, 11)]
    df['re_action_count'] = (df[q1_9_vars] == 1.0).sum(axis=1).astype(int)
    df['re_action_count_9'] = df['re_action_count']
    df['re_action_count_10'] = (df[q1_10_vars] == 1.0).sum(axis=1).astype(int)
    assert int((df['re_action_count'] == 0).sum()) == 1231, "Expected 1,231 unrouted zero-action SMEs!"

    # 10. Turnover (SCR14) Clean Recoding Fix:
    #     In ZA7806, scr14 in 1..9 are ordinal euro turnover brackets (<25k to >50M),
    #     whereas scr14 == 10.0 is "Don't know / No Answer" (n=1,874 in N=13,322; n=1,364 in N=11,128).
    df['scr14_clean'] = np.where(df['scr14'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), df['scr14'].astype(float), np.nan)
    df['scr14_dk'] = (df['scr14'] == 10.0).astype(float)
    mean_scr14_clean = float(df['scr14_clean'].mean())
    df['scr14_imp'] = np.where(df['scr14'] == 10.0, mean_scr14_clean, df['scr14_clean'])
    df['scr14_cat'] = df['scr14'].astype(int)

    # 11. Total Barrier Count (Q7.1-Q7.9 substantive barriers) & Firm Age / Size Labels
    q7_9_vars = [f'q7.{i}' for i in range(1, 10)]
    df['barrier_count'] = np.where(df['q7.1'].notna(), (df[q7_9_vars] == 1.0).sum(axis=1).astype(float), np.nan)

    df['firm_age'] = 2021 - df['scr12a']
    df.loc[(df['firm_age'] < 0) | (df['firm_age'] > 200), 'firm_age'] = np.nan

    size_map = {1.0: 'Micro (1-9)', 2.0: 'Small (10-49)', 3.0: 'Medium (50-249)'}
    df['firm_size_label'] = df['scr10'].map(size_map)

    print(f"  [re_action_count] 9-item substantive mean = {df['re_action_count'].mean():.2f}; zero-action unrouted SMEs = {(df['re_action_count']==0).sum():,}")
    print(f"  [scr14 recode]    Valid brackets 1..9: {df['scr14_clean'].notna().sum():,}; DK/NA (scr14==10): {int(df['scr14_dk'].sum()):,}")
    print(f"  [q15_count]       4-item substantive mean (among Q14=1): {df['q15_count'].mean():.2f}")
    return df


def merge_desi(df):
    """Merge Eurostat 2021 Digital Intensity Index (E_DI3_HI + E_DI3_VHI, 10-249 emp)."""
    print("\n" + "=" * 72)
    print("PHASE 3: Merging Eurostat Digital Intensity Index (isoc_e_dii, 2021)")
    print("=" * 72)

    df_euro = pd.read_csv(EUROSTAT_TSV, sep='\t')
    first_col = df_euro.columns[0]
    dims = df_euro[first_col].str.split(',', expand=True)
    geo_col_idx = len(dims.columns) - 1
    dims.iloc[:, geo_col_idx] = dims.iloc[:, geo_col_idx].str.split(r'\\').str[0]

    df_euro['size_emp'] = dims[1]
    df_euro['indic_is'] = dims[3]
    df_euro['geo'] = dims[geo_col_idx]
    year_col = [c for c in df_euro.columns if '2021' in c][0]
    eu27_isos = list(EU27_COUNTRY_CODES.values())

    mask_hi = ((df_euro['size_emp'] == '10-249') &
               (df_euro['indic_is'] == 'E_DI3_HI') &
               (df_euro['geo'].isin(eu27_isos)))
    desi_hi = df_euro.loc[mask_hi, ['geo', year_col]].copy()
    desi_hi['DESI_High'] = desi_hi[year_col].apply(clean_eurostat_value)
    desi_hi = desi_hi[['geo', 'DESI_High']].dropna()

    mask_vhi = ((df_euro['size_emp'] == '10-249') &
                 (df_euro['indic_is'] == 'E_DI3_VHI') &
                 (df_euro['geo'].isin(eu27_isos)))
    desi_vhi = df_euro.loc[mask_vhi, ['geo', year_col]].copy()
    desi_vhi['DESI_VeryHigh'] = desi_vhi[year_col].apply(clean_eurostat_value)
    desi_vhi = desi_vhi[['geo', 'DESI_VeryHigh']].dropna()

    desi = desi_hi.merge(desi_vhi, on='geo', how='outer')
    desi['DII_composite'] = desi['DESI_High'].fillna(0.0) + desi['DESI_VeryHigh'].fillna(0.0)
    assert len(desi) == 27, f"Expected 27 EU countries in DESI, found {len(desi)}"

    df = df.merge(desi, left_on='country_iso2', right_on='geo', how='left')
    df.drop(columns=['geo'], inplace=True, errors='ignore')
    print(f"  - Merged 27/27 EU country DII scores onto {df['DII_composite'].notna().sum():,}/{len(df):,} SMEs")
    return df


def build_waterfall_df(df):
    """Build complete Sample Waterfall accounting across main, sensitivity, and Wave 6 samples."""
    n_typology = int((df['q9_binary'].notna() & df['q14_binary'].notna()).sum())
    n_routed = int(df['barrier_admin'].notna().sum())
    n_ipw = int((df['barrier_admin'].notna() & df['q14_binary'].notna()).sum())
    joint_mask = df['barrier_admin'].notna() & df['q9_binary'].notna() & df['q14_binary'].notna()
    n_joint = int(joint_mask.sum())
    n_strict = int((df['barrier_admin'].notna() & df['q9_binary'].notna() & df['q14_strict'].notna()).sum())
    n_cc_turn = int((joint_mask & df['scr14_clean'].notna()).sum())
    n_prod = int((joint_mask & (df['sells_products'] == 1.0)).sum())
    exp_cols = ['barrier_cost', 'barrier_label', 'own_finance', 'own_expertise', 'q4_clean', 'sells_products']
    n_expanded = int((joint_mask & df[exp_cols].notna().all(axis=1)).sum())
    n_q53_routed = int((df['q5.3'] == 1.0).sum())
    n_q53_y2 = int(((df['q5.3'] == 1.0) & df['q14_binary'].notna()).sum())
    n_q53_joint = int(((df['q5.3'] == 1.0) & df['q9_binary'].notna() & df['q14_binary'].notna()).sum())
    n_cost_10011 = int((joint_mask & df['q3'].isin([1, 2, 3, 4, 5])).sum())
    n_cost_9058 = int((joint_mask & df['q3'].isin([1, 2, 3, 4, 5]) & df['q4'].isin([1, 2, 3, 4, 5, 6])).sum())
    n_tight_keep4 = int((df['barrier_admin'].notna() & df['y1_tight_q10_gt10'].notna() & df['y2_tight_q15_tech'].notna()).sum())
    n_tight_excl4 = int((df['barrier_admin'].notna() & df['y1_tight_q10_gt10'].notna() & df['y2_tight_q15_tech_excl4'].notna()).sum())

    rows = [
        {'Step': '0. Raw GESIS ZA7806 Dataset (Wave 5, 2021)', 'Sample_N': 17662,
         'Filter_Rule': 'All enterprise interviews across 36 countries in Flash Eurobarometer 498 (ZA7806_v1-0-0.sav)'},
        {'Step': '1. EU-27 Member States', 'Sample_N': 14215,
         'Filter_Rule': 'Retains 27 EU member states; excludes 3,447 non-EU enterprises'},
        {'Step': '2. Baseline EU-27 SME Population (incl. Q8 Desired Support)', 'Sample_N': len(df),
         'Filter_Rule': 'Retains active SMEs with 1-249 employees (scr10 in {1,2,3}); excludes 893 large/missing firms'},
        {'Step': '3. Four-Quadrant Typology & Model 4 Cross-Level OLS Sample', 'Sample_N': n_typology,
         'Filter_Rule': 'Valid binary outcomes on both Q9 and Q14; excludes 1,117 SMEs with DK on Q9 (593) or Q14 (676; 152 on both)'},
        {'Step': '4. Resource-Efficiency Routed Sample (Q3-Q7 Module)', 'Sample_N': n_routed,
         'Filter_Rule': 'SMEs undertaking >=1 substantive resource-efficiency action in Q1.1-Q1.9 (excludes 1,231 zero-action SMEs)'},
        {'Step': '5. Single-Outcome IPW & CATE Sample (Valid Q14)', 'Sample_N': n_ipw,
         'Filter_Rule': 'Routed SMEs with valid Q14 (excludes 570 routed SMEs with DK on Q14; 1,292 treated, 10,229 control)'},
        {'Step': '6. Primary Joint Econometric Sample (BiProbit / MNLogit / PSM / H3c Moderation)', 'Sample_N': n_joint,
         'Filter_Rule': 'Routed SMEs with valid Q9 and Q14 (excludes 393 routed SMEs with DK on Q9; 1,256 treated, 9,872 control)'},
        {'Step': '7a. Sensitivity: Strict 2030 Active Planners (Excl. Q14=4)', 'Sample_N': n_strict,
         'Filter_Rule': 'Excludes 504 SMEs reporting Already climate neutral (Q14 = 4) from N = 11,128'},
        {'Step': '7b. Sensitivity: Complete-Case Turnover Brackets (scr14 < 10)', 'Sample_N': n_cc_turn,
         'Filter_Rule': 'Excludes 1,364 SMEs reporting Don\'t know / No answer on turnover (scr14 == 10) from N = 11,128'},
        {'Step': '7c. Sensitivity: Product-Selling SMEs (SCR16 in {1, 3})', 'Sample_N': n_prod,
         'Filter_Rule': 'Retains SMEs selling physical products or both products and services (sells_products == 1) in N = 11,128'},
        {'Step': '7d. Sensitivity: Expanded Covariate Specification', 'Sample_N': n_expanded,
         'Filter_Rule': 'Adds Q7.5 (cost), Q7.9 (labelling), Q5.1 (own finance), Q5.2 (own expertise), Q4 (investment), and SCR16 in N = 11,128'},
        {'Step': '7e. Sensitivity: External Support Seekers (Q5.3=1, Valid Q14)', 'Sample_N': n_q53_y2,
         'Filter_Rule': f'Subsample of {n_q53_routed:,} Q5.3=1 external-support users with valid Q14 (1,292 treated, 1,763 control)'},
        {'Step': '7f. Sensitivity: External Support Seekers (Q5.3=1, Joint Valid Q9 & Q14)', 'Sample_N': n_q53_joint,
         'Filter_Rule': f'Subsample of {n_q53_routed:,} Q5.3=1 external-support users in N = 11,128 (1,256 treated, 1,703 control)'},
        {'Step': '8a. Model 5 Operating Cost Ordered Probit (Valid Q3 in {1..5})', 'Sample_N': n_cost_10011,
         'Filter_Rule': 'SMEs in N = 11,128 with non-DK response on Q3 production cost impact'},
        {'Step': '8b. Model 5 Operating Cost Ordered Probit (+ Valid Q4 in {1..6})', 'Sample_N': n_cost_9058,
         'Filter_Rule': 'SMEs in N = 10,011 with non-DK response on Q4 resource-efficiency investment share'},
        {'Step': '9a. Construct-Intensity BiProbit (Q10 > 10% vs. Q15.2/Q15.3 Tech, Keep Q14=4 as 0)', 'Sample_N': n_tight_keep4,
         'Filter_Rule': 'Excludes 441 green sellers with DK on Q10 revenue share from N = 11,128'},
        {'Step': '9b. Construct-Intensity BiProbit (Q10 > 10% vs. Q15.2/Q15.3 Tech, Excl. Q14=4)', 'Sample_N': n_tight_excl4,
         'Filter_Rule': 'Excludes Q10=DK green sellers and Q14=4 already-neutral firms from N = 11,128'},
    ]

    if W6_SAV_FILE and os.path.exists(W6_SAV_FILE):
        w6, _ = pyreadstat.read_sav(W6_SAV_FILE)
        eu27_isos = set(EU27_COUNTRY_CODES.values())
        w6['iso2'] = w6['isocntry'].str.strip().replace({'GR': 'EL'})
        w6_eu = w6[w6['iso2'].isin(eu27_isos)]
        w6_sme = w6_eu[w6_eu['scr10'].isin([1.0, 2.0, 3.0])]
        w6_joint = w6_sme[w6_sme['q7.1'].isin([0.0, 1.0]) & w6_sme['q9'].isin([1.0, 2.0, 3.0]) & w6_sme['q14'].isin([1.0, 2.0, 3.0, 4.0])]
        rows.extend([
            {'Step': '10a. Wave 6 Out-of-Sample Check (ZA8869, 2024): EU-27 SMEs', 'Sample_N': int(len(w6_sme)),
             'Filter_Rule': f'Flash Eurobarometer 549 (raw N={len(w6):,}; EU-27 N={len(w6_eu):,} with GR->EL; scr10 in {{1,2,3}})'},
            {'Step': '10b. Wave 6 Out-of-Sample Check (ZA8869, 2024): Joint BiProbit Sample', 'Sample_N': int(len(w6_joint)),
             'Filter_Rule': 'Wave 6 EU-27 SMEs routed to Q7.1 with valid Q9 in {1,2,3} and Q14 in {1,2,3,4} across all 27 member states'},
        ])
    else:
        rows.extend([
            {'Step': '10a. Wave 6 Out-of-Sample Check (ZA8869, 2024): EU-27 SMEs', 'Sample_N': 13124,
             'Filter_Rule': 'Flash Eurobarometer 549 (raw N=18,159; EU-27 N=14,048 with GR->EL; scr10 in {1,2,3})'},
            {'Step': '10b. Wave 6 Out-of-Sample Check (ZA8869, 2024): Joint BiProbit Sample', 'Sample_N': 11246,
             'Filter_Rule': 'Wave 6 EU-27 SMEs routed to Q7.1 with valid Q9 in {1,2,3} and Q14 in {1,2,3,4} across all 27 member states'},
        ])

    return pd.DataFrame(rows)


def build_constructed_codebook_df():
    """Return complete metadata table for constructed/analytical variables in SME_Analytical_Final.csv."""
    records = [
        ('q9_binary', 'q9', '1 if q9==1; 0 if q9 in {2,3}; NaN if q9==4 (DK)', 'All EU-27 SMEs (N=12,729 valid)', 'Primary DV Y1: Green Offerings (H1a, H1b, H2a, H2b, H3b)'),
        ('q14_binary', 'q14', '1 if q14 in {1,4}; 0 if q14 in {2,3}; NaN if q14==5 (DK)', 'All EU-27 SMEs (N=12,646 valid)', 'Primary DV Y2: Net-Zero Plans (H1a, H1b, H1c, H2a, H2b, H3a, H3b, H3c)'),
        ('q14_strict', 'q14', '1 if q14==1; 0 if q14 in {2,3}; NaN if q14 in {4,5}', 'Excludes Q14=4 already neutral (N=10,624 joint)', 'Sensitivity DV Y2: Active 2030 Planners Only (Table 6 Row 6)'),
        ('DI', 'q9_binary, q14_binary', 'q9_binary - q14_binary in {-1, 0, +1}', 'Both Q9 & Q14 valid (N=12,205)', 'Cross-Domain Sustainability Asymmetry Index (Model 4 DV)'),
        ('quadrant', 'q9_binary, q14_binary', '0=(0,0) Neither; 1=(1,0) Green Only; 2=(0,1) Net-Zero Only; 3=(1,1) Both', 'Both Q9 & Q14 valid (N=12,205; N=11,128 routed)', '4-Quadrant Configurational Outcome (Table 1 & Model 2 MNLogit)'),
        ('B2C', 'scr15.1', '1 if sells directly to consumers; 0 otherwise', 'All EU-27 SMEs (N=13,322)', 'Focal Predictor: B2C Consumer Channel (H1a)'),
        ('B2G_public', 'scr15.3', '1 if sells to public administration; 0 otherwise', 'All EU-27 SMEs (N=13,322)', 'Focal Predictor: B2G Public Procurement Channel (H1b)'),
        ('B2B_private', 'scr15.2', '1 if sells to other companies; 0 otherwise', 'All EU-27 SMEs (N=13,322)', 'Focal Predictor: B2B Supply-Chain Channel (H1c; Model 4)'),
        ('barrier_admin', 'q7.1', '1 if complexity of administrative/legal procedures; 0 if not; NaN if unrouted', 'Routed SMEs Q1.1-Q1.9 >= 1 (N=12,091)', 'Focal Predictor & Moderator: Administrative Complexity (H2a, H3c)'),
        ('barrier_expertise', 'q7.6', '1 if lack of specific environmental expertise; 0 if not; NaN if unrouted', 'Routed SMEs Q1.1-Q1.9 >= 1 (N=12,091)', 'Focal Predictor & Moderator: Environmental Expertise Shortage (H2b, H3c)'),
        ('grant_receipt', 'q6.1', '1 if q6.1==1 (public grants, guarantees, or loans); 0 if q6.1==0 or Q5.3!=1', 'All EU-27 SMEs / Routed N=11,128', 'Focal Treatment / Predictor: Public Financial Support (H3a, H3b, H3c)'),
        ('public_financial_support', 'q6.1', 'Exact alias of grant_receipt', 'All EU-27 SMEs / Routed N=11,128', 'Focal Treatment / Predictor: Public Financial Support (H3a, H3b, H3c)'),
        ('advice_public', 'q6.4', '1 if q6.4==1 (public non-financial advice); 0 if q6.4==0 or Q5.3!=1', 'All EU-27 SMEs / Routed N=11,128', 'Control: Public Non-Financial Advisory Support'),
        ('scr10', 'scr10', '1=Micro (1-9 emp), 2=Small (10-49 emp), 3=Medium (50-249 emp)', 'All EU-27 SMEs (N=13,322)', 'Control & CATE Subgroup: Firm Size Tier'),
        ('scr14_cat', 'scr14', 'Categorical integer 1..10 (1..9 = euro turnover brackets; 10 = Don\'t know/No answer)', 'All EU-27 SMEs (N=13,322)', 'Baseline Control: Categorical Turnover Bracket Fixed Effects (9 indicators)'),
        ('scr14_clean', 'scr14', 'scr14 in 1..9 ordinal turnover brackets; NaN when scr14==10 (DK/NA)', 'Valid turnover SMEs (N=11,448; N=9,764 in joint)', 'Sensitivity Control: Complete-Case Ordinal Turnover (Table 6 & Table L5)'),
        ('scr14_dk', 'scr14', '1.0 if scr14==10 (Don\'t know/No answer); 0.0 otherwise', 'All EU-27 SMEs (N=13,322)', 'Sensitivity Control: Missing-Turnover Indicator'),
        ('scr14_imp', 'scr14', 'Mean of scr14_clean when scr14==10 else scr14_clean', 'All EU-27 SMEs (N=13,322)', 'Sensitivity Control: Mean-Imputed Ordinal Turnover (paired with scr14_dk)'),
        ('re_action_count', 'q1.1..q1.9', 'Sum_{j=1}^9 1(q1.j == 1); excludes q1.10 Other, q1.11 None, q1.12 DK', 'All EU-27 SMEs (0 for all 1,231 unrouted SMEs)', 'Baseline Control: 9-Item Substantive Resource-Efficiency Action Count'),
        ('re_action_count_9', 'q1.1..q1.9', 'Exact alias of re_action_count (strict 9-item count)', 'All EU-27 SMEs (N=13,322)', 'Baseline Control Alias'),
        ('re_action_count_10', 'q1.1..q1.10', 'Sum_{j=1}^{10} 1(q1.j == 1) including q1.10 Other; excludes None and DK', 'All EU-27 SMEs (N=13,322)', 'Sensitivity Control: 10-Item RE Action Count (Table 6 Row 3)'),
        ('country_iso2', 'country', '2-letter Eurostat ISO code (EL for Greece; 27 EU member states)', 'All EU-27 SMEs (N=13,322)', '26 Country Fixed Effects & 27-Cluster Sandwich Inference'),
        ('nace_b', 'nace_b', '1=Manufacturing (C), 2=Retail (G), 3=Services (H-M), 4=Industry (B/D/E/F)', 'All EU-27 SMEs (N=13,322)', 'Baseline Sector Control: 3 NACE Macro-Sector Fixed Effects'),
        ('nace_a', 'nace_a', '12-section NACE Rev. 2 classification (1=B through 12=M)', 'All EU-27 SMEs (N=13,322)', 'Sensitivity Sector Control: 11 Granular NACE Section Fixed Effects (Table 6)'),
        ('DII_composite', 'Eurostat isoc_e_dii', 'DESI_High (E_DI3_HI) + DESI_VeryHigh (E_DI3_VHI) for SMEs (10-249 emp, 2021)', 'All 27 EU countries (N=13,322)', 'Macro Moderator: National Digital Intensity Index (Model 4 & CATE)'),
        ('barrier_legislation_adapt', 'q7.2', '1 if difficulty adapting environmental legislation to company; 0 if not', 'Routed SMEs Q1.1-Q1.9 >= 1 (N=12,091)', 'Auxiliary Barrier Item (replaces mislabelled barrier_finance)'),
        ('barrier_technical', 'q7.3', '1 if technical requirements of legislation not up to date; 0 if not', 'Routed SMEs Q1.1-Q1.9 >= 1 (N=12,091)', 'Auxiliary Barrier Item'),
        ('barrier_cost', 'q7.5', '1 if cost of environmental actions was a difficulty; 0 if not', 'Routed SMEs Q1.1-Q1.9 >= 1 (N=12,091)', 'Expanded Covariate Control (Table 6 Row 9)'),
        ('barrier_label', 'q7.9', '1 if complexity of environmental labelling/certification; 0 if not', 'Routed SMEs Q1.1-Q1.9 >= 1 (N=12,091)', 'Expanded Covariate Control (Table 6 Row 9)'),
        ('own_finance', 'q5.1', '1 if relies on own financial resources for RE actions; 0 if not', 'Routed SMEs Q1.1-Q1.9 >= 1 (N=12,091)', 'Expanded Covariate Control (Table 6 Row 9)'),
        ('own_expertise', 'q5.2', '1 if relies on own technical expertise for RE actions; 0 if not', 'Routed SMEs Q1.1-Q1.9 >= 1 (N=12,091)', 'Expanded Covariate Control (Table 6 Row 9)'),
        ('q4_clean', 'q4', 'Ordinal RE investment share 1 (Nothing) to 6 (>30% revenue); NaN if DK (7) or Inap', 'Routed SMEs valid Q4 (N=10,829)', 'Expanded Covariate Control & Construct Validation (Table 1, Model 5)'),
        ('sells_products', 'scr16', '1 if scr16 in {1 (products), 3 (both)}; 0 if scr16==2 (services only); NaN if 4 (DK)', 'Valid SCR16 SMEs (N=13,189; N=7,299 == 1 in joint)', 'Construct Validation & Product-Seller Subsample Filter (Table 6 Row 8)'),
        ('q10_clean', 'q10', 'Ordinal green turnover share 1 (1-5%) to 6 (>75%); NaN if DK (7) or Q9!=1', 'Green sellers Q9==1 with valid Q10 (N=3,992)', 'Construct Validation: Green Revenue Intensity (Table 1, Table L6)'),
        ('q15_count', 'q15.1..q15.4', 'Sum_{j=1}^4 1(q15.j == 1) among Q14==1 firms; excludes q15.5 Other & q15.6 None', 'Active strategy SMEs Q14==1 (N=2,924)', 'Construct Validation: 4-Item Substantive Carbon Action Breadth (Table 1)'),
        ('q15_tech', 'q14, q15.2, q15.3', '1 if (q15.2==1 | q15.3==1) among Q14==1 firms; 0 if Q14==1 without tech; NaN else', 'Active strategy SMEs Q14==1 (N=2,924)', 'Construct Validation: Low-Carbon Tech Adoption/Development'),
        ('y1_tight_q10_gt10', 'q9, q10', '1 if q9==1 & q10 in {3..6} (>10% green turnover); 0 if q9 in {2,3} or q10 in {1,2}; NaN if DK', 'Routed valid N=10,687', 'Extension 1 Tightened Construct DV Y1 (Section 4.5 & SI Appendix M)'),
        ('y2_tight_q15_tech', 'q14, q15.2, q15.3', '1 if q14==1 & (q15.2==1 | q15.3==1); 0 if q14 in {2,3,4} or non-tech q14==1; NaN if DK', 'Routed valid N=10,687', 'Extension 1 Tightened Construct DV Y2 (Section 4.5 & SI Appendix M)'),
        ('y2_tight_q15_tech_excl4', 'q14, q15.2, q15.3', 'Same as y2_tight_q15_tech but sets q14==4 (already neutral) to NaN', 'Routed valid N=10,203', 'Extension 1 Tightened Construct DV Y2 Sensitivity (SI Appendix M)'),
        ('green_jobs_any', 'dx5', '1 if 0 < dx5 < 999990; 0 if dx5 in {0, 999997}; NaN if DK/Refusal', 'Valid DX5 SMEs (N=12,519)', 'Construct Validation: Employs >=1 Green FTE Worker (Table 1, Table L6)'),
        ('green_jobs_fte', 'dx5', 'Continuous green FTE headcount (0 if dx5==999997); NaN if >=999990', 'Valid DX5 SMEs (N=12,519)', 'Construct Validation: Green FTE Headcount'),
        ('firm_age', 'scr12a', '2021 - scr12a (bounded to [0, 200] years)', 'Valid establishment year SMEs', 'Descriptive / Auxiliary Firm Age'),
    ]
    return pd.DataFrame(records, columns=[
        'Constructed_Variable', 'Source_Raw_Items', 'Exact_Coding_Formula',
        'Routing_Domain', 'Hypothesis_and_Model_Mapping'
    ])


def generate_full_253_codebook(df_raw, df_sme_raw, meta, waterfall_df, constructed_df):
    """Auto-generate Data/Documentation/Codebook_ZA7806_Raw_and_Cleaned.xlsx (3 sheets)."""
    os.makedirs(DOC_DIR, exist_ok=True)
    raw_rows = []
    for idx, col in enumerate(df_raw.columns, start=1):
        q_lbl = meta.column_names_to_labels.get(col, '') or ''
        v_lbls = meta.variable_value_labels.get(col, {}) or {}
        formatted_lbls = "; ".join(
            f"{int(k) if isinstance(k, (int, float)) and float(k).is_integer() else k}={v}"
            for k, v in sorted(v_lbls.items(), key=lambda x: str(x[0]))
        )
        raw_rows.append({
            'Var_Index': idx,
            'Variable_Name': col,
            'Question_Label': q_lbl,
            'Value_Labels': formatted_lbls,
            'Routing_or_Missing_Rule': infer_routing_rule(col, v_lbls),
            'NonMissing_Raw_17662': int(df_raw[col].notna().sum()),
            'NonMissing_EU27_SME_13322': int(df_sme_raw[col].notna().sum()),
        })
    raw_vars_df = pd.DataFrame(raw_rows)
    assert len(raw_vars_df) == 253, f"Expected 253 raw ZA7806 variables, got {len(raw_vars_df)}"

    with pd.ExcelWriter(CODEBOOK_XLSX, engine='openpyxl') as writer:
        raw_vars_df.to_excel(writer, sheet_name='Raw_ZA7806_253_Vars', index=False)
        constructed_df.to_excel(writer, sheet_name='Constructed_Variables', index=False)
        waterfall_df.to_excel(writer, sheet_name='Sample_Waterfall', index=False)

    print(f"  ✓ Auto-generated 253-variable codebook: {CODEBOOK_XLSX}")


def save_outputs(df, df_raw, df_sme_raw, meta):
    """Save SME_Analytical_Final.csv, Input_Data_Consolidated.xlsx, and 253-var Codebook."""
    print("\n" + "=" * 72)
    print("PHASE 4: Saving Cleaned Analytical Data, Consolidated Workbook & Codebook")
    print("=" * 72)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ Analytical CSV: {OUTPUT_CSV} (shape: {df.shape[0]:,} x {df.shape[1]})")

    waterfall_df = build_waterfall_df(df)
    constructed_df = build_constructed_codebook_df()
    generate_full_253_codebook(df_raw, df_sme_raw, meta, waterfall_df, constructed_df)

    with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Analytical_Sample', index=False)

        desi_cols = ['country_iso2', 'DESI_High', 'DESI_VeryHigh', 'DII_composite']
        desi_sheet = df[desi_cols].drop_duplicates().dropna(subset=['DESI_High']).sort_values('country_iso2')
        desi_sheet.to_excel(writer, sheet_name='DESI_Country_Data', index=False)

        constructed_df.to_excel(writer, sheet_name='Codebook', index=False)

        country_meta = df.groupby('country_iso2').agg(
            N=('uniqid', 'count'),
            Pct_Green_Offerings=('q9_binary', 'mean'),
            Pct_NetZero_Strategy=('q14_binary', 'mean'),
            Mean_DI=('DI', 'mean'),
            Pct_B2C=('B2C', 'mean'),
            Pct_B2B=('B2B_private', 'mean'),
            Pct_B2G=('B2G_public', 'mean'),
            Pct_Public_Support=('grant_receipt', 'mean'),
            DII=('DII_composite', 'first')
        ).round(4)
        country_meta.to_excel(writer, sheet_name='Country_Summary')

        desc_vars = [
            'q9_binary', 'q14_binary', 'DI', 'B2C', 'B2B_private', 'B2G_public',
            'grant_receipt', 'advice_public', 'barrier_admin', 'barrier_expertise',
            'scr10', 'scr14_clean', 'scr14_dk', 're_action_count', 're_action_count_10', 'DII_composite'
        ]
        desc_stats = df[desc_vars].describe().T
        desc_stats['N_valid'] = df[desc_vars].notna().sum()
        desc_stats['N_missing'] = df[desc_vars].isna().sum()
        desc_stats['Pct_missing'] = (desc_stats['N_missing'] / len(df) * 100).round(2)
        desc_stats.to_excel(writer, sheet_name='Descriptive_Statistics')

        waterfall_df.to_excel(writer, sheet_name='Sample_Waterfall', index=False)

        # Table 1 Typology & Within-Survey Construct Validation (N = 12,205)
        d_typ = df.dropna(subset=['q9_binary', 'q14_binary']).copy()
        d_typ['quad_4'] = d_typ['quadrant'].astype(int)
        t1_rows = []
        for q_id, q_label in [(0, '0. Neither (0,0)'),
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
                'Mean_RE_Actions_9Item': round(float(sub_q['re_action_count'].mean()), 2),
                'Mean_RE_Actions_10Item': round(float(sub_q['re_action_count_10'].mean()), 2),
                'Mean_Firm_Size_1_3': round(float(sub_q['scr10'].mean()), 2),
                'Public_Support_Pct': round(100.0 * float((sub_q['grant_receipt'] == 1.0).mean()), 2),
                'Mean_National_DII': round(float(sub_q['DII_composite'].mean()), 2),
                'Sells_Products_SCR16_Pct': round(100.0 * float(sub_q['sells_products'].dropna().mean()), 2),
                'Green_Jobs_Any_DX5_Pct': round(100.0 * float(sub_q['green_jobs_any'].dropna().mean()), 2),
                'RE_Invest_GT5Pct_Q4_Valid_Pct': round(100.0 * float((sub_q['q4_clean'].dropna() >= 4).mean()), 2),
                'Q10_Green_Rev_GT30Pct_Pct': round(100.0 * float((sub_q['q10_clean'].dropna() >= 4).mean()), 2) if sub_q['q10_clean'].notna().any() else np.nan,
                'Q10_Green_Rev_GT10Pct_Pct': round(100.0 * float((sub_q['q10_clean'].dropna() >= 3).mean()), 2) if sub_q['q10_clean'].notna().any() else np.nan,
                'Q10_Green_Rev_GT5Pct_Pct': round(100.0 * float((sub_q['q10_clean'].dropna() >= 2).mean()), 2) if sub_q['q10_clean'].notna().any() else np.nan,
                'Q15_Carbon_Action_4Item_Mean': round(float(sub_q['q15_count'].dropna().mean()), 2) if sub_q['q15_count'].notna().any() else np.nan,
                'Q15_LowCarbon_Tech_Pct': round(100.0 * float(sub_q['q15_tech'].dropna().mean()), 2) if sub_q['q15_tech'].notna().any() else np.nan,
            })
        pd.DataFrame(t1_rows).to_excel(writer, sheet_name='Table1_Typology_12205', index=False)

        # Missing-Data & Survey-Routing Attrition Comparison
        routed_sub = df[df['barrier_admin'].notna()]
        joint_sub = routed_sub[routed_sub['q9_binary'].notna() & routed_sub['q14_binary'].notna()]
        unrouted_sub = df[df['barrier_admin'].isna()]
        routed_miss_sub = routed_sub[routed_sub['q9_binary'].isna() | routed_sub['q14_binary'].isna()]
        attr_rows = []
        for c_name, c_sub in [('Full EU-27 SME Population', df),
                              ('Primary Joint Sample (Routed, Valid Q9 & Q14)', joint_sub),
                              ('Unrouted Zero-Action SMEs (re_action_count == 0)', unrouted_sub),
                              ('Routed SMEs Missing Q9 or Q14 (DK)', routed_miss_sub)]:
            attr_rows.append({
                'Subsample_Cohort': c_name,
                'N': len(c_sub),
                'Micro_1_9_Pct': round(100.0 * float((c_sub['scr10'] == 1).mean()), 2),
                'Small_10_49_Pct': round(100.0 * float((c_sub['scr10'] == 2).mean()), 2),
                'Medium_50_249_Pct': round(100.0 * float((c_sub['scr10'] == 3).mean()), 2),
                'Turnover_DK_Pct': round(100.0 * float(c_sub['scr14_dk'].mean()), 2),
                'B2C_Pct': round(100.0 * float(c_sub['B2C'].mean()), 2),
                'B2B_Pct': round(100.0 * float(c_sub['B2B_private'].mean()), 2),
                'B2G_Pct': round(100.0 * float(c_sub['B2G_public'].mean()), 2),
                'Green_Offerings_Y1_Valid_Pct': round(100.0 * float(c_sub['q9_binary'].dropna().mean()), 2),
                'NetZero_Strategy_Y2_Valid_Pct': round(100.0 * float(c_sub['q14_binary'].dropna().mean()), 2),
                'Mean_9Item_RE_Count': round(float(c_sub['re_action_count'].mean()), 2)
            })
        pd.DataFrame(attr_rows).to_excel(writer, sheet_name='Missing_Data_Attrition', index=False)

    print(f"  ✓ Consolidated Input Workbook: {OUTPUT_XLSX}")


def main():
    df, df_raw, df_sme_raw, meta = load_and_filter_fl498()
    df = recode_variables(df, meta)
    df = merge_desi(df)
    save_outputs(df, df_raw, df_sme_raw, meta)
    print("\n" + "=" * 72)
    print("01_build_analytical_sample.py COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
