# Replication Codebase: Market-Channel Orientation and Capability Constraints Shape Cross-Domain Sustainability Asymmetry in European Small and Medium-Sized Enterprises

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23619994.svg)](https://doi.org/10.5281/zenodo.23619994)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository provides the complete, self-contained empirical replication codebase for the study:

**"Market-Channel Orientation and Capability Constraints Shape Cross-Domain Sustainability Asymmetry in European Small and Medium-Sized Enterprises"**  
**Target Journal:** *Business Strategy and the Environment* (Wiley)  
**Authors:** Sezgin Tunca, Mausam Budhathoki, and Hari Gopal Risal  

---

## Repository Overview

This repository contains the full econometric estimation pipeline and data processing routines. In compliance with data sharing and open science protocols, this repository provides **the complete code files without proprietary raw survey microdata**, accompanied by exact data access instructions below.

```
├── README.md                            # Main project documentation and replication guide
├── requirements.txt                     # Core scientific computing dependencies
├── LICENSE                              # MIT License
├── .gitignore                           # Excludes all datasets and bytecode
├── .zenodo.json                         # Zenodo open-science metadata
│
└── Scripts/                             # Essential replication pipeline (01–06)
    ├── 01_build_analytical_sample.py    # Master data ingestion, filtering, and DESI merge
    ├── 02_bivariate_probit.py           # Joint full-ML bivariate probit, FE, clustered SEs, AMEs (Table 2)
    ├── 03_causal_forest.py              # IPW-CATE models for public-financial-support heterogeneity (Section 4.4, Fig. 4)
    ├── 04_psm_and_remaining.py          # 1:1 PSM, Rosenbaum bounds, OLS moderation, Ordered Probits (Tables 3–4)
    ├── 05_robustness.py                 # Clustered robustness checks & probability-scale moderation (Appendix L)
    ├── 06_verify_all_results.py         # Automated verification suite (31 end-to-end econometric tests)
    └── README.md                        # Script-by-script technical documentation
```

---

## Data Access & Preparation

To run the data processing pipeline from scratch:

1. **Flash Eurobarometer 498 (`ZA7806`) [Analyzed Wave]:**
   - **Source:** GESIS Data Archive for the Social Sciences
   - **Study:** *Flash Eurobarometer 498: SMEs, resource efficiency and green markets, wave 5*
   - **GESIS Study ID:** ZA7806
   - **GESIS URL:** [https://search.gesis.org/research_data/ZA7806](https://search.gesis.org/research_data/ZA7806)
   - **DOI:** [10.4232/1.13934](https://doi.org/10.4232/1.13934)
   - **Dataset Version:** 1.0.0 (2022)
   - **Action:** Download the primary SPSS file `FL498_2021_ZA7806.sav` (or `ZA7806_v1-0-0.sav`) and place it in `Data/Raw/`.
   - **Note on Subsequent Wave (Wave 6):** The subsequent 2024 wave of this survey series is *Flash Eurobarometer 549* (GESIS Study ID: `ZA8869`, DOI: [10.4232/1.14493](https://doi.org/10.4232/1.14493), URL: [https://search.gesis.org/research_data/ZA8869](https://search.gesis.org/research_data/ZA8869)). The empirical analyses, sample filtering ($N = 17,662 \to 13,322$), and econometric models in this replication repository specifically draw on Wave 5 (`ZA7806`).

2. **Eurostat Digital Intensity Index (DESI):**
   - **Source:** European Commission / Eurostat Database
   - **Dataset:** *Enterprises with high/very high digital intensity index* (`isoc_e_dii`)
   - **Action:** Save as `Eurostat_Digital_Intensity.tsv` and place in `Data/Raw/`.

*Directory layout for raw inputs:*
```
Data/
└── Raw/
    ├── FL498_2021_ZA7806.sav
    └── Eurostat_Digital_Intensity.tsv
```

---

## Quick Start & Environment Setup

All scripts require **Python 3.9+**. Install the lightweight dependency stack:

```bash
pip install -r requirements.txt
```

### Dependencies
- `pandas` (≥ 2.0.0): High-throughput data manipulation
- `numpy` (≥ 1.24.0): Vector and numerical linear algebra
- `scipy` (≥ 1.10.0): Maximum likelihood optimization (BFGS) and asymptotic distribution tests
- `statsmodels` (≥ 0.14.0): Probit/logit models, cluster-robust covariance
- `patsy` (≥ 0.5.3): Design matrices for the moderation models
- `scikit-learn` (≥ 1.2.0): Logistic propensity score modeling
- `openpyxl` (≥ 3.1.0): Multi-sheet Excel workbook export
- `pyreadstat` (≥ 1.2.0): Lossless reading of raw Eurobarometer SPSS data (`.sav`)

---

## Sequential Replication Workflow

Run the scripts in numerical order from the project root:

```bash
# Step 1: Preprocess raw survey microdata, apply EU-27/SME filters, and merge Eurostat DESI macro data
python3 Scripts/01_build_analytical_sample.py

# Step 2: Estimate the joint bivariate probit (country + sector FE, RECount, country-clustered SEs), true AMEs, and cross-equation Wald tests (~3 min)
python3 Scripts/02_bivariate_probit.py

# Step 3: Run Inverse Propensity Score Weighting (IPW-CATE) models for public-financial-support heterogeneity
python3 Scripts/03_causal_forest.py

# Step 4: Execute 1:1 PSM matching, Rosenbaum sensitivity analysis, OLS moderation, and Ordered Probit models
python3 Scripts/04_psm_and_remaining.py

# Step 5: Run clustered robustness checks (no RECount, survey weights, routed-only, strict Q14, sector subsamples) and probability-scale moderation
python3 Scripts/05_robustness.py

# Step 6: Execute automated econometric verification suite (31 end-to-end checks)
python3 Scripts/06_verify_all_results.py
```

*Total execution time: about 4 minutes on a standard laptop (step 2 dominates). Manuscript figures are built separately and are not part of this pipeline.*

---

## Summary of Hypotheses & Econometric Methods

| Script | Methodological Focus | Key Econometric Quantities | Manuscript Mapping |
| :--- | :--- | :--- | :--- |
| `01_build_analytical_sample.py` | Data Harmonization | EU-27 SME sample filtering ($N = 13,322 \times 92$ variables) | **Table 1**; Section 3.1 |
| `02_bivariate_probit.py` | Joint Maximum Likelihood | Bivariate probit with 26 country + sector FE and RECount; country-clustered SEs (27 clusters); disturbance correlation ($\rho = 0.257$); true average marginal effects; cross-equation Wald tests on the joint covariance; Holm adjustment | **Table 2** (H1a, H1b, H2a, H2b, H3a) |
| `03_causal_forest.py` | Propensity Score / IPW | Stabilized IPW with country + sector FE in the propensity model ($ATE = 0.045$, bootstrap SE 0.015, cluster SE 0.020); CATE heterogeneity across barriers, size and digital intensity | **Figure 4**; Section 4.4; Appendix F |
| `04_psm_and_remaining.py` | Matching & Discrete Choice | 1:1 PSM with a 0.2 SD logit-scale caliper and FE in the propensity model ($ATT = 0.050$, $p = 0.006$ on $Y_2$; $ATT = 0.047$, $p = 0.015$ on $Y_1$); Rosenbaum bounds ($\Gamma^* \approx 1.10$); asymmetry-index OLS moderation; ordered probit costs | **Tables 3–4**; Appendices D, E, H, I |
| `05_robustness.py` | Sensitivity & Moderation | Clustered probits without RECount, survey-weighted, routed-only, strict $Y_2$, sector-stratified; cluster-robust delta-method second difference ($\Delta\text{AME} = +0.073$, $p = 0.001$) | **Appendix L** |
| `06_verify_all_results.py` | Automated Verification | 31 end-to-end tests validating FIML convergence, survey routing, sample sizes, and parameter bounds | **Quality Assurance** (`31 PASSED`) |

---

## Citation & License

This replication codebase is open-source under the **MIT License**.

When citing this research or using these scripts, please cite:

```bibtex
@article{tunca2026market,
  title={Market-Channel Orientation and Capability Constraints Shape Cross-Domain Sustainability Asymmetry in European Small and Medium-Sized Enterprises},
  author={Tunca, Sezgin and Budhathoki, Mausam and Risal, Hari Gopal},
  journal={Business Strategy and the Environment},
  year={2026},
  publisher={Wiley}
}
```
