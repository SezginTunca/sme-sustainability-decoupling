# Replication Codebase: Market-Channel Orientation, Capability Constraints, and Sustainability Decoupling

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23619995.svg)](https://doi.org/10.5281/zenodo.23619995)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository provides the complete, self-contained empirical replication codebase for the study:

**"Market-Channel Orientation, Capability Constraints, and Sustainability Decoupling: Evidence from European Small and Medium-Sized Enterprises"**  
**Target Journal:** *Business Strategy and the Environment* (Wiley)  
**Authors:** Sezgin Tunca, Mausam Budhathoki, and Hari Gopal Risal  

---

## 📌 Repository Overview

This repository contains the full econometric estimation pipeline and data processing routines. In compliance with data sharing and open science protocols, this repository provides **the complete code files without proprietary raw survey microdata**, accompanied by exact data access instructions below.

```
├── README.md                            # Main project documentation and replication guide
├── requirements.txt                     # Core scientific computing dependencies
├── LICENSE                              # MIT License
├── .gitignore                           # Excludes all datasets and bytecode
├── .zenodo.json                         # Zenodo open-science metadata
│
└── Scripts/                             # Essential replication pipeline (01–05)
    ├── 01_build_analytical_sample.py    # Master data ingestion, filtering, and DESI merge
    ├── 02_bivariate_probit.py           # Joint Bivariate Probit regressions & Wald tests (Table 2)
    ├── 03_causal_forest.py              # IPW-CATE models for grant heterogeneity (Table 3)
    ├── 04_psm_and_remaining.py          # 1:1 PSM, Rosenbaum bounds, OLS moderation, Ordered Probits (Table 4)
    ├── 05_robustness.py                 # Pure econometric robustness & probability-scale moderation
    └── README.md                        # Script-by-script technical documentation
```

---

## 💾 Data Access & Preparation

To run the data processing pipeline from scratch:

1. **Flash Eurobarometer 498 (`ZA7806`):**
   - **Source:** GESIS Data Archive for the Social Sciences
   - **Study:** *Flash Eurobarometer 498: SMEs, resource efficiency and green markets, wave 5*
   - **DOI:** [10.4232/1.13941](https://doi.org/10.4232/1.13941)
   - **Action:** Download the primary SPSS file `FL498_2021_ZA7806.sav` (or `ZA7806_v1-0-0.sav`) and place it in `Data/Raw/`.

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

## 🚀 Quick Start & Environment Setup

All scripts require **Python 3.9+**. Install the lightweight dependency stack:

```bash
pip install -r requirements.txt
```

### Dependencies
- `pandas` (≥ 2.0.0) — High-throughput data manipulation
- `numpy` (≥ 1.24.0) — Vector and numerical linear algebra
- `scipy` (≥ 1.10.0) — Maximum likelihood optimization (BFGS) and asymptotic distribution tests
- `statsmodels` (≥ 0.14.0) — Robust covariance estimation (HC1) and delta-method gradients
- `scikit-learn` (≥ 1.2.0) — Logistic propensity score modeling
- `openpyxl` (≥ 3.1.0) — Multi-sheet Excel workbook export
- `pyreadstat` (≥ 1.2.0) — Lossless reading of raw Eurobarometer SPSS data (`.sav`)

---

## ⚙️ Sequential Replication Workflow

Run the scripts in numerical order from the project root:

```bash
# Step 1: Preprocess raw survey microdata, apply EU-27/SME filters, and merge Eurostat DESI macro data
python3 Scripts/01_build_analytical_sample.py

# Step 2: Estimate joint Bivariate Probit regressions, disturbance correlation (ρ), and Wald equality tests
python3 Scripts/02_bivariate_probit.py

# Step 3: Run Inverse Propensity Score Weighting (IPW-CATE) models for grant heterogeneity
python3 Scripts/03_causal_forest.py

# Step 4: Execute 1:1 PSM matching, Rosenbaum sensitivity analysis, OLS moderation, and Ordered Probit models
python3 Scripts/04_psm_and_remaining.py

# Step 5: Run econometric robustness checks (strict Q14, sector stratification, survey weights, and probability-scale moderation)
python3 Scripts/05_robustness.py
```

*Total execution time: ~60–90 seconds on standard multi-core hardware.*

---

## 📊 Summary of Hypotheses & Econometric Methods

| Script | Methodological Focus | Key Econometric Quantities | Manuscript Mapping |
| :--- | :--- | :--- | :--- |
| `01_build_analytical_sample.py` | Data Harmonization | EU-27 SME sample filtering ($N = 13,322 \times 92$ variables) | Data section |
| `02_bivariate_probit.py` | Joint Maximum Likelihood | Bivariate Probit with disturbance correlation ($\rho = 0.1870, p < 0.001$); cross-equation Wald $z$-tests | **Table 2** (H1a, H1b, H2a, H2b) |
| `03_causal_forest.py` | Propensity Score / IPW | Stabilized IPW weighting ($ATE = 0.0346, p = 0.021$); 1,000-rep bootstrap CATE heterogeneity | **Table 3**; Appendix F (H3b) |
| `04_psm_and_remaining.py` | Matching & Discrete Choice | 1:1 Caliper PSM ($ATT = 0.0525, p = 0.0041$); Rosenbaum bounds ($\Gamma^* \approx 1.11$); Ordered Probit costs | **Table 4**; Appendices D, E, I |
| `05_robustness.py` | Sensitivity & Moderation | Strict $Y_2$ recoding; sector splits; survey weights (`w1_sme`); delta-method second differences ($\Delta\text{AME} = +0.0711, p = 0.0097$) | **Appendix J (Tables J1, J3)** |

---

## 📜 Citation & License

This replication codebase is open-source under the **MIT License**.

When citing this research or utilizing these scripts, please cite:

```bibtex
@article{tunca2026customer,
  title={Market-Channel Orientation, Capability Constraints, and Sustainability Decoupling: Evidence from European Small and Medium-Sized Enterprises},
  author={Tunca, Sezgin and Budhathoki, Mausam and Risal, Hari Gopal},
  journal={Business Strategy and the Environment},
  year={2026},
  publisher={Wiley}
}
```
