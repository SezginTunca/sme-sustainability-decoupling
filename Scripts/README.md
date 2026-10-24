# Core Empirical Scripts: Data Processing, Methods, and Econometric Analyses

This directory contains the essential data processing, econometric modeling, and empirical analysis scripts for the study:

**"Market-Channel Orientation and Capability Constraints Shape Sustainability Decoupling in European Small and Medium-Sized Enterprises"**  
**Target Journal:** *Business Strategy and the Environment* (Wiley)  
**Authors:** Sezgin Tunca, Mausam Budhathoki, and Hari Gopal Risal  

> [!NOTE]
> This folder is dedicated to **data processing, econometric methods, and quantitative statistical analyses** for open-science replication.

---

## Quick Start & Environment Setup

All scripts require **Python 3.9+**. Install dependencies with `pip`:

```bash
pip install -r requirements.txt
```

### Essential Dependencies
- `pandas` (≥ 2.0.0): Data manipulation, aggregation, and tabular export
- `numpy` (≥ 1.24.0): Matrix algebra and vector computations
- `scipy` (≥ 1.10.0): Maximum likelihood optimization (BFGS), distributions, and asymptotic tests
- `statsmodels` (≥ 0.14.0): Robust covariance estimation (HC1), logit models, and delta-method gradients
- `scikit-learn` (≥ 1.2.0): Logistic propensity score modeling and feature handling
- `openpyxl` (≥ 3.1.0): Multi-sheet Excel workbook read/write engine
- `pyreadstat` (≥ 1.2.0): Lossless SPSS (`.sav`) file reader for Eurobarometer microdata

---

## Sequential Execution Workflow

Run the scripts in numerical order from the project root or the `Scripts/` directory:

```bash
# Step 1: Preprocess raw survey microdata, apply EU-27/SME filters, and merge Eurostat DESI macro data
python3 Scripts/01_build_analytical_sample.py

# Step 2: Estimate joint Bivariate Probit regressions, disturbance correlation (ρ), and Wald equality tests
python3 Scripts/02_bivariate_probit.py

# Step 3: Run Inverse Propensity Score Weighting (IPW-CATE) models for grant heterogeneity
python3 Scripts/03_causal_forest.py

# Step 4: Perform 1:1 PSM matching, Rosenbaum sensitivity analysis, OLS moderation, and Ordered Probit models
python3 Scripts/04_psm_and_remaining.py

# Step 5: Run econometric robustness checks (strict Q14, sector subsamples, survey weights, and probability-scale moderation)
python3 Scripts/05_robustness.py
```

*Total execution time: ~60–90 seconds on standard multi-core systems.*

---

## Script-by-Script Breakdown

### 1. [`01_build_analytical_sample.py`](01_build_analytical_sample.py)
- **Role**: Master data ingestion, filtering, and feature engineering pipeline.
- **Inputs**:
  - `Data/Raw/FL498_2021_ZA7806.sav` (Flash Eurobarometer 498, $N = 17,662$)
  - `Data/Raw/Eurostat_Digital_Intensity.tsv` (Eurostat DESI 2021 indicators)
- **Key Operations**:
  - Filters sample to EU-27 Member States ($N = 14,215$) and SMEs ($1\text{--}249$ employees, $N = 13,322$).
  - Aligns survey variables (`scr15.1` = B2C, `scr15.2` = B2B private, `scr15.3` = B2G public; `Q6.1` = public grants, `Q6.4` = public advice).
  - Recodes market-facing environmental action ($Y_1$, `Q9`: green products/services) and internal strategic action ($Y_2$, `Q14`: net-zero strategy).
  - Merges Eurostat national Digital Intensity Index (DII) indicators.
- **Outputs**:
  - `Data/Processed/SME_Analytical_Final.csv` ($N = 13,322 \times 92$ variables)
  - `Data/Input_Data_Consolidated.xlsx` (5-sheet consolidated input workbook)
- **Manuscript Mapping**: **Table 1**; Section 3.1; Appendix G.

---

### 2. [`02_bivariate_probit.py`](02_bivariate_probit.py)
- **Role**: Joint Bivariate Probit econometric estimation of market-facing green product offerings ($Y_1$) and reported net-zero decarbonization strategy ($Y_2$).
- **Methodology**:
  - Maximum likelihood BFGS estimation of joint bivariate normal disturbances.
  - Accounts for correlated unobserved factors through disturbance correlation ($\rho = 0.1870, p < 0.001$).
  - Cross-equation Wald tests ($z$-tests) to evaluate cross-equation asymmetry.
- **Inputs**: `Data/Processed/SME_Analytical_Final.csv`
- **Outputs**:
  - Appends to `Data/Results_Consolidated.xlsx`:
    - `BiProbit_Coefficients`: Probit coefficients, SEs, $p$-values, and average marginal effects (AMEs).
    - `BiProbit_CrossEq_Tests`: Cross-equation parameter differences and Wald test statistics.
- **Manuscript Mapping**: **Table 2** (Hypotheses 1a, 1b, 2a, 2b, 3a).

---

### 3. [`03_causal_forest.py`](03_causal_forest.py)
- **Role**: Inverse Propensity Score Weighting (IPW) and Conditional Average Treatment Effect (CATE) heterogeneity modeling.
- **Methodology**:
  - Logistic regression modeling of grant selection probability ($e(X)$) on $N = 11,521$ routed SMEs.
  - Stabilized IPW weighting (mean 1.00, max 3.07) for overall Average Treatment Effect ($ATE = 0.0346, p = 0.021$).
  - Non-parametric bootstrap (500 replications) for standard errors across administrative barriers, technical expertise deficits, firm size, and digital intensity.
- **Inputs**: `Data/Processed/SME_Analytical_Final.csv`
- **Outputs**:
  - Appends to `Data/Results_Consolidated.xlsx`:
    - `CausalForest_ATE`: Sample-wide treatment effect.
    - `CausalForest_CATEs`: Subgroup conditional estimates across capability constraints and enterprise size classes.
- **Manuscript Mapping**: **Figure 4**; Section 4.4; Appendix F (Hypothesis 3b).

---

### 4. [`04_psm_and_remaining.py`](04_psm_and_remaining.py)
- **Role**: 1:1 Nearest-Neighbor Propensity Score Matching (PSM), Rosenbaum sensitivity bounds, OLS digital moderation, and Ordered Probit operating cost modeling.
- **Methodology**:
  - PSM with strict caliper ($0.2 \times \text{SD}(\text{logit } e(X))$) producing 1,256 matched pairs (all $|\text{SMD}| \le 0.029$).
  - Rosenbaum sensitivity bounds testing robustness to unobserved hidden selection ($\Gamma^* \approx 1.11$).
  - OLS regression with clustered/robust standard errors testing B2B $\times$ DII digital moderation (Model 4).
  - Maximum-likelihood Ordered Probit testing operating cost impacts (Model 5: Q9 vs Q14).
- **Inputs**: `Data/Processed/SME_Analytical_Final.csv`
- **Outputs**:
  - Appends to `Data/Results_Consolidated.xlsx`:
    - `PSM_ATT`: Matched treatment effects on net-zero strategy and green product outcomes.
    - `PSM_Balance`: Covariate balance verification.
    - `DI_Moderation_OLS`: Country-level DII moderation estimates.
    - `DI_Country_Level`: Country-level mean Decoupling Index and DII metrics.
    - `GreenPremium_CrossTab`: Operating cost cross-tabulations.
- **Manuscript Mapping**: **Tables 3–4**; Appendices D, E, H, I (Hypothesis 3a and Supplementary Analyses).

---

### 5. [`05_robustness.py`](05_robustness.py)
- **Role**: Econometric robustness and probability-scale moderation testing.
- **Methodology**:
  1. *Strict Net-Zero Strategy Definition*: Re-estimates bivariate models excluding firms reporting existing climate neutrality ($N = 10,624$).
  2. *Sector Subsample Probits*: Re-estimates across Manufacturing, Services, Retail, and Industry/Construction sectors.
  3. *Survey-Weighted Probits*: Re-estimates incorporating sampling weights (`w1_sme`) to produce population-weighted estimates.
  4. *Delta-Method Probability-Scale Moderation (H3b)*: Computes predicted probabilities, conditional AMEs, and second differences ($\Delta\text{AME}$) with 95% confidence intervals for `Grant × Admin_Barrier` ($\Delta\text{AME} = +0.0711, p = 0.0097$) and `Grant × Expertise_Deficit` ($\Delta\text{AME} = -0.0055, p = 0.8423$).
- **Inputs**: `Data/Processed/SME_Analytical_Final.csv`, `Data/Results_Consolidated.xlsx`
- **Outputs**:
  - Appends to `Data/Results_Consolidated.xlsx`:
    - `Robustness_Sector`: Sector-by-sector estimates.
    - `Robustness_Weighted`: Survey-weighted probit coefficients (Appendix Table D1 / J1).
    - `AppendixJ_ProbScale_Moderation`: Probability-scale moderation parameters and second differences (Appendix Table J3).
- **Manuscript Mapping**: **Appendix J (Tables J1–J3)**; Appendix D; Appendix E; Appendix G.
