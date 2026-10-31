# Core Empirical Scripts: Data Processing, Methods, and Econometric Analyses

This directory contains the essential data processing, econometric modeling, empirical analysis, and automated verification scripts for the study:

**"Green Products, Net-Zero Plans: Market Channels and Finance in European SMEs"**  
**Target Journal:** *Business Strategy and the Environment* (Wiley)  
**Authors:** Sezgin Tunca, Mausam Budhathoki, and Hari Gopal Risal  

> [!NOTE]
> This folder is dedicated to **data processing, econometric methods, quantitative statistical analyses, and automated verification** for open-science replication. Figure and manuscript-building scripts live in `Manuscript_Build/` (`make_figures_joint_model.py` generates Figures 3, 5, S5, and S9 from `Data/Processed/BiProbit_Joint_FE.xlsx`; `make_figures_other.py` regenerates remaining figures) and are not part of the core replication pipeline.

---

## Quick Start & Environment Setup

All scripts require **Python 3.9+**. Install dependencies with `pip`:

```bash
pip install -r requirements.txt
```

### Essential Dependencies

- `pandas` (≥ 2.0.0): Data manipulation, aggregation, and tabular export
- `numpy` (≥ 1.24.0): Matrix algebra and vector computations
- `scipy` (≥ 1.10.0): Maximum likelihood optimization (BFGS), distributions, and asymptotic/finite-cluster tests
- `statsmodels` (≥ 0.14.0): Robust covariance estimation (HC1), cluster-robust probit/logit/multinomial logit models, and delta-method gradients
- `scikit-learn` (≥ 1.2.0): Feature handling and baseline model utilities
- `openpyxl` (≥ 3.1.0): Multi-sheet Excel workbook read/write engine
- `pyreadstat` (≥ 1.2.0): Lossless SPSS (`.sav`) file reader for Eurobarometer microdata

---

## Sequential Execution Workflow

Run the 6 core replication and verification scripts in numerical order from the project root or the `Scripts/` directory:

```bash
# Step 1: Preprocess raw survey microdata, apply EU-27/SME filters, construct validation variables, and merge Eurostat DESI macro data (~10s)
python3 Scripts/01_build_analytical_sample.py

# Step 2: Estimate joint full-ML Bivariate Probit with country and sector FE, RECount, clustered SEs, and finite-cluster t26 inference (~3 min)
python3 Scripts/02_bivariate_probit.py

# Step 3: Run Newton-Raphson Inverse Propensity Score Weighting (IPW/AIPW/CATE) models for public financial support (~15s)
python3 Scripts/03_causal_forest.py

# Step 4: Perform 1:1 PSM matching (Austin 2011 logit caliper), Rosenbaum sensitivity analysis, DI OLS moderation, and Ordered Probit (~20s)
python3 Scripts/04_psm_and_remaining.py

# Step 5: Run clustered probit robustness checks, 4-quadrant Multinomial Logit AMEs, LPM, and delta-method probability-scale moderation (~25s)
python3 Scripts/05_robustness.py

# Step 6: Execute 32 automated verification assertions across all data workbooks and manuscript documents (~5s)
python3 Scripts/06_verify_all_results.py
```

*Total execution time: ~4 minutes on standard multi-core systems (Step 2 takes about 3 minutes).*

---

## Script-by-Script Breakdown

### 1. [`01_build_analytical_sample.py`](01_build_analytical_sample.py)

- **Role**: Master data ingestion, filtering, within-survey construct validation, attrition analysis, and feature engineering pipeline.
- **Inputs**:
  - `Data/Raw/FL498_2021_ZA7806.sav` (Flash Eurobarometer 498, $N = 17,662$)
  - `Data/Raw/Eurostat_Digital_Intensity.tsv` (Eurostat DESI 2021 indicators)
- **Key Operations**:
  - Filters sample to EU-27 Member States ($N = 14,215$) and SMEs ($1\text{--}249$ employees, $N = 13,322$).
  - Aligns survey variables (`scr15.1` = B2C, `scr15.2` = B2B private, `scr15.3` = B2G public; `Q6.1` = public financial support [grants, loans or guarantees], `Q6.4` = public advice).
  - Recodes market-facing environmental offerings ($Y_1$, `Q9`: green products/services) and organization-level climate planning ($Y_2$, `Q14`: net-zero strategy).
  - Constructs the Four-Quadrant Sustainability Typology (`Quadrant_4`) and Cross-Domain Sustainability Asymmetry Index ($DI = Y_1 - Y_2 \in \{-1, 0, +1\}$).
  - Extracts within-survey construct validation indicators (`SCR16` green revenue share, `DX5` environmental roles, `Q4` investment share, `Q10` green offering types, `Q15` carbon strategy scope) and sample attrition comparison (`Table G2`, $N = 11,128$ vs. $2,194$ excluded).
  - Merges Eurostat national Digital Intensity Index (DII) indicators.
- **Outputs**:
  - `Data/Processed/SME_Analytical_Final.csv` ($N = 13,322$)
  - `Data/Processed/Construct_Validity_and_Attrition.xlsx` (Sheets: `Construct_Validity_Table1`, `Attrition_TableG2`)
  - `Data/Input_Data_Consolidated.xlsx` (5-sheet consolidated input workbook)
- **Runtime**: ~10 seconds.
- **Manuscript Mapping**: **Table 1**; Section 3.1; Section 3.2; Appendix G (`Tables G1–G2`).

---

### 2. [`02_bivariate_probit.py`](02_bivariate_probit.py) *(Main Joint Econometric Model)*

- **Role**: Full-information maximum likelihood (joint ML) bivariate probit estimation with 26 country fixed effects, sector fixed effects, resource-efficiency action count (RECount), country-clustered sandwich standard errors ($G = 27$ clusters), and finite-cluster $t_{26}$ degrees-of-freedom inference.
- **Methodology & Key Operations**:
  - Joint ML estimation of correlated latent equations ($Y_1, Y_2$) using Plackett's identity with vectorised 24-point Gauss-Legendre quadrature (verified against 48-point quadrature to $1.38 \times 10^{-7}$).
  - Direct estimation of cross-equation disturbance correlation ($\hat{\rho} = 0.257, SE = 0.026, z = 9.81, p < 0.001, 95\%\text{ CI } [0.206, 0.308]$).
  - Full joint sandwich covariance matrix clustered by country ($G = 27$ EU Member State clusters) with both asymptotic normal and finite-cluster $t_{G-1} = t_{26}$ $p$-values.
  - Cross-equation Wald tests for coefficient equality ($\beta_{1k} = \beta_{2k}$) and discrete marginal effect equality ($\text{AME}_{1k} = \text{AME}_{2k}$) using the joint parameter covariance block, with Holm-Bonferroni multiplicity adjustments across 9 covariates.
  - True discrete finite-difference Average Marginal Effects: $\text{AME}_k = \frac{1}{N}\sum_{i=1}^N \left[\Phi(\mathbf{X}_i^{(X_k=1)}\hat{\mathbf{\beta}}_j) - \Phi(\mathbf{X}_i^{(X_k=0)}\hat{\mathbf{\beta}}_j)\right]$ with numerical Jacobian delta-method standard errors.
- **Inputs**: `Data/Processed/SME_Analytical_Final.csv` ($N = 11,128$)
- **Outputs**:
  - `Data/Processed/BiProbit_Joint_FE.xlsx` (Sheets: `coef`, `cross_eq_tests`, `AME`, `diagnostics`)
  - `Data/Processed/BiProbit_Joint_FE.csv`
- **Runtime**: ~3 minutes.
- **Manuscript Mapping**: **Table 2** (Hypotheses `H1a`, `H1b`, `H1c`, `H2a`, `H2b`, `H3a`, `H3b`); Appendix H; Appendix L (`Tables L1, L2, L6`).

---

### 3. [`03_causal_forest.py`](03_causal_forest.py)

- **Role**: Inverse Propensity Score Weighting (IPW), Doubly Robust Augmented IPW (AIPW), and Conditional Average Treatment Effect (CATE) heterogeneity modeling for public financial support (`Q6.1`).
- **Methodology & Key Operations**:
  - Exact Newton-Raphson maximum likelihood logistic regression (`fit_logit_newton`) modeling support selection probability ($e(X)$) on $N = 11,521$ routed SMEs (1,292 treated, 10,229 control; treated mean PS = 0.160), incorporating 26 country and sector fixed effects.
  - Stabilized IPW weighting (mean 1.00, Kish $ESS_T = 879.8$, $ESS_C = 10,143.4$) for overall Average Treatment Effect ($\widehat{\text{ATE}} = +0.045$, re-estimated PS bootstrap $\text{SE} = 0.015, p = 0.003$; country-clustered influence-function $\text{SE} = 0.020, p = 0.022$) and doubly robust AIPW ($\widehat{\text{ATE}} = +0.044, \text{SE} = 0.015, p = 0.004$).
  - Two-stage re-estimated propensity-score bootstrap (500 replications) for subgroup CATEs across administrative barriers (constrained: $+0.088$ vs. unconstrained: $+0.012$; $\Delta\widehat{\text{CATE}} = +0.076, \text{SE}_{\text{reest}} = 0.032, p = 0.017$), technical expertise deficits (deficit: $+0.056$ vs. no deficit: $+0.041$; $\Delta\widehat{\text{CATE}} = +0.015, p = 0.654$), firm size, and digital intensity.
- **Inputs**: `Data/Processed/SME_Analytical_Final.csv`
- **Outputs**:
  - `Data/Processed/IPW_FE_Diagnostics.xlsx`
  - Appends `CausalForest_ATE` and `CausalForest_CATEs` to `Data/Results_Consolidated.xlsx`.
- **Runtime**: ~15 seconds.
- **Manuscript Mapping**: **Figure 4**; Section 4.4; Appendix F; Appendix I (`Tables I1–I2`).

---

### 4. [`04_psm_and_remaining.py`](04_psm_and_remaining.py)

- **Role**: 1:1 Nearest-Neighbor Propensity Score Matching (PSM), Rosenbaum sensitivity bounds, OLS digital moderation (`Model 4`), and Ordered Probit operating cost modeling (`Model 5`).
- **Methodology & Key Operations**:
  - PSM with strict caliper ($0.2 \times \text{SD}(\text{logit } e(X))$) following Austin (2011) with 26 country and sector fixed effects in the propensity model, producing 1,256 matched pairs (all $|\text{SMD}| < 0.07$; B2G $-0.065$, B2C $-0.029$).
  - ATT on net-zero strategy: $\hat{\tau}_{\text{ATT}} = +0.050$ ($SE = 0.018, p = 0.006, 95\% \text{ CI } [+0.015, +0.086]$); ATT on green offerings: $\hat{\tau}_{\text{ATT}} = +0.047$ ($SE = 0.019, p = 0.015, 95\% \text{ CI } [+0.009, +0.085]$).
  - Rosenbaum sensitivity bounds testing robustness to unobserved hidden selection ($n_+ = 327, n_- = 264, n_d = 591$): critical threshold $\Gamma^*_{\text{exact}} = 1.097 \approx 1.10$ ($p_{\text{exact}} = 0.0033$ at $\Gamma = 1.0$, $p_{\text{exact}} = 0.0518$ at $\Gamma = 1.10$, $p_{\text{exact}} = 0.2649$ at $\Gamma = 1.20$) and normal approximation $\Gamma^*_{\text{norm}} \approx 1.08$ ($p_{\text{norm}} = 0.0487$ at $\Gamma = 1.10$; sensitive to modest hidden bias).
  - OLS regression with 27-country clustered standard errors testing $\text{B2B} \times \text{DII}$ cross-level digital moderation on $DI$ (`Model 4`: $\hat{\beta}_3 = -0.010, \text{SE}_{\text{cluster}} = 0.017, p = 0.570, t_{26}\ p = 0.575$; $\text{SE}_{\text{HC1}} = 0.013, p = 0.448$).
  - Maximum-likelihood Ordered Probit testing operating cost impacts (`Model 5` across $N = 10,011$: green offerings only $\hat{\beta}_1 = +0.071, p = 0.006$ vs. net-zero strategy $\hat{\beta}_2 = -0.006, p = 0.871$; and across $N = 9,058$ controlling for capabilities: $\hat{\beta}_1 = +0.056, p = 0.044$ vs. $\hat{\beta}_2 = +0.009, p = 0.831$).
- **Inputs**: `Data/Processed/SME_Analytical_Final.csv`
- **Outputs**:
  - `Data/Processed/PSM_Balance_FE.xlsx`, `Data/Processed/PSM_Sensitivity_FE.xlsx`
  - Appends `PSM_ATT`, `PSM_Balance`, `DI_Moderation_OLS`, `DI_Country_Level`, `GreenPremium_CrossTab`, and `OrderedProbit_FE` to `Data/Results_Consolidated.xlsx`.
- **Runtime**: ~20 seconds.
- **Manuscript Mapping**: **Tables 4–5**; Section 4.4; Section 4.6; Appendices D, E, H, I, L (`Table L5`).

---

### 5. [`05_robustness.py`](05_robustness.py)

- **Role**: Clustered probit robustness specifications, Four-Quadrant Multinomial Logit discrete marginal effects (`Table 3`), Linear Probability Model (`Table 6`), and probability-scale moderation testing (`H3c`).
- **Methodology & Key Operations**:
  - Equation-by-equation probits and LPMs with 26 country fixed effects, sector fixed effects, RECount, and country-clustered standard errors across:
    1. *Reference Specification*: Probits matching the baseline joint specification.
    2. *Models without RECount*: Testing sensitivity to excluding RECount as a potential post-treatment control.
    3. *Survey-Weighted Probits*: Applying official Eurobarometer sampling weights (`w1_sme` and `w93_sme`) via Binder (1983) pseudo-maximum likelihood (`Table L3`).
    4. *Routed-Only Subsample*: Restricting to external support seekers (`Q5.3 = 1`, $N = 3,055$ and $N = 2,959$) to test sensitivity to survey filtering (`Table J2`).
    5. *Strict Net-Zero Strategy Definition*: Excluding enterprises reporting existing climate neutrality ($N = 10,624$).
    6. *Sector-Stratified Probits*: Estimating within Manufacturing, Services, Retail, and Industry sectors with within-sector country FE.
    7. *Four-Quadrant Multinomial Logit (`mnlogit_4quadrants`)*: Estimating log-odds coefficients (`Table L4`) and discrete sample-average marginal effects across all four quadrants (`Table 3`).
    8. *Country-Clustered Linear Probability Model (`LPM_FE`)*: Estimating cross-equation differences via OLS (`Table 6`).
  - *Delta-Method Probability-Scale Moderation (`H3c`, `Table J3`)*: Computes predicted probabilities, conditional AMEs, and second differences ($\Delta\text{AME}$) with country-clustered standard errors and 95% confidence intervals for `Support × Admin_Barrier` ($\text{AME}(\text{Support}\mid\text{Admin}=1) = +0.085, p < 0.001$ vs. $+0.012, p = 0.545$; $\Delta\text{AME}_{\text{Admin}} = +0.073, \text{SE}_{\text{cl}} = 0.022, z = 3.28, p = 0.0010, t_{26}\ p = 0.0029$) and `Support × Expertise_Deficit` ($\Delta\text{AME}_{\text{Exp}} = -0.006, \text{SE}_{\text{cl}} = 0.026, p = 0.821$).
- **Inputs**: `Data/Processed/SME_Analytical_Final.csv`
- **Outputs**:
  - `Data/Processed/Robustness_FE.xlsx` (Sheets: `robustness`, `moderation_clustered`, `weighted_comparison`, `mnlogit_4quadrants`, `mnlogit_4quadrants_AME`, `LPM_FE`)
- **Runtime**: ~25 seconds.
- **Manuscript Mapping**: **Tables 3 and 6**; Section 4.3; Section 4.4; Section 4.5; Appendices J and L.

---

### 6. [`06_verify_all_results.py`](06_verify_all_results.py)

- **Role**: Independent automated co-author verification script.
- **Methodology & Key Operations**:
  - Executes 32 programmatic verification checks across `Data/Processed/BiProbit_Joint_FE.xlsx`, `Data/Processed/Robustness_FE.xlsx`, `Data/Processed/PSM_Sensitivity_FE.xlsx`, `Data/Processed/IPW_FE_Diagnostics.xlsx`, `Data/Processed/Construct_Validity_and_Attrition.xlsx`, `Data/Results_Consolidated.xlsx`, and the markdown manuscript files.
  - Verifies sample accounting ($N = 13,322 \to 12,205 \to 12,091 \to 11,521 \to 11,128$), bivariate probit convergence diagnostics, all eight hypothesis test statistics (`H1a`–`H3c`), 4-quadrant MNLogit AMEs, IPW/AIPW/PSM estimates, Rosenbaum bounds, and zero em dashes (`—`) in the manuscript prose.
- **Runtime**: ~5 seconds (returns exit code `0` when all 32 checks pass).

