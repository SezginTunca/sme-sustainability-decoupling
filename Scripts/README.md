# Core Empirical Scripts: Data Processing, Methods, and Econometric Analyses

This directory contains the essential data processing, econometric modeling, and empirical analysis scripts for the study:

**"Market-Channel Orientation and Capability Constraints Shape Cross-Domain Sustainability Asymmetry in European Small and Medium-Sized Enterprises"**  
**Target Journal:** *Business Strategy and the Environment* (Wiley)  
**Authors:** Sezgin Tunca, Mausam Budhathoki, and Hari Gopal Risal  

> [!NOTE]
> This folder is dedicated to **data processing, econometric methods, and quantitative statistical analyses** for open-science replication. Figure and manuscript-building scripts live in `Manuscript_Build/` (`make_figures_joint_model.py` generates Figures 3, 5, S5, and S9 from `Data/Processed/BiProbit_Joint_FE.xlsx`; `make_figures_other.py` regenerates remaining figures) and are not part of the core replication pipeline.

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
- `statsmodels` (≥ 0.14.0): Robust covariance estimation (HC1), cluster-robust probit/logit models, and delta-method gradients
- `scikit-learn` (≥ 1.2.0): Logistic propensity score modeling and feature handling
- `openpyxl` (≥ 3.1.0): Multi-sheet Excel workbook read/write engine
- `pyreadstat` (≥ 1.2.0): Lossless SPSS (`.sav`) file reader for Eurobarometer microdata

---

## Sequential Execution Workflow

Run the 5 core replication scripts in numerical order from the project root or the `Scripts/` directory:

```bash
# Step 1: Preprocess raw survey microdata, apply EU-27/SME filters, and merge Eurostat DESI macro data (~10s)
python3 Scripts/01_build_analytical_sample.py

# Step 2: Estimate joint full-ML Bivariate Probit with country and sector FE, RECount, and clustered SEs (~3 min)
python3 Scripts/02_bivariate_probit.py

# Step 3: Run Inverse Propensity Score Weighting (IPW/CATE) models for public financial support heterogeneity (~15s)
python3 Scripts/03_causal_forest.py

# Step 4: Perform 1:1 PSM matching (Austin 2011 logit caliper), Rosenbaum sensitivity analysis, DI OLS moderation, and Ordered Probit (~20s)
python3 Scripts/04_psm_and_remaining.py

# Step 5: Run clustered probit robustness checks and delta-method probability-scale moderation (~25s)
python3 Scripts/05_robustness.py
```

*Total execution time: ~4 minutes on standard multi-core systems (Step 2 takes about 3 minutes).*

---

## Script-by-Script Breakdown

### 1. [`01_build_analytical_sample.py`](01_build_analytical_sample.py)

- **Role**: Master data ingestion, filtering, and feature engineering pipeline.
- **Inputs**:
  - `Data/Raw/FL498_2021_ZA7806.sav` (Flash Eurobarometer 498, $N = 17,662$)
  - `Data/Raw/Eurostat_Digital_Intensity.tsv` (Eurostat DESI 2021 indicators)
- **Key Operations**:
  - Filters sample to EU-27 Member States ($N = 14,215$) and SMEs ($1\text{--}249$ employees, $N = 13,322$).
  - Aligns survey variables (`scr15.1` = B2C, `scr15.2` = B2B private, `scr15.3` = B2G public; `Q6.1` = public financial support [grants, loans or guarantees], `Q6.4` = public advice).
  - Recodes market-facing environmental action ($Y_1$, `Q9`: green products/services) and internal strategic action ($Y_2$, `Q14`: net-zero strategy).
  - Constructs the Cross-Domain Sustainability Asymmetry Index ($DI = Y_1 - Y_2 \in \{-1, 0, +1\}$).
  - Merges Eurostat national Digital Intensity Index (DII) indicators.
- **Outputs**:
  - `Data/Processed/SME_Analytical_Final.csv` ($N = 13,322 \times 92$ variables)
  - `Data/Input_Data_Consolidated.xlsx` (5-sheet consolidated input workbook)
- **Runtime**: ~10 seconds.
- **Manuscript Mapping**: **Table 1**; Section 3.1; Appendix G.

---

### 2. [`02_bivariate_probit.py`](02_bivariate_probit.py) *(Main Joint Econometric Model)*

- **Role**: Full-information maximum likelihood (joint ML) bivariate probit estimation with 26 country fixed effects, sector fixed effects, resource-efficiency action count (RECount), and country-clustered sandwich standard errors (27 clusters).
- **Methodology & Key Operations**:
  - Joint ML estimation of correlated latent equations ($Y_1, Y_2$) using Plackett's identity with vectorised 24-point Gauss-Legendre quadrature for bivariate normal CDF $\Phi_2(\cdot, \cdot; \rho)$.
  - Direct estimation of cross-equation disturbance correlation ($\rho = 0.257, SE = 0.026, p < 0.001$).
  - Full joint sandwich covariance matrix clustered by country (27 EU Member State clusters).
  - Cross-equation Wald tests for coefficient equality ($\beta_{1k} = \beta_{2k}$) and discrete marginal effect equality ($\text{AME}_{1k} = \text{AME}_{2k}$) using the joint parameter covariance block, with Holm–Bonferroni multiplicity adjustments.
  - True discrete finite-difference Average Marginal Effects: $\text{AME}_k = \frac{1}{N}\sum_{i=1}^N \left[\Phi(\mathbf{X}_i^{(X_k=1)}\hat{\mathbf{\beta}}_j) - \Phi(\mathbf{X}_i^{(X_k=0)}\hat{\mathbf{\beta}}_j)\right]$ with numerical Jacobian delta-method standard errors.
- **Inputs**: `Data/Processed/SME_Analytical_Final.csv` ($N = 11,128$)
- **Outputs**:
  - `Data/Processed/BiProbit_Joint_FE.xlsx` (Sheets: `coef`, `cross_eq_tests`, `AME`, `diagnostics`)
  - `Data/Processed/BiProbit_Joint_FE.csv`
- **Runtime**: ~3 minutes.
- **Manuscript Mapping**: **Table 2** (Hypotheses 1a, 1b, 2a, 2b, 3a); Appendix H.

---

### 3. [`03_causal_forest.py`](03_causal_forest.py)

- **Role**: Inverse Propensity Score Weighting (IPW) and Conditional Average Treatment Effect (CATE) heterogeneity modeling for public financial support (grants, loans or guarantees).
- **Methodology & Key Operations**:
  - Logistic regression modeling of support selection probability ($e(X)$) on $N = 11,521$ routed SMEs (1,292 treated, 10,229 control; treated mean PS = 0.160), incorporating 26 country and sector fixed effects.
  - Stabilized IPW weighting (mean 1.00) for overall Average Treatment Effect ($ATE = 0.045$, $SE = 0.016$, 95% CI [0.014, 0.076], $p = 0.005$).
  - Non-parametric bootstrap (500 replications) for standard errors across administrative barriers (high: 0.091 vs none: 0.021; $\Delta = +0.070, SE = 0.031, p = 0.027$, heterogeneity ratio $\approx 4.3\times$), technical expertise deficits (deficit: 0.062 vs no deficit: 0.040), firm size (micro: 0.018, small: 0.065, medium: 0.063), and digital intensity (low DII: 0.041, high DII: 0.053).
- **Inputs**: `Data/Processed/SME_Analytical_Final.csv`
- **Outputs**:
  - Appends to `Data/Results_Consolidated.xlsx`:
    - `CausalForest_ATE`: Sample-wide treatment effect.
    - `CausalForest_CATEs`: Subgroup conditional estimates across capability constraints and enterprise size classes.
- **Runtime**: ~15 seconds.
- **Manuscript Mapping**: **Figure 4**; Section 4.4; Appendix F (Hypothesis 3b).

---

### 4. [`04_psm_and_remaining.py`](04_psm_and_remaining.py)

- **Role**: 1:1 Nearest-Neighbor Propensity Score Matching (PSM), Rosenbaum sensitivity bounds, OLS digital moderation, and Ordered Probit operating cost modeling.
- **Methodology & Key Operations**:
  - PSM with strict caliper ($0.2 \times \text{SD}(\text{logit } e(X))$) following Austin (2011) with 26 country and sector fixed effects in the propensity model, producing 1,256 matched pairs (all $|\text{SMD}| < 0.07$; B2G 0.065, B2C 0.029).
  - ATT on net-zero strategy: $\hat{\tau}_{\text{ATT}} = 0.051$ ($SE = 0.018, p = 0.005, 95\% \text{ CI } [0.016, 0.086]$); ATT on green offerings: $\hat{\tau}_{\text{ATT}} = 0.046$ ($SE = 0.019, p = 0.017, 95\% \text{ CI } [0.008, 0.084]$).
  - Rosenbaum sensitivity bounds testing robustness to unobserved hidden selection: critical threshold $\Gamma^* \approx 1.10$ ($p = 0.003$ at $\Gamma = 1.0$, $p = 0.049$ at $\Gamma = 1.10$, $p = 0.257$ at $\Gamma = 1.20$; sensitive to modest hidden bias).
  - OLS regression with country-clustered standard errors testing B2B $\times$ DII cross-level digital moderation on cross-domain sustainability asymmetry ($DI$, Model 4: $\beta = -0.010, p = 0.448$).
  - Maximum-likelihood Ordered Probit testing operating cost impacts (Model 5: green offerings only $\beta = +0.073, p = 0.003$ vs net-zero strategy $\beta = +0.0003, p = 0.990$).
- **Inputs**: `Data/Processed/SME_Analytical_Final.csv`
- **Outputs**:
  - Appends to `Data/Results_Consolidated.xlsx`:
    - `PSM_ATT`: Matched treatment effects on net-zero strategy and green product offerings.
    - `PSM_Balance`: Covariate balance verification.
    - `DI_Moderation_OLS`: Country-level DII moderation estimates.
    - `DI_Country_Level`: Country-level mean asymmetry index and DII metrics.
    - `GreenPremium_CrossTab`: Operating cost cross-tabulations.
- **Runtime**: ~20 seconds.
- **Manuscript Mapping**: **Tables 3–4**; Appendices D, E, H, I.

---

### 5. [`05_robustness.py`](05_robustness.py)

- **Role**: Clustered probit robustness specifications and probability-scale moderation testing under the corrected fixed-effects specification.
- **Methodology & Key Operations**:
  - Equation-by-equation probits with 26 country fixed effects, sector fixed effects, RECount, and country-clustered standard errors across:
    1. *Reference Specification*: Probits matching the baseline joint specification.
    2. *Models without RECount*: Testing sensitivity to excluding RECount as a potential post-treatment control.
    3. *Survey-Weighted Probits*: Applying official Eurobarometer post-stratification sampling weights (`w1_sme`) via cluster-robust GLM.
    4. *Routed-Only Subsample*: Restricting to external support seekers ($Q5.3 = 1$) to test sensitivity to survey filtering.
    5. *Strict Net-Zero Strategy Definition*: Excluding enterprises reporting existing climate neutrality ($N = 10,624$).
    6. *Sector-Stratified Probits*: Estimating within Manufacturing, Services, Retail, and Other sectors with within-sector country FE.
  - *Delta-Method Probability-Scale Moderation (H3b)*: Computes predicted probabilities, conditional AMEs, and second differences ($\Delta\text{AME}$) with country-clustered standard errors and 95% confidence intervals for `Support × Admin_Barrier` ($\Delta\text{AME} = +0.073, p = 0.001$) and `Support × Expertise_Deficit` ($\Delta\text{AME} = -0.006, p = 0.821$).
- **Inputs**: `Data/Processed/SME_Analytical_Final.csv`
- **Outputs**:
  - `Data/Processed/Robustness_FE.xlsx` (Sheets: `robustness`, `moderation_clustered`)
- **Runtime**: ~25 seconds.
- **Manuscript Mapping**: **Appendix L**; updates and supersedes earlier pre-revision tables in Appendices D, E, J.
