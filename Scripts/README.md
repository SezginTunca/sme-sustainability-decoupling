# Core Empirical Scripts: Data Processing, Econometric Models, and Automated Verification

This directory contains the reproducible econometric estimation, document compilation, and automated verification scripts for:

**"Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs"**  
**Target Journal:** *Business Strategy and the Environment* (Wiley)  
**Authors:** Sezgin Tunca, Mausam Budhathoki, and Hari Gopal Risal  
**Canonical Zenodo Concept DOI:** [10.5281/zenodo.23619994](https://doi.org/10.5281/zenodo.23619994)  

---

## Quick Start

From the project root, run the complete SHA-256 audited replication and verification pipeline in one command:

```bash
pip install -r requirements.txt
python3 run_all_analysis.py
python3 generate_all_figures.py
```

Or run each script individually in numerical order:

```bash
python3 Scripts/01_build_analytical_sample.py
python3 Scripts/02_bivariate_probit.py
python3 Scripts/03_causal_forest.py
python3 Scripts/04_psm_and_remaining.py
python3 Scripts/05_robustness.py
python3 Scripts/06_verify_all_results.py
```

---

## Script-by-Script Breakdown

### 1. `01_build_analytical_sample.py`
- **Role:** Raw GESIS `ZA7806` ($N = 17,662$) ingestion, EU-27 ($N = 14,215$) and SME ($N = 13,322$) filtering, strictly 9-item (`re_action_count`, `Q1.1`-`Q1.9`) and 10-item (`re_action_count_10`) operational counts, categorical annual turnover (`scr14_cat` retaining `scr14 = 10` DK/refusal as a distinct category), within-survey behavioral construct validation (`SCR16`, `DX5`, `Q4`, `Q10`, `Q15`, `Q8`), Eurostat `isoc_e_dii` merge, and 253-variable codebook generation.
- **Outputs:** `Data/Processed/SME_Analytical_Final.csv`, `Data/Input_Data_Consolidated.xlsx` (8 sheets), `Data/Documentation/Codebook_ZA7806_Raw_and_Cleaned.xlsx` (3 sheets), and waterfall/typology sheets in `Data/Results_Consolidated.xlsx`.
- **Manuscript Mapping:** **Table 1**; Section 3.1; Section 3.2; Appendix G (`Tables G1-G2`).

### 2. `02_bivariate_probit.py`
- **Role:** Full-information maximum likelihood (FIML) bivariate probit (`Model 1`, $N = 11,128, G = 27$ clusters, $k = 48$ parameters per equation, $97$ total parameters) with categorical turnover (`scr14_cat`), 26 country FE, 3 sector FE, 9-item `RECount`, analytical gradients, 24-pt vs. 48-pt Gauss-Legendre verification ($4.44 \times 10^{-16}$), country-clustered sandwich covariance, finite-cluster $t_{26}$ $p$-values, discrete sample-average marginal effects (AMEs), joint FE Wald tests, and Holm-Bonferroni multiplicity adjustments across the $m = 6$ prespecified cross-equation hypotheses ($\hat{\rho} = +0.2551, \text{SE} = 0.0252, \ln\mathcal{L} = -12,930.5538$).
- **Outputs:** `Data/Processed/BiProbit_Joint_FE.csv`, `Data/Processed/BiProbit_Joint_FE.xlsx` (5 sheets), and `BiProbit_*` sheets in `Data/Results_Consolidated.xlsx`.
- **Manuscript Mapping:** **Table 2** (`H1a`-`H3b`); **Figures 3, 5, S5, S9**; Appendix H; Appendix L (`Tables L1-L3`).

### 3. `03_causal_forest.py`
- **Role:** Exact Newton-Raphson logistic propensity model ($N = 11,521, k = 46, \|\nabla\ell\|_\infty = 5.68 \times 10^{-14}$), Hajek stabilized IPW ($\widehat{\text{ATE}} = +0.0433$, re-estimated PS bootstrap $\text{SE} = 0.0148, p = 0.0036$; 27-country clustered influence-function $\text{SE} = 0.0195, p = 0.0267$; Kish $\text{ESS}_T = 876.1$), and subgroup CATE heterogeneity ($\Delta\widehat{\text{CATE}}_{\text{Admin}} = +0.0732, \text{SE} = 0.0319, p = 0.0220$).
- **Outputs:** `Data/Processed/IPW_Propensity_Scores.csv` and `CausalForest_ATE`, `CausalForest_CATEs`, `IPW_Diagnostics` in `Data/Results_Consolidated.xlsx`.
- **Manuscript Mapping:** **Figure 4**, **Figures S3, S6**; Section 4.4; Appendix F; Appendix I (`Tables I1-I2`).

### 4. `04_psm_and_remaining.py`
- **Role:** Order-invariant optimal 1:1 bipartite logit-caliper matching (`scipy.optimize.linear_sum_assignment`, $0.2 \times \text{SD}$ caliper $= 0.1617$, $1,256$ matched pairs, $\max |\text{SMD}| = 0.0579$, $\widehat{\text{ATT}}_{Y_2} = +0.0398$, paired $\text{SE} = 0.0190, p = 0.0363$, cluster $\text{SE} = 0.0306, p = 0.1930$; $\widehat{\text{ATT}}_{Y_1} = +0.0231, p = 0.2320$), 13-specification matching/AIPW sensitivity suite (AIPW $\widehat{\text{ATE}}_{Y_2} = +0.0413, p = 0.0056, p_{\text{cl}} = 0.0334$), exact and normal-approximation Rosenbaum bounds across three samples ($\Gamma^* = 1.0345$ on $N = 11,128$; $\Gamma^* = 1.1024$ on $N = 9,764$; $\Gamma^* = 1.0877$ on $N = 11,521$), `Model 4` cross-level DII OLS ($N = 12,205, \hat{\beta}_3 = -0.0091, p_{\text{cl}} = 0.5980$), and `Model 5` 5-spec operating-cost Ordered Probit ($N = 10,011$ and $N = 9,058$).
- **Outputs:** `PSM_ATT`, `PSM_Balance`, `PSM_Sensitivity_Specs`, `Rosenbaum_Bounds`, `DI_Moderation_OLS`, `DI_Country_Level`, `GreenPremium_CrossTab`, and `GreenPremium_OrderedProbit` sheets in `Data/Results_Consolidated.xlsx`.
- **Manuscript Mapping:** **Tables 4-5**; Section 4.4; Section 4.6; Appendices D, E, H, I, L (`Table L6`).

### 5. `05_robustness.py`
- **Role:** Four-Quadrant Multinomial Logit discrete AMEs (`Table 3`, $N = 11,128$), 12-specification joint bivariate probit robustness matrix (`Table 6`: baseline, no `RECount`, 10-item `RECount`, `w1_sme` and `w93_sme` PML, strict 2030 planners $N = 10,624$, `Q5.3=1` support seekers $N = 2,959 / 3,055$, `SCR16` product-selling SMEs $N = 7,299$, expanded covariates $N = 9,857$, 12-section NACE `nace_a` FE, complete-case turnover $N = 9,764$, mean-imputed turnover + DK dummy, plus sector splits), cluster-robust delta-method probability-scale moderation (`H3c`: $\Delta\text{AME}_{\text{Admin}} = +0.0730, \text{SE}_{\text{cl}} = 0.0223, p = 0.0011, t_{26}\ p = 0.0030$), and **Appendix M** extensions (`Ext1` tightened constructs, `Ext2` `Q8` desired support, `Ext3` 2024 Wave 6 `ZA8869` out-of-sample replication $N = 11,246$).
- **Outputs:** `Data/Processed/Robustness_FE.xlsx` (9 sheets) and corresponding sheets in `Data/Results_Consolidated.xlsx`.
- **Manuscript Mapping:** **Tables 3 and 6**; Section 4.3; Section 4.4; Section 4.5; Section 4.6; Appendices B, C, J, L, M.

### 6. `06_verify_all_results.py`
- **Role:** Automated end-to-end verification suite executing **195+ programmatic assertions** across `Data/Results_Consolidated.xlsx` (all 33 sheets), `Data/Input_Data_Consolidated.xlsx` (all 8 sheets), `Data/Documentation/Codebook_ZA7806_Raw_and_Cleaned.xlsx` (all 3 sheets, 253 raw variables), all 14 figure assets, citation reciprocity (72 references), Zenodo DOI consistency (`10.5281/zenodo.23619994`), Markdown table column integrity, and prose/number alignment in `BSE_Main_Manuscript_Revision.md` and `BSE_Supporting_Information_Revision.md`.

### 7. `compile_revision_docx_and_pdf.py`
- **Role:** Reproducible Pandoc (`markdown-smart`) + `python-docx` (12 pt Times New Roman, booktabs tables) + Headless Chrome PDF compiler and dynamic section/paragraph-aligned native Word Track Changes (`<w:ins>` / `<w:del>`) generator for all resubmission deliverables.
