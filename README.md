# Replication Package: Green Offerings and Net-Zero Plans in European SMEs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23619994.svg)](https://doi.org/10.5281/zenodo.23619994)
[![GitHub Repository](https://img.shields.io/badge/GitHub-SezginTunca%2Fsme--sustainability--decoupling-blue?logo=github)](https://github.com/SezginTunca/sme-sustainability-decoupling)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**"Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs"**  
**Target Journal:** *Business Strategy and the Environment* (Wiley)  
**Authors:** Sezgin Tunca, Mausam Budhathoki, and Hari Gopal Risal  
**Canonical Zenodo Concept DOI:** [10.5281/zenodo.23619994](https://doi.org/10.5281/zenodo.23619994)  

---

## Repository and Replication Package Structure

```text
├── README.md                            # Primary replication documentation
├── run_all_analysis.py                  # One-click raw-to-verified econometric pipeline
├── generate_all_figures.py              # One-click generator for Figures 1-5 & S1-S9 (PNG/PDF/SVG)
├── requirements.txt                     # Python package dependencies
│
├── Data/
│   ├── Raw/                             # Unmodified raw survey & macro indicator microdata
│   │   ├── FL498_2021_ZA7806.sav        # Flash Eurobarometer 498 Wave 5 SPSS file (ZA7806, N=17,662)
│   │   ├── Eurostat_Digital_Intensity.tsv # Eurostat DESI digital intensity indicators (isoc_e_dii)
│   │   └── ZA8869_v1-0-0.sav            # Optional: Flash Eurobarometer 549 Wave 6 (ZA8869, Ext3)
│   │
│   ├── Processed/                       # Generated clean analytical datasets & model workbooks
│   │   ├── SME_Analytical_Final.csv     # Firm-level EU-27 SME sample (N=13,322)
│   │   ├── BiProbit_Joint_FE.csv        # Baseline joint bivariate probit coefficients & SEs
│   │   ├── BiProbit_Joint_FE.xlsx       # Joint FIML bivariate probit outputs (5 sheets)
│   │   ├── IPW_Propensity_Scores.csv    # Firm-level propensity scores & stabilized IPW weights (N=11,521)
│   │   └── Robustness_FE.xlsx           # 12-spec robustness, MNLogit, H3c & Appendix M (9 sheets)
│   │
│   ├── Input_Data_Consolidated.xlsx     # 8-sheet consolidated input & sample waterfall workbook
│   ├── Results_Consolidated.xlsx        # 33-sheet master econometric & diagnostics workbook
│   └── Documentation/                   # 253-variable codebook, questionnaire PDFs & hypothesis map
│       ├── Codebook_ZA7806_Raw_and_Cleaned.xlsx
│       └── SME_Codebook_Hypothesis_Mapping.md
│
├── Scripts/
│   ├── 01_build_analytical_sample.py    # Clean recoding, sample waterfall & 253-variable codebook
│   ├── 02_bivariate_probit.py           # Joint FIML bivariate probit, Wald/AME & m=6 Holm tests
│   ├── 03_causal_forest.py              # Exact NR propensity, stabilized IPW ATE & subgroup CATEs
│   ├── 04_psm_and_remaining.py          # Optimal 1:1 bipartite PSM, Rosenbaum bounds & Models 4-5
│   ├── 05_robustness.py                 # 4-Quadrant MNLogit, 12-spec Table 6, H3c & Appendix M
│   ├── 06_verify_all_results.py         # 195+ automated econometric & manuscript assertions
│   ├── compile_revision_docx_and_pdf.py # Pandoc/python-docx/Chrome DOCX, TrackChanges & PDF exporter
│   ├── requirements.txt                 # Core Python dependencies
│   └── README.md                        # Script-by-script technical guide
│
├── Figures/                             # 14 display items in 600 DPI PNG, vector PDF, and SVG
├── Manuscript/                          # Synchronized manuscript, figures, and DOCX/PDF exports
└── Revision/                            # Resubmission manuscripts, SI, matrices, and response letters
```

---

## Data Access and Preparation

To run the data processing pipeline from scratch:

1. **Flash Eurobarometer 498 (`ZA7806`, Wave 5, Primary Dataset):**
   - **Source:** GESIS Data Archive for the Social Sciences
   - **Study:** *Flash Eurobarometer 498: SMEs, resource efficiency and green markets, wave 5* (November-December 2021)
   - **DOI:** [10.4232/1.13934](https://doi.org/10.4232/1.13934) (`https://search.gesis.org/research_data/ZA7806`)
   - **Action:** Place `FL498_2021_ZA7806.sav` (or `ZA7806_v1-0-0.sav`) in `Data/Raw/`.
2. **Eurostat Digital Intensity Index (`isoc_e_dii`):**
   - **Source:** Eurostat (*Enterprises with high/very high digital intensity index*, `E_DI3_HI` and `E_DI3_VHI`, 2021)
   - **Action:** Place `Eurostat_Digital_Intensity.tsv` in `Data/Raw/`.
3. **Flash Eurobarometer 549 (`ZA8869`, Wave 6, Optional Out-of-Sample Extension `Ext3`):**
   - **Source:** GESIS Data Archive (*Flash Eurobarometer 549: SMEs, resource efficiency and green markets, wave 6*, 2024)
   - **DOI:** [10.4232/1.14493](https://doi.org/10.4232/1.14493) (`https://search.gesis.org/research_data/ZA8869`)
   - **Action:** Place `ZA8869_v1-0-0.sav` in `Data/Raw/` to run the Appendix M `Ext3` replication ($N = 11,246$ EU-27 SMEs).

---

## One-Click Replication Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full raw-to-verified econometric pipeline (Steps 01-06 + 195+ assertions)
python3 run_all_analysis.py

# 3. Regenerate all 14 publication figures (Figures 1-5 and S1-S9 in PNG, PDF, SVG)
python3 generate_all_figures.py
```

*Total runtime: approximately 3.5 to 4 minutes on a standard laptop.*

---

## Summary of Hypotheses and Verified Empirical Findings

The analysis examines how market-channel orientation (`scr15`), perceived capability barriers (`Q7`), and public financial support (`Q6.1`) relate to market-facing green product/service offerings ($Y_1$, `Q9`) and reported organization-wide net-zero decarbonization strategies ($Y_2$, `Q14`) across **$N = 13,322$ EU-27 SMEs** ($N = 12,205$ valid typology sample; $N = 11,521$ single-outcome IPW sample; $N = 11,128$ primary joint econometric sample across $G = 27$ country clusters).

### 1. Four-Quadrant Sustainability Typology (`Table 1`, $N = 12,205$; `Table 3`, $N = 11,128$)
- **Neither / Non-Adopters** ($Y_1 = 0, Y_2 = 0$): $6,213$ ($50.9\%$)
- **Green Offerings Only** ($Y_1 = 1, Y_2 = 0$; $DI = +1$): $2,602$ ($21.3\%$)
- **Net-Zero Strategy Only** ($Y_1 = 0, Y_2 = 1$; $DI = -1$): $1,678$ ($13.7\%$)
- **Dual Adopters** ($Y_1 = 1, Y_2 = 1$): $1,712$ ($14.0\%$)
- **Residual Disturbance Correlation ($\hat{\rho}$):** $+0.255$ ($\text{SE} = 0.025, z = 10.11, p < 0.001, \ln\mathcal{L} = -12,930.55$).

### 2. Market-Channel Orientations (`H1a`, `H1b`, `H1c`; `Table 2`)
- **H1a (B2C Orientation):** **Strongly supported.** $\hat{\beta}_1 = +0.203$ ($p < 0.001, \text{AME}_1 = +0.071$) vs. $\hat{\beta}_2 = +0.023$ ($p = 0.530, \text{AME}_2 = +0.007$); cross-equation Wald $\Delta\hat{\beta} = +0.179$ ($z = 4.21, p < 0.001, m = 6\text{ Holm } p = 0.0002, t_{26}\text{ Holm } p = 0.0016$).
- **H1b (B2G Orientation):** **Qualified / borderline support.** $\hat{\beta}_1 = +0.060$ ($p = 0.040, t_{26}\ p = 0.050, \text{AME}_1 = +0.021$) vs. $\hat{\beta}_2 = -0.021$ ($p = 0.525$); cross-equation $\Delta\hat{\beta} = +0.081$ ($z = 2.44, p = 0.015, t_{26}\ p = 0.022$, $m = 6$ Holm $p = 0.060$).
- **H1c (B2B Orientation):** **Not supported in the full sample.** $\hat{\beta}_1 = +0.012$ ($p = 0.730$), $\hat{\beta}_2 = -0.034$ ($p = 0.314$), $\Delta\hat{\beta} = +0.046$ ($p = 0.244$).

### 3. Perceived Capability Constraints (`H2a`, `H2b`; `Table 2`)
- **H2a (Administrative Complexity, `Q7.1`):** **Cross-equation difference not supported.** Positively associated with $Y_1$ ($\hat{\beta}_1 = +0.095, p = 0.002, \text{AME}_1 = +0.034$) and null on $Y_2$ ($\hat{\beta}_2 = +0.036, p = 0.231$); $\Delta\hat{\beta} = +0.059$ ($p = 0.125$).
- **H2b (Environmental Expertise Shortages, `Q7.6`):** **Supported with theoretical refinement.** Negatively associated with both $Y_1$ ($\hat{\beta}_1 = -0.090, p = 0.008, \text{AME}_1 = -0.032$) and $Y_2$ ($\hat{\beta}_2 = -0.220, p < 0.001, \text{AME}_2 = -0.067$), penalizing net-zero planning twice as severely ($\Delta\hat{\beta} = +0.129, z = 3.05, p = 0.002, m = 6\text{ Holm } p = 0.012$).

### 4. Public Financial Support & Administrative Moderation (`H3a`, `H3b`, `H3c`; `Tables 2, 4, 5, 6`)
- **H3a (Direct Net-Zero Association):** **Supported with sensitivity qualifications.** Joint probit $\hat{\beta}_2 = +0.121$ ($p = 0.009, \text{AME}_2 = +0.039$); IPW $\widehat{\text{ATE}} = +0.043$ ($p_{\text{reest}} = 0.004, p_{\text{IF,cl}} = 0.027$, $\text{ESS}_T = 876.1$); AIPW $\widehat{\text{ATE}} = +0.041$ ($p = 0.006, p_{\text{cl}} = 0.033$); optimal 1:1 bipartite PSM $\widehat{\text{ATT}} = +0.040$ (paired $p = 0.036$, cluster $p = 0.193$; $\max |\text{SMD}| = 0.058$); exact Rosenbaum critical $\Gamma^* = 1.035$ ($N = 11,128$) and $\Gamma^* = 1.102$ ($N = 9,764$).
- **H3b (Cross-Domain Support Contrast):** **Not supported on binary indicators** ($\hat{\beta}_1 = +0.072, p = 0.060$; $\Delta\hat{\beta} = -0.049, p = 0.449$).
- **H3c (Administrative Barrier Moderation):** **Supported.** Conditional $\text{AME}(\text{Admin}=1) = +0.081$ ($p < 0.001$) vs. $\text{AME}(\text{Admin}=0) = +0.008$ ($p = 0.688$), yielding $\Delta\text{AME}_{\text{Admin}} = +0.073$ ($\text{SE}_{\text{cl}} = 0.022, 95\%\text{ CI } [+0.029, +0.117], p = 0.0011, t_{26}\ p = 0.0030$; IPW $\Delta\widehat{\text{CATE}}_{\text{Admin}} = +0.073, p = 0.022$). Expertise moderation is null ($\Delta\text{AME}_{\text{Exp}} = -0.008, p = 0.788$).

### 5. Exploratory Extensions (`Section 4.6` & `Appendix M`)
- **Model 4 (Digital Intensity Moderation, $N = 12,205$):** $\text{B2B} \times \text{DII}_{\text{std}} = -0.009$ ($\text{SE}_{\text{cl}} = 0.017, p = 0.598$).
- **Model 5 (Operating Cost Ordered Probit, $N = 10,011$ & $N = 9,058$):** Green offerings positively predict perceived cost increases ($\hat{\beta}_{Q9} = +0.079, p = 0.008$ on $N = 10,011$; $+0.064, p = 0.048$ controlling for `Q4` on $N = 9,058$), whereas reported net-zero strategies are null ($\hat{\beta}_{Q14} = +0.001, p = 0.980$).
- **Appendix M (`Ext1`-`Ext3`):** Intensity-conditioned constructs (`Ext1`, $\Delta\hat{\beta}_{\text{Support}} = -0.172, p = 0.015$), `Q8` desired-support validation (`Ext2`), and 2024 Wave 6 `ZA8869` out-of-sample replication (`Ext3`, $N = 11,246$, $\hat{\rho} = +0.197$, `Support -> Y2` $\hat{\beta}_2 = +0.153, p < 0.001$).

---

## Consolidated Excel Workbooks

- **`Data/Input_Data_Consolidated.xlsx` (8 sheets):** `Analytical_Sample`, `DESI_Country_Data`, `Codebook`, `Country_Summary`, `Descriptive_Statistics`, `Sample_Waterfall`, `Table1_Typology_12205`, `Missing_Data_Attrition`.
- **`Data/Results_Consolidated.xlsx` (33 sheets):** Complete model estimates, variance-covariance diagnostics, FE Wald tests, 4-quadrant MNLogit, IPW/AIPW/optimal-PSM balance & 13-spec sensitivity suite, Rosenbaum bounds across 3 samples, 12-spec `Table 6` robustness matrix, probability-scale moderation, construct validity, and Appendix M (`Ext1`-`Ext3`).
- **`Data/Documentation/Codebook_ZA7806_Raw_and_Cleaned.xlsx` (3 sheets):** `Raw_ZA7806_253_Vars`, `Constructed_Variables`, `Sample_Waterfall`.

---

## Citation and License

Provided under the **MIT License**. Please cite:

```bibtex
@article{tunca2026green,
  title={Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs},
  author={Tunca, Sezgin and Budhathoki, Mausam and Risal, Hari Gopal},
  journal={Business Strategy and the Environment},
  year={2026},
  publisher={Wiley},
  doi={10.5281/zenodo.23619994}
}
```
