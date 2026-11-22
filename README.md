# Replication Package: Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23619994.svg)](https://doi.org/10.5281/zenodo.23619994)
[![GitHub Repository](https://img.shields.io/badge/GitHub-SezginTunca%2Fsme--sustainability--decoupling-blue?logo=github)](https://github.com/SezginTunca/sme-sustainability-decoupling)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**"Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs"**  
**Target Journal:** *Business Strategy and the Environment* (Wiley) | **Manuscript ID:** `BSE-26-6866`  
**Status:** Revision prepared for resubmission (Version `2.3.0`).  
**Authors:** Sezgin Tunca, Mausam Budhathoki, and Hari Gopal Risal  
**Canonical Zenodo Concept DOI (all versions / latest):** [10.5281/zenodo.23619994](https://doi.org/10.5281/zenodo.23619994)  
**Version 2.3.1 Specific Zenodo DOI:** [10.5281/zenodo.24436201](https://doi.org/10.5281/zenodo.24436201)  

---

## Repository and Replication Package Structure

```text
├── README.md                            # Primary replication documentation
├── run_all_analysis.py                  # One-click raw-to-verified econometric pipeline
├── generate_all_figures.py              # One-click generator for Figures 1-5 & S1-S9 (PNG/PDF/SVG)
├── requirements.txt                     # Python package dependencies
│
├── Data/
│   ├── Raw/                             # User-supplied GESIS microdata (.sav, not redistributed) + Eurostat TSV
│   │   ├── FL498_2021_ZA7806.sav        # User-supplied: Flash Eurobarometer 498 Wave 5 (ZA7806, N=17,662)
│   │   ├── Eurostat_Digital_Intensity.tsv # Eurostat DESI digital intensity indicators (isoc_e_dii)
│   │   ├── SHA256SUMS_RAW.txt           # Exact SHA-256 verification checksums for raw inputs
│   │   └── ZA8869_v1-0-0.sav            # Optional user-supplied: Flash Eurobarometer 549 Wave 6 (ZA8869, Ext3)
│   │
│   ├── Processed/                       # Generated analytical datasets & model output tables
│   │   ├── SME_Analytical_Final.csv     # Generated locally by Step 01 (not redistributed in public archive)
│   │   ├── BiProbit_Joint_FE.csv        # Baseline joint bivariate probit coefficients & SEs
│   │   ├── BiProbit_Joint_FE.xlsx       # Joint FIML bivariate probit outputs (5 sheets)
│   │   ├── IPW_Propensity_Scores.csv    # Generated locally by Step 03 (not redistributed in public archive)
│   │   └── Robustness_FE.xlsx           # 12-spec robustness, MNLogit, H3c, Q5/Q6 DK audit & Appendix M (10 sheets)
│   │
│   ├── Input_Data_Consolidated.xlsx     # Generated locally by Step 01 (8 sheets; not redistributed in public archive)
│   ├── Results_Consolidated.xlsx        # 34-sheet master econometric & diagnostics workbook
│   └── Documentation/                   # 253-variable codebook, questionnaire PDFs & hypothesis map
│       ├── Codebook_ZA7806_Raw_and_Cleaned.xlsx
│       └── SME_Codebook_Hypothesis_Mapping.md
│
├── Scripts/
│   ├── 01_build_analytical_sample.py    # Clean recoding, Q5/Q6 routing audit, waterfall & 253-var codebook
│   ├── 02_bivariate_probit.py           # Joint FIML bivariate probit, Wald/AME & m=6 Holm tests
│   ├── 03_causal_forest.py              # Exact NR propensity, stabilized IPW ATE & subgroup CATEs
│   ├── 04_psm_and_remaining.py          # Optimal 1:1 bipartite PSM, Rosenbaum bounds & Models 4-5
│   ├── 05_robustness.py                 # 4-Quadrant MNLogit, 12-spec Table 6, H3c, Q5/Q6 DK audit & Appendix M
│   ├── requirements.txt                 # Core Python dependencies
│   ├── run_all_analysis.py              # Copy of master runner for Scripts/ execution
│   ├── generate_all_figures.py          # Copy of figure generator for Scripts/ execution
│   └── README.md                        # Script-by-script technical guide
│
├── Figures/                             # 14 display items in 600 DPI PNG, vector PDF, and SVG
```

---

## Data Access and Preparation

To run the data processing pipeline from scratch:

1. **Flash Eurobarometer 498 (`ZA7806`, Wave 5, Primary Dataset):**
   - **Source:** GESIS Data Archive for the Social Sciences
   - **Study:** *Flash Eurobarometer 498: SMEs, resource efficiency and green markets, wave 5* (Fieldwork: November–December 2021)
   - **DOI:** [10.4232/1.13934](https://doi.org/10.4232/1.13934) (`https://search.gesis.org/research_data/ZA7806`)
   - **Action:** Place `FL498_2021_ZA7806.sav` (or `ZA7806_v1-0-0.sav`) in `Data/Raw/` or directly in `Data/`. The runner automatically searches both paths.
2. **Eurostat Digital Intensity Index (`isoc_e_dii`):**
   - **Source:** Eurostat (*Enterprises with high/very high digital intensity index*, `E_DI3_HI` and `E_DI3_VHI`, 2021)
   - **Action:** Place `Eurostat_Digital_Intensity.tsv` in `Data/Raw/` or `Data/`.
3. **Flash Eurobarometer 549 (`ZA8869`, Wave 6, Optional Out-of-Sample Extension `Ext3`):**
   - **Source:** GESIS Data Archive (*Flash Eurobarometer 549: SMEs, resource efficiency and green markets, wave 6*, 2024)
   - **DOI:** [10.4232/1.14493](https://doi.org/10.4232/1.14493) (`https://search.gesis.org/research_data/ZA8869`)
   - **Action:** Place `ZA8869_v1-0-0.sav` in `Data/Raw/` or `Data/` to run the Appendix M `Ext3` replication ($N = 11,246$ EU-27 SMEs).

> **Data Redistribution & Licensing Notice:** In accordance with GESIS Data Archive terms of use and European Commission data policies, raw microdata files (`ZA7806` and `ZA8869`) and derived firm-level microdata extracts (`SME_Analytical_Final.csv`, `IPW_Propensity_Scores.csv`, `Input_Data_Consolidated.xlsx`) are excluded from the public GitHub repository and Zenodo archive. The raw survey datasets are freely available for scientific research upon simple registration on the GESIS portal (links above). Once downloaded and placed in either `Data/Raw/` or `Data/`, `python3 run_all_analysis.py` regenerates every clean dataset, workbook, and check automatically.

---

## One-Click Replication Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full raw-to-verified econometric pipeline (Steps 01-05; + Step 06 if Scripts/confidential/ is present)
python3 run_all_analysis.py

# 3. Regenerate all 14 publication figures (Figures 1-5 and S1-S9 in PNG, PDF, SVG)
python3 generate_all_figures.py
```

*Total runtime: approximately 3.5 to 4 minutes on a standard laptop.*

## Citation and License

Provided under the **MIT License**. Please cite the replication software package (`v2.3.1`; manuscript `BSE-26-6866` prepared for resubmission to *Business Strategy and the Environment*, Wiley):

```bibtex
@software{tunca2026green_code,
  author={Tunca, Sezgin and Budhathoki, Mausam and Risal, Hari Gopal},
  title={Replication Codebase: Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs},
  version={2.3.1},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.24436201},
  url={https://doi.org/10.5281/zenodo.23619994},
  note={Canonical Concept DOI: 10.5281/zenodo.23619994. Replication package for manuscript BSE-26-6866, revision prepared for resubmission to Business Strategy and the Environment.}
}
```
