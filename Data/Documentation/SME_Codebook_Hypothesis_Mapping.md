# SME Codebook and Hypothesis Mapping

**Manuscript:** *Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs*  
**Target Journal:** *Business Strategy and the Environment* (Wiley)  
**Primary Survey Dataset:** Flash Eurobarometer 498 (*SMEs, resource efficiency and green markets, wave 5*, November-December 2021; GESIS `ZA7806`, v1.0.0, DOI: `10.4232/1.13934`)  
**Temporal Replication Dataset (`Ext3`):** Flash Eurobarometer 549 (*Wave 6*, 2024; GESIS `ZA8869`, v1.0.0, DOI: `10.4232/1.14493`)  
**Macro Digital Indicator:** Eurostat `isoc_e_dii` (`E_DI3_HI` + `E_DI3_VHI`, 2021)  
**Machine-Readable 253-Variable Codebook:** `Data/Documentation/Codebook_ZA7806_Raw_and_Cleaned.xlsx`  

---

## 1. Dependent Variables and Configurational Typology

| Code Column | Survey Item & Coding | Construct & Measurement Boundary | Role in Models |
|---|---|---|---|
| `q9_binary` ($Y_1$) | `Q9`: `1` = Currently offers green products or services; `0` = No (`2` planning in next 2 years, `3` no); `4` (DK/NA) = missing | **Market-Facing Green Offerings** (self-reported commercial offering; can reflect genuine eco-innovation rather than superficial compliance) | Outcome 1 in Model 1 (`BiProbit`), Model 3 (`PSM`), `Table 6` |
| `q14_binary` ($Y_2$) | `Q14`: `1` = 1 (Yes, concrete strategy in place) or 4 (Already climate neutral); `0` = No strategy (`2` planning to define one, `3` no strategy and not planning); `5` (DK/NA) = missing | **Reported Net-Zero Strategy** (self-reported organization-level climate commitment, not audited emissions reductions) | Outcome 2 in Model 1 (`BiProbit`), Outcome in Model 2 (`IPW`), Model 3 (`PSM`), `H3c` |
| `q14_strict` | Same as `q14_binary` after excluding `Q14 = 4` ($n = 504$ already climate-neutral firms; $N = 10,624$) | **Strict 2030 Net-Zero Planners** | `Table 6` Spec 6 |
| `typology_4` | `0` = Neither $(0,0)$ [$50.9\%$]; `1` = Green Offerings Only $(1,0)$ [$21.3\%$]; `2` = Net-Zero Strategy Only $(0,1)$ [$13.7\%$]; `3` = Dual Adopters $(1,1)$ [$14.0\%$] ($N = 12,205$) | **Four-Quadrant Sustainability Typology** (primary configurational classification) | `Table 1`, `Table 3` (`MNLogit`, $N = 11,128$) |
| `DI` | $DI = Y_1 - Y_2 \in \{-1, 0, +1\}$ (mean $+0.0757$ on $N = 12,205$) | **Cross-Domain Sustainability Asymmetry Index** ($DI = 0$ is zero net directional asymmetry pooling Neither and Dual Adopters) | Outcome in Exploratory Model 4 (`OLS`, $N = 12,205$) |
| `Ext1a` / `Ext1b` | `Ext1`: $Y_1^{\text{strict}} = 1$ iff `Q9 = 1` & `Q10 > 10%` green turnover; $Y_2^{\text{strict}} = 1$ iff `Q14 = 1` (active 2030 strategy) & `Q15.2 = 1` or `Q15.3 = 1` low-carbon tech ($N = 10,687$ / $10,203$) | **Intensity-Conditioned Outcomes** | Appendix M (`Ext1`) |

---

## 2. Focal Explanatory Variables (`H1a`-`H3c`)

| Code Column | Survey Item (Verified Against `ZA7806` SPSS Metadata) | Construct | Hypothesis & Verified Baseline Result ($N = 11,128$) |
|---|---|---|---|
| `B2C` | `scr15_1` = 1 ("Sells products or services directly to consumers") | **B2C Retail Market Orientation** | **H1a (Strongly Supported):** $\hat{\beta}_1 = +0.203^{***}$, $\hat{\beta}_2 = +0.023$, $\Delta\hat{\beta} = +0.179^{***}$ ($p < 0.001, m=6\text{ Holm } p = 0.0002$) |
| `B2G_public` | `scr15_3` = 1 ("Sells to public administration") | **B2G Public Procurement Orientation** | **H1b (Qualified / Borderline):** $\hat{\beta}_1 = +0.060^{*}$, $\hat{\beta}_2 = -0.021$, $\Delta\hat{\beta} = +0.081^{*}$ ($p = 0.015, m=6\text{ Holm } p = 0.060$) |
| `B2B_private` | `scr15_2` = 1 ("Sells to other companies") | **B2B Commercial Supply-Chain Orientation** | **H1c (Not Supported in Full Sample):** $\hat{\beta}_1 = +0.012$, $\hat{\beta}_2 = -0.034$ ($p = 0.314$), $\Delta\hat{\beta} = +0.046$ ($p = 0.244$) |
| `barrier_admin` | `Q7.1` = 1 ("Complexity of administrative or legal procedures"; routed on $\ge 1$ `Q1.1`-`Q1.9` action) | **Perceived Administrative Complexity** | **H2a (Cross-Eq Diff Not Supported):** $\hat{\beta}_1 = +0.095^{**}$ ($p = 0.002$), $\hat{\beta}_2 = +0.036$ ($p = 0.231$), $\Delta\hat{\beta} = +0.059$ ($p = 0.125$) |
| `barrier_expertise` | `Q7.6` = 1 ("Lack of specific environmental expertise"; routed on $\ge 1$ `Q1.1`-`Q1.9` action) | **Perceived Environmental Expertise Shortage** | **H2b (Supported w/ Refinement):** $\hat{\beta}_1 = -0.090^{**}$ ($p = 0.008$), $\hat{\beta}_2 = -0.220^{***}$ ($p < 0.001$), $\Delta\hat{\beta} = +0.129^{**}$ ($p = 0.002, \text{Holm } p = 0.012$) |
| `grant_receipt` | `Q6.1` = 1 ("Public funding such as grants, guarantees or subsidized loans"; coded `0` for routed firms with `Q5.3 != 1` [95.0% internal/other support only], tested excluding all `516` `Q5.5`/`Q6.9` DK/NA in `Table J4` [$N = 10,612 / 10,979$] and within `Q5.3 = 1` in `Table 6` Spec 7 & `Table J4`) | **Public Financial Support** (code column retained as `grant_receipt` for backward pipeline compatibility) | **H3a (Supported w/ Qualifications):** $\hat{\beta}_2 = +0.121^{**}$ ($p = 0.009$); IPW $\widehat{\text{ATE}} = +0.043$ ($p = 0.004$); AIPW $+0.041$ ($p = 0.006$); Optimal 1:1 PSM $+0.040$ ($p = 0.036$, $\Gamma^* = 1.035$-$1.102$)<br>**H3b (Binary Diff Not Supported):** $\hat{\beta}_1 = +0.072^{+}$ ($p = 0.060$), $\Delta\hat{\beta} = -0.049$ ($p = 0.449$)<br>**H3c (Supported, prespecified):** $\Delta\text{AME}_{\text{Admin}} = +0.073^{**}$ ($p = 0.0011$) |

---

## 3. Controls, Fixed Effects, and Validation Indicators

| Code Column | Source & Coding | Purpose |
|---|---|---|
| `advice_public` | `Q6.4` = 1 ("Advice or other non-financial assistance from public administration") | Non-financial public support control ($\hat{\beta}_1 = +0.039, \hat{\beta}_2 = +0.035$) |
| `scr10` | Employee size class: `1` = Micro (1-9), `2` = Small (10-49), `3` = Medium (50-249) | Firm size control ($\hat{\beta}_1 = -0.003, p = 0.929$; $\hat{\beta}_2 = +0.066^{*}, p = 0.014$) |
| `scr14_cat` | Categorical annual turnover (`1` = `< EUR 25,000` ref., `2`..`9` substantive brackets, `10` = DK/refusal retained as separate category, $9$ FE dummies) | Baseline turnover fixed effects ($\chi^2(18) = 187.96, p < 0.001$; complete-case `scr14 < 10` $N = 9,764$ tested in Spec 11) |
| `re_action_count` | Sum of 9 substantive resource-efficiency actions `Q1.1`-`Q1.9` (`0`..`9`; `0` exactly identifies $1,231$ unrouted firms) | Operational resource-efficiency intensity (`re_action_count_10` tested in Spec 3; omitted in Spec 2) |
| `country_iso2` | 27 EU Member States (`AT` reference, 26 country FE; clustering unit $G = 27, df = 26$) | Absorbs national regulatory, fiscal, and cultural heterogeneity ($\chi^2(52) = 731.43$) |
| `nace_b` / `nace_a` | 4 NACE macro-sectors (`C` Manufacturing ref., `G` Retail, `B/D/E/F` Industry, `Services`; 12-section `nace_a` in Spec 10) | Absorbs sectoral technology and product/service intensity ($\chi^2(6) = 98.77$) |
| `SCR16`, `DX5`, `Q4`, `Q10`, `Q15`, `Q8`, `Q3`, `DII_std` | Within-survey business model (`SCR16`), green FTEs (`DX5`), RE investment (`Q4`), green revenue (`Q10`), carbon actions (`Q15`), desired support (`Q8`), cost impact (`Q3`), and standardized Eurostat DII (`DII_std`) | Construct validation (`Table 1`), Exploratory Models 4-5, and Appendix M (`Ext1`-`Ext2`) |
| `q5_q6_dk` | `q5.5` = 1 (Q5 DK/NA) or `q6.9` = 1 (Q6 DK/NA) among routed SMEs (`622` routed; `542` IPW; `516` joint) | Q5/Q6 item non-response routing audit (`AppendixJ_Q5Q6_DK_Audit`, `Table J4`) |

---

## 4. Master Sample Waterfall (`Sample_Waterfall` Sheet)

1. **Raw GESIS `ZA7806` interviews:** $N = 17,662$
2. **EU-27 Member States:** $N = 14,215$
3. **EU-27 SMEs (`1 <= scr10 <= 3`):** $N = 13,322$
4. **Valid `Q9` & `Q14` Four-Quadrant Typology & Model 4 Sample:** $N = 12,205$
5. **Routed Resource-Efficiency SMEs (`re_action_count >= 1`):** $N = 12,091$ ($1,231$ zero-action unrouted)
6. **Single-Outcome `Q14` IPW Sample:** $N = 11,521$ ($1,292$ treated, $10,229$ controls)
7. **Primary Joint Econometric Sample (Models 1, 3, MNLogit, `H3c`):** $N = 11,128$ ($1,256$ treated, $9,872$ controls)
8. **Auxiliary Subsamples:** Strict 2030 planners $N = 10,624$; Operating cost `Q3` $N = 10,011$ (`+Q4` $N = 9,058$); Expanded covariates $N = 9,857$; Complete-case turnover $N = 9,764$; Product-selling SMEs $N = 7,299$; External support seekers `Q5.3 = 1` $N = 3,055$ (`Q14`) / $N = 2,959$ (joint); 2024 Wave 6 `ZA8869` replication $N = 11,246$.
