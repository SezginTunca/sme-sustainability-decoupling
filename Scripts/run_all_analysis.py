#!/usr/bin/env python3
"""
Scripts/run_all_analysis.py
================================================================================
Master Analytical Pipeline Runner for:
  "Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs"
  (Business Strategy and the Environment; Concept DOI: 10.5281/zenodo.23619994)

Orchestrates the entire empirical research workflow starting from unmodified raw data:
  0. Environment & SHA-256 Raw Data Integrity Audit (never modifies raw files)
  1. Data Processing & Data Handling:
     - Scripts/01_build_analytical_sample.py
       -> Raw Flash Eurobarometer 498 (ZA7806) filtering (EU-27, SMEs 1-249 employees)
       -> Systematic questionnaire routing verification & feature engineering
       -> Eurostat Digital Intensity Index (isoc_e_dii, 2021) country merge
       -> Cleaned analytical dataset (Data/Processed/SME_Analytical_Final.csv)
       -> Complete 253-variable codebook (Data/Documentation/Codebook_ZA7806_Raw_and_Cleaned.xlsx)
       -> Input data consolidated workbook (Data/Input_Data_Consolidated.xlsx)

  2. Formal Econometric Analysis:
     - Scripts/02_bivariate_probit.py
       -> Joint Full Information Maximum Likelihood (FIML) Bivariate Probit (N = 11,128)
       -> 27-country cluster-robust covariance and finite-sample t(26) inference
       -> Average Marginal Effects (AMEs) and cross-equation Wald tests
       -> Holm-Bonferroni family-wise error rate corrections (m = 6 focal hypotheses)

  3. Data Analysis & Causal Estimation:
     - Scripts/03_causal_forest.py
       -> Exact Newton-Raphson propensity score estimation without heuristic clipping
       -> Strict common support trimming [0.01, 0.99] and Kish Effective Sample Size diagnostics
       -> Stabilized Inverse Probability Weighting (IPW) Average Treatment Effects
       -> Subgroup Conditional Average Treatment Effects (CATEs) across sectors, sizes & countries

  4. Formal Causal & Sensitivity Analysis:
     - Scripts/04_psm_and_remaining.py
       -> Austin (2011) Optimal 1:1 Minimum-Distance Bipartite Matching on logit propensity
       -> Exact McNemar Binomial and Normal approximation Rosenbaum sensitivity bounds
       -> Cross-level Digital Intensity Index OLS regressions (Model 4, N = 12,205)
       -> Operating cost impact Ordered Probit regressions (Model 5, N = 10,011 & 9,058)

  5. Formal Robustness Analysis & Extensions:
     - Scripts/05_robustness.py
       -> Table 3 Four-Quadrant Multinomial Logit discrete AMEs
       -> Table 6 Twelve-Specification Consolidated Robustness Matrix & Sector Splits
       -> Hypothesis H3c Public Support Moderation (Country-Clustered Delta-Method Delta-AME)
       -> Appendix M Empirical Extensions (Ext1 Tightened Constructs, Ext2 Q8 Mechanism, Ext3 Wave 6)

  6. Comprehensive Verification Suite (Author QA):
     - Scripts/06_verify_all_results.py (or Scripts/confidential/06_verify_all_results.py)
       -> 200 automated programmatic assertions validating workbooks, figures, and manuscripts

Usage:
  python Scripts/run_all_analysis.py              # Run complete pipeline including verification
  python Scripts/run_all_analysis.py --core-only  # Run core data process, handling, and formal analysis (Steps 1-5)
  python Scripts/run_all_analysis.py --verify     # Run verification suite only (Step 6)
================================================================================
"""

import argparse
import hashlib
import importlib
import os
import subprocess
import sys
import time

_this_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_this_dir) == "Scripts":
    BASE_DIR = os.path.dirname(_this_dir)
    SCRIPTS_DIR = _this_dir
else:
    BASE_DIR = _this_dir
    SCRIPTS_DIR = os.path.join(BASE_DIR, "Scripts")

RAW_DIR = os.path.join(BASE_DIR, "Data", "Raw")
CONFIDENTIAL_DIR = os.path.join(SCRIPTS_DIR, "confidential")

EXPECTED_SHA256 = {
    "ZA7806_SAV": (
        ["ZA7806_v1-0-0.sav", "FL498_2021_ZA7806.sav"],
        "27b688ef9e0dff7a193eaf1ff6a8d188eced56da4ed1ad3f8d957facbbce976d",
        True,
    ),
    "EUROSTAT_DII": (
        ["Eurostat_Digital_Intensity.tsv"],
        "ceb99ce287b9cf277e037d2f2d0b8f5a5cc14a7b4daa1c3c612a04fc4dc62c72",
        True,
    ),
    "ZA8869_WAVE6": (
        [
            "ZA8869_v1-0-0.sav",
            "ZA8869_v2-0-0.sav",
            os.path.join(
                "All_Raw_datasets_SME_Res_Eff",
                "Flash Eurobarometer 549 (SMEs, Resource Efficiency and Green Markets, wave 6)",
                "ZA8869_v1-0-0.sav",
            ),
        ],
        "f6945520fb41f66b46efce9e02f2776ba5f77e3d679c4e73810412582137d6cf",
        False,
    ),
}

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("statsmodels", "statsmodels"),
    ("patsy", "patsy"),
    ("pyreadstat", "pyreadstat"),
    ("openpyxl", "openpyxl"),
]

CORE_PIPELINE_STEPS = [
    (
        "01_build_analytical_sample.py",
        "Data Processing & Data Handling: Sample Construction, Clean Recoding & 253-Variable Codebook",
    ),
    (
        "02_bivariate_probit.py",
        "Formal Econometric Analysis: Joint FIML Bivariate Probit, Clustered Inference & Holm (m=6) Tests",
    ),
    (
        "03_causal_forest.py",
        "Data Analysis: Exact NR Propensity Scores, Stabilized IPW ATE & Subgroup CATEs",
    ),
    (
        "04_psm_and_remaining.py",
        "Formal Analysis: Austin (2011) Optimal 1:1 Logit PSM, Exact/Normal Rosenbaum & Models 4-5",
    ),
    (
        "05_robustness.py",
        "Formal Robustness Analysis: 4-Quadrant MNLogit, 12-Spec Matrix, H3c Moderation & Appendix M",
    ),
]

VERIFICATION_STEP = (
    "06_verify_all_results.py",
    "Comprehensive Verification: Automated Cross-Document & Econometric Verification Suite (200 Checks)",
)


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_environment_and_raw_data():
    print("=" * 80)
    print("STEP 0: Python Environment & Unmodified Raw Data Integrity Check")
    print("=" * 80)
    print(f"  Python executable : {sys.executable}")
    print(f"  Python version    : {sys.version.split()[0]}")
    print(f"  Project root      : {BASE_DIR}")
    print(f"  Scripts directory : {SCRIPTS_DIR}")

    missing_pkgs = []
    for pkg_label, mod_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [PKG OK] {pkg_label:<14} ({ver})")
        except ImportError:
            missing_pkgs.append(pkg_label)
            print(f"  [PKG MISSING] {pkg_label}")

    if missing_pkgs:
        print(f"\nERROR: Missing required packages: {', '.join(missing_pkgs)}")
        print("Install via: pip install -r requirements.txt")
        sys.exit(1)

    print("\n  Verifying SHA-256 checksums of unmodified raw inputs:")
    for key, (candidates, expected_hash, required) in EXPECTED_SHA256.items():
        found_path = None
        for rel in candidates:
            p = os.path.join(RAW_DIR, rel)
            if os.path.exists(p):
                found_path = p
                break
        if found_path is None:
            if required:
                print(f"  [RAW MISSING] {key}: Expected in {RAW_DIR} ({candidates[0]})")
                print("  Download ZA7806 (https://doi.org/10.4232/1.13934) into Data/Raw/ before running.")
                sys.exit(1)
            else:
                print(f"  [RAW OPTIONAL] {key}: Not present in Data/Raw/ (Wave 6 out-of-sample check will skip gracefully)")
                continue

        actual_hash = sha256_file(found_path)
        if expected_hash and actual_hash != expected_hash:
            if key == "ZA8869_WAVE6":
                print(f"  [RAW INFO] {os.path.basename(found_path)} SHA-256 = {actual_hash[:16]}... (alternate GESIS release)")
            else:
                print(f"  [HASH MISMATCH] {found_path}\n    Expected: {expected_hash}\n    Actual  : {actual_hash}")
                sys.exit(1)
        else:
            print(f"  [RAW OK] {os.path.relpath(found_path, BASE_DIR)} (SHA-256: {actual_hash[:16]}...)")


def resolve_script_path(script_name):
    # Check standard Scripts/ location first
    p1 = os.path.join(SCRIPTS_DIR, script_name)
    if os.path.exists(p1):
        return p1
    # Check Scripts/confidential/
    p2 = os.path.join(CONFIDENTIAL_DIR, script_name)
    if os.path.exists(p2):
        return p2
    return None


def run_pipeline(steps):
    t_total = time.time()
    step_timings = []

    for idx, (script_name, desc) in enumerate(steps, start=1):
        script_path = resolve_script_path(script_name)
        if not script_path:
            print(f"\nERROR: Missing pipeline script: {script_name}")
            sys.exit(1)

        print("\n" + "=" * 80)
        print(f"STEP {idx}/{len(steps)}: {script_name} -- {desc}")
        print("=" * 80)
        t0 = time.time()
        res = subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
        dt = time.time() - t0
        step_timings.append((script_name, dt, res.returncode))
        if res.returncode != 0:
            print(f"\n[FAILED] {script_name} exited with code {res.returncode} after {dt:.1f}s.")
            sys.exit(res.returncode)

    total_dt = time.time() - t_total
    print("\n" + "=" * 80)
    print("ANALYTICAL PIPELINE EXECUTION SUMMARY (ALL STEPS PASSED)")
    print("=" * 80)
    for script_name, dt, code in step_timings:
        print(f"  [PASS] {script_name:<34} {dt:7.1f}s (exit {code})")
    print("-" * 80)
    print(f"  Total elapsed time : {total_dt:.1f}s")
    print(f"  Cleaned dataset    : {os.path.join('Data', 'Processed', 'SME_Analytical_Final.csv')}")
    print(f"  253-var codebook   : {os.path.join('Data', 'Documentation', 'Codebook_ZA7806_Raw_and_Cleaned.xlsx')}")
    print(f"  Consolidated tables: {os.path.join('Data', 'Results_Consolidated.xlsx')}")
    print("  Next step (figures): python generate_all_figures.py")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Master Analytical Pipeline Runner for European SME Sustainability Decoupling Research"
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run core data processing, data handling, data analysis, and formal analysis steps (01-05)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification suite only (06_verify_all_results.py)",
    )
    args = parser.parse_args()

    check_environment_and_raw_data()

    if args.verify:
        steps = [VERIFICATION_STEP]
    elif args.core_only:
        steps = CORE_PIPELINE_STEPS
    else:
        # Default: Core analytical steps + verification if available
        steps = list(CORE_PIPELINE_STEPS)
        if resolve_script_path(VERIFICATION_STEP[0]):
            steps.append(VERIFICATION_STEP)

    run_pipeline(steps)


if __name__ == "__main__":
    main()
