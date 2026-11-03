#!/usr/bin/env python3
"""
run_all_analysis.py
===================
Master One-Click Econometric Replication Runner for:
  "Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs"
  (Business Strategy and the Environment; Concept DOI: 10.5281/zenodo.23619994)

Executes the complete raw-to-verified pipeline sequentially from unmodified raw files:
  0. Environment & SHA-256 Raw Data Integrity Audit (never modifies raw files)
  1. Scripts/01_build_analytical_sample.py -> Cleaned CSV/XLSX + 253-Var Codebook
  2. Scripts/02_bivariate_probit.py        -> Joint FIML Bivariate Probit + Wald/AME/Holm (m=6)
  3. Scripts/03_causal_forest.py           -> Exact NR Propensity + Stabilized IPW + Subgroup CATEs
  4. Scripts/04_psm_and_remaining.py       -> Optimal 1:1 Logit PSM + Exact/Normal Rosenbaum + Models 4 & 5
  5. Scripts/05_robustness.py              -> 4-Quadrant MNLogit + 12-Spec Robustness + H3c Moderation + Appendix M
  6. Scripts/06_verify_all_results.py      -> 120+ Automated Assertions Against Results, Manuscript & SI

Usage:
  python run_all_analysis.py
"""

import hashlib
import importlib
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "Data", "Raw")
SCRIPTS_DIR = os.path.join(BASE_DIR, "Scripts")

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

PIPELINE_STEPS = [
    ("01_build_analytical_sample.py", "Sample Construction, Clean Recoding & 253-Variable Codebook"),
    ("02_bivariate_probit.py", "Joint FIML Bivariate Probit, Clustered Inference & Holm (m=6) Tests"),
    ("03_causal_forest.py", "Exact NR Propensity, Stabilized IPW ATE & Subgroup CATEs"),
    ("04_psm_and_remaining.py", "Optimal 1:1 Bipartite Logit PSM, Exact/Normal Rosenbaum & Models 4-5"),
    ("05_robustness.py", "4-Quadrant MNLogit, 12-Spec Robustness, H3c Moderation & Appendix M"),
    ("06_verify_all_results.py", "Automated Cross-Document & Econometric Verification Suite"),
]


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
    print(f"  Working root      : {BASE_DIR}")

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


def main():
    t_total = time.time()
    check_environment_and_raw_data()

    step_timings = []
    for idx, (script_name, desc) in enumerate(PIPELINE_STEPS, start=1):
        script_path = os.path.join(SCRIPTS_DIR, script_name)
        if not os.path.exists(script_path):
            conf_path = os.path.join(SCRIPTS_DIR, "confidential", script_name)
            if os.path.exists(conf_path):
                script_path = conf_path
            else:
                print(f"\nERROR: Missing pipeline script: {script_path}")
                sys.exit(1)

        print("\n" + "=" * 80)
        print(f"STEP {idx}/{len(PIPELINE_STEPS)}: {script_name} -- {desc}")
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
    print("ONE-CLICK REPLICATION PIPELINE SUMMARY (ALL STEPS PASSED)")
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


if __name__ == "__main__":
    main()
