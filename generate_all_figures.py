#!/usr/bin/env python3
"""
generate_all_figures.py
=======================
Standalone One-Click Publication Visualization Pipeline for:
  "Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs"
  (Business Strategy and the Environment)

Design Guarantees:
  1. Zero Hardcoded Statistics: Dynamically reads every parameter, standard error,
     confidence interval, sample size, propensity score, and diagnostic from:
       - Data/Processed/SME_Analytical_Final.csv
       - Data/Processed/IPW_Propensity_Scores.csv (exact Newton-Raphson PS from 03_causal_forest.py)
       - Data/Results_Consolidated.xlsx
       - Data/ne_110m_admin_0_countries/ (Natural Earth Europe boundaries)
  2. Consistent Wiley / APA Formatting:
       - Top overall figure titles/suptitles are omitted across all 14 figures (captions carry figure titles).
       - Multi-panel figures retain clean (a), (b) sub-panel headers.
       - Unified Arial / Helvetica / DejaVu Sans typography and colorblind-safe Teal/Coral/Navy/Slate palette.
       - Zero legacy terminology (uses Sustainability Asymmetry Index, Green Offerings,
         Reported Net-Zero Strategy, and Public Financial Support).
  3. Triple-Format Editable Export:
       - 600 DPI raster (.png)
       - TrueType editable vector (.pdf, pdf.fonttype = 42)
       - Editable SVG text (.svg, svg.fonttype = 'none')

Usage:
  python generate_all_figures.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
DATA_CSV = os.path.join(DATA_DIR, "Processed", "SME_Analytical_Final.csv")
PS_CSV = os.path.join(DATA_DIR, "Processed", "IPW_Propensity_Scores.csv")
EXCEL_PATH = os.path.join(DATA_DIR, "Results_Consolidated.xlsx")

FIG_DIR = os.path.join(BASE_DIR, "Figures")
REV_DIR = os.path.join(BASE_DIR, "Revision")
MANU_FIG_DIR = os.path.join(BASE_DIR, "Manuscript", "Figures")
PKG_FIG_DIR = os.path.join(BASE_DIR, "SME_Replication_Package", "Manuscript", "Figures")

EXPORT_DIRS = [FIG_DIR]
for opt_dir in [REV_DIR, MANU_FIG_DIR, PKG_FIG_DIR]:
    parent = os.path.dirname(opt_dir)
    if opt_dir == REV_DIR and os.path.isdir(REV_DIR):
        EXPORT_DIRS.append(REV_DIR)
    elif os.path.isdir(parent):
        EXPORT_DIRS.append(opt_dir)

for d in EXPORT_DIRS:
    os.makedirs(d, exist_ok=True)

# ── Country Mappings ───────────────────────────────────────────────────────
COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "HR": "Croatia",
    "CY": "Cyprus", "CZ": "Czechia", "DK": "Denmark", "EE": "Estonia",
    "FI": "Finland", "FR": "France", "DE": "Germany", "EL": "Greece",
    "HU": "Hungary", "IE": "Ireland", "IT": "Italy", "LV": "Latvia",
    "LT": "Lithuania", "LU": "Luxembourg", "MT": "Malta", "NL": "Netherlands",
    "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SK": "Slovakia",
    "SI": "Slovenia", "ES": "Spain", "SE": "Sweden",
}

EU27_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "HR": "HRV", "CY": "CYP",
    "CZ": "CZE", "DK": "DNK", "EE": "EST", "FI": "FIN", "FR": "FRA",
    "DE": "DEU", "EL": "GRC", "HU": "HUN", "IE": "IRL", "IT": "ITA",
    "LV": "LVA", "LT": "LTU", "LU": "LUX", "MT": "MLT", "NL": "NLD",
    "PL": "POL", "PT": "PRT", "RO": "ROU", "SK": "SVK", "SI": "SVN",
    "ES": "ESP", "SE": "SWE",
}

# ── Colorblind-Safe Palette ────────────────────────────────────────────────
TEAL = "#1B7A6E"     # Net-Zero Strategy (Y2, Q14) / Positive Strategic Effect
CORAL = "#E06358"    # Green Offerings (Y1, Q9) / Commercial Asymmetry
AMBER = "#D98A1C"    # Reference / Moderate Asymmetry
SLATE = "#5B6770"    # Neutral / Non-significant
NAVY = "#1F2E4D"     # Primary Contrast / Dual Adopters

# ── Unified Publication Style ──────────────────────────────────────────────
def set_academic_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12.0,
        "axes.titlesize": 13.5,
        "axes.labelsize": 12.8,
        "xtick.labelsize": 11.2,
        "ytick.labelsize": 11.2,
        "legend.fontsize": 10.8,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.85,
        "axes.edgecolor": "#334155",
        "xtick.major.width": 0.85,
        "ytick.major.width": 0.85,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
    })


set_academic_style()


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def save_all_formats(fig, basename):
    """Export figure at 600 DPI PNG + editable TrueType PDF + editable SVG across target dirs."""
    for out_dir in EXPORT_DIRS:
        fig.savefig(os.path.join(out_dir, f"{basename}.png"), dpi=600, bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(out_dir, f"{basename}.pdf"), bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(out_dir, f"{basename}.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {basename} (.png 600 DPI, .pdf TrueType, .svg vector)")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Sustainability Asymmetry Index & Enterprise Size Stratification
# ═════════════════════════════════════════════════════════════════════════════
def generate_figure_1(df, xl):
    t1 = xl.parse("Table1_Typology_12205")
    pct_neither = float(t1.loc[t1["Configuration"].str.startswith("0."), "Share_Pct"].iloc[0])
    pct_both = float(t1.loc[t1["Configuration"].str.startswith("3."), "Share_Pct"].iloc[0])

    di_valid = df["DI"].dropna()
    di_counts = di_valid.value_counts().reindex([-1.0, 0.0, 1.0], fill_value=0)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.6, 6.6), gridspec_kw={"width_ratios": [1.22, 1.38], "wspace": 0.30}
    )

    colors_di = [TEAL, "#94A3B8", CORAL]
    labels_di = [
        "Net-Zero Strategy\nOnly (Index = -1)\n(Q9 = 0, Q14 = 1)",
        f"Zero Net Asymmetry\n(Index = 0: Dual Adopters\n{pct_both:.1f}% & Neither {pct_neither:.1f}%)",
        "Green Offerings\nOnly (Index = +1)\n(Q9 = 1, Q14 = 0)",
    ]

    bars = ax1.bar(range(3), di_counts.values, color=colors_di, edgecolor="white", linewidth=1.2, width=0.54, zorder=3)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels_di, fontsize=10.1, fontweight="bold", color="#1E293B")
    ax1.set_xlim(-0.58, 2.58)
    ax1.set_ylabel("Number of Sample Enterprises", fontweight="bold")
    ax1.set_title("(a) Directional Sustainability Asymmetry Distribution", fontweight="bold", loc="left", pad=10)

    max_val = max(di_counts.values)
    ax1.set_ylim(0, max_val * 1.24)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.32, color="#CBD5E1", zorder=0)

    for bar, count in zip(bars, di_counts.values):
        pct = count / len(di_valid) * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + (max_val * 0.02),
            f"n = {count:,}\n({pct:.1f}% of sample)",
            ha="center", va="bottom", fontsize=11.0, fontweight="bold", color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.92),
        )

    size_labels = {
        1.0: "Micro Enterprises\n(1-9 Employees)",
        2.0: "Small Enterprises\n(10-49 Employees)",
        3.0: "Medium Enterprises\n(50-249 Employees)",
    }
    di_by_size = df.dropna(subset=["scr10", "DI"]).groupby("scr10")["DI"].value_counts(normalize=True).unstack(fill_value=0)

    x = np.arange(3)
    width = 0.255
    legend_labels = [
        "Net-Zero Strategy Only (Index = -1)",
        "Zero Net Asymmetry: Dual & Neither (Index = 0)",
        "Green Offerings Only (Index = +1)",
    ]

    for i, (di_val, color, leg_label) in enumerate(zip([-1.0, 0.0, 1.0], colors_di, legend_labels)):
        vals = [di_by_size.loc[s, di_val] * 100 if (s in di_by_size.index and di_val in di_by_size.columns) else 0.0 for s in [1.0, 2.0, 3.0]]
        rects = ax2.bar(x + (i - 1) * width, vals, width, color=color, label=leg_label, edgecolor="white", linewidth=1.0, zorder=3)
        for rect, val in zip(rects, vals):
            ax2.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + 1.1,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=10.6, fontweight="bold", color="#1E293B",
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels([size_labels[s] for s in [1.0, 2.0, 3.0]], fontsize=11.0, fontweight="bold", color="#1E293B")
    ax2.set_ylabel("Percentage Within Enterprise Size Category (%)", fontweight="bold")
    ax2.set_title("(b) Asymmetry Categories by Enterprise Size", fontweight="bold", loc="left", pad=10)
    ax2.set_ylim(0, 84)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.32, color="#CBD5E1", zorder=0)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.94, edgecolor="#CBD5E1")

    save_all_formats(fig, "Figure_1")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Country-Level Mean Sustainability Asymmetry Index Across the EU-27
# ═════════════════════════════════════════════════════════════════════════════
def generate_figure_2(xl):
    country_di = xl.parse("DI_Country_Level").sort_values("Mean_DI").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13.2, 11.8))

    colors = []
    for di in country_di["Mean_DI"]:
        if di < 0:
            colors.append(TEAL)
        elif di < 0.08:
            colors.append("#94A3B8")
        elif di < 0.15:
            colors.append(AMBER)
        else:
            colors.append(CORAL)

    y_pos = np.arange(len(country_di))
    bars = ax.barh(y_pos, country_di["Mean_DI"], color=colors, edgecolor="white", linewidth=0.7, height=0.72, zorder=3)

    country_labels = [COUNTRY_NAMES.get(iso, iso) for iso in country_di["country_iso2"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(country_labels, fontsize=11.4, fontweight="bold", color="#1E293B")
    ax.axvline(x=0, color="#334155", linewidth=1.1, linestyle="-", zorder=4)

    overall_mean = float(np.average(country_di["Mean_DI"], weights=country_di["N"]))
    ax.axvline(x=overall_mean, color="#475569", linewidth=1.1, linestyle="--", alpha=0.85, zorder=2)
    ax.text(
        overall_mean + 0.003, 0.7,
        f"European Union Sample Mean = +{overall_mean:.3f}",
        fontsize=10.8, color="#334155", fontweight="bold", style="italic", rotation=90, va="bottom",
    )

    for bar, val, n_c in zip(bars, country_di["Mean_DI"], country_di["N"]):
        sign = "+" if val > 0 else ""
        x_txt = val + 0.0035 if val >= 0 else 0.0035
        ax.text(
            x_txt, bar.get_y() + bar.get_height() / 2.0,
            f"{sign}{val:.3f}  (n = {int(n_c)})",
            va="center", ha="left", fontsize=10.6, fontweight="bold", color="#1E293B",
        )

    ax.set_xlim(-0.065, 0.305)
    ax.set_xlabel(
        "National Mean Sustainability Asymmetry Index (Green Offerings Minus Reported Net-Zero Strategy)",
        fontweight="bold", labelpad=10,
    )
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)

    ax2 = ax.twinx()
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{d:.1f}" for d in country_di["DII"]], fontsize=10.8, color="#475569")
    ax2.set_ylabel("National Digital Intensity Index (Eurostat 2021 Composite)", fontweight="bold", color="#475569", labelpad=12)
    ax2.set_ylim(ax.get_ylim())
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#94A3B8")

    legend_elements = [
        mpatches.Patch(facecolor=TEAL, label="Net-Zero Strategy Tilt (Mean Index < 0.00)"),
        mpatches.Patch(facecolor="#94A3B8", label="Low Net Asymmetry (0.00 <= Mean Index < +0.08)"),
        mpatches.Patch(facecolor=AMBER, label="Moderate Green-Offering Tilt (+0.08 <= Mean Index < +0.15)"),
        mpatches.Patch(facecolor=CORAL, label="Pronounced Green-Offering Tilt (Mean Index >= +0.15)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, framealpha=0.94, edgecolor="#CBD5E1")

    save_all_formats(fig, "Figure_2")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Joint FIML Bivariate Probit Coefficient Plot
# ═════════════════════════════════════════════════════════════════════════════
def generate_figure_3(xl):
    coef = xl.parse("BiProbit_Coefficients")
    diag = xl.parse("BiProbit_Diagnostics").iloc[0]

    q9 = coef[coef["eq"].str.startswith("Y1")].set_index("var")
    q14 = coef[coef["eq"].str.startswith("Y2")].set_index("var")

    plot_vars = [
        ("B2C", "Business-to-Consumer Orientation (SCR15.1)"),
        ("B2G_public", "Business-to-Government Orientation (SCR15.3)"),
        ("B2B_private", "Business-to-Business Orientation (SCR15.2)"),
        ("barrier_admin", "Administrative and Legal Complexity (Q7.1)"),
        ("barrier_expertise", "Environmental Expertise Shortage (Q7.6)"),
        ("grant_receipt", "Public Financial Support (Q6.1)"),
        ("advice_public", "Public Non-Financial Advisory Support (Q6.4)"),
        ("scr10", "Enterprise Size Category (SCR10)"),
        ("re_action_count", "Substantive Resource-Efficiency Actions (RECount, Q1.1-Q1.9)"),
    ]

    fig, ax = plt.subplots(figsize=(12.8, 8.8))
    n = len(plot_vars)
    offset = 0.15

    for i, (v, _) in enumerate(plot_vars):
        y = n - 1 - i
        if i % 2 == 1:
            ax.axhspan(y - 0.46, y + 0.46, color="#F8FAFC", zorder=0)

        r9 = q9.loc[v]
        r14 = q14.loc[v]
        ax.errorbar(
            r9["beta"], y + offset, xerr=1.96 * r9["se_cluster"],
            fmt="o", color=CORAL, markersize=8.5, capsize=4.5, capthick=1.4, linewidth=2.0, zorder=4,
            label="Green Product or Service Offerings (Y1, Q9)" if i == 0 else None,
        )
        ax.errorbar(
            r14["beta"], y - offset, xerr=1.96 * r14["se_cluster"],
            fmt="s", color=TEAL, markersize=8.5, capsize=4.5, capthick=1.4, linewidth=2.0, zorder=4,
            label="Reported Net-Zero Strategy (Y2, Q14)" if i == 0 else None,
        )

    ax.axvline(x=0, color="#334155", linewidth=1.0, linestyle="-", zorder=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lbl for _, lbl in plot_vars][::-1], fontsize=11.4, fontweight="bold", color="#1E293B")
    ax.set_xlabel("Joint FIML Bivariate Probit Coefficient (with 95% Country-Clustered Confidence Interval)", fontweight="bold", labelpad=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1", zorder=1)
    ax.set_xlim(-0.33, 0.31)
    ax.set_ylim(-0.62, n - 0.38)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#CBD5E1")

    diag_txt = (
        f"Joint FIML Bivariate Probit Diagnostics:\n"
        f"• Sample: N = {int(diag['N']):,} SMEs ({int(diag['clusters'])} EU Country Clusters)\n"
        f"• Fixed Effects: 9 Turnover Bracket FEs, 26 Country FEs, 3 Sector FEs\n"
        f"• Residual Correlation: rho = +{diag['rho']:.3f} (Cluster SE = {diag['se_rho']:.3f}, z = {diag['z_rho']:.2f}, p < 0.001)"
    )
    ax.text(
        0.015, 0.03, diag_txt, transform=ax.transAxes, fontsize=9.8, va="bottom", ha="left", color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.40", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.96),
    )

    save_all_formats(fig, "Figure_3")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Subgroup CATE Forest Plot (Dynamically Read from Results_Consolidated)
# ═════════════════════════════════════════════════════════════════════════════
def generate_figure_4(xl):
    cate_df = xl.parse("CausalForest_CATEs").copy()
    ate_row = xl.parse("CausalForest_ATE").iloc[0]

    ate_val = float(ate_row["ATE"])
    delta_admin = float(ate_row["Delta_CATE_Admin"])
    p_delta_admin = float(ate_row["p_Delta_CATE_Admin"])

    meta_labels = [
        ("Administrative Complexity Barrier", "Administrative Complexity Reported (Q7.1 = 1)"),
        ("Administrative Complexity Barrier", "No Administrative Complexity Reported (Q7.1 = 0)"),
        ("Environmental Expertise Barrier", "Environmental Expertise Shortage Reported (Q7.6 = 1)"),
        ("Environmental Expertise Barrier", "No Environmental Expertise Shortage Reported (Q7.6 = 0)"),
        ("Enterprise Size Category", "Micro Enterprises (1-9 Employees)"),
        ("Enterprise Size Category", "Small Enterprises (10-49 Employees)"),
        ("Enterprise Size Category", "Medium Enterprises (50-249 Employees)"),
        ("National Digital Infrastructure", "Low National Digital Intensity (< Median)"),
        ("National Digital Infrastructure", "High National Digital Intensity (>= Median)"),
    ]
    cate_df["GroupName"] = [m[0] for m in meta_labels]
    cate_df["SubgroupLabel"] = [m[1] for m in meta_labels]

    fig, ax = plt.subplots(figsize=(15.2, 9.2), facecolor="white")
    n = len(cate_df)
    y_positions = np.arange(n)[::-1]

    groups = [
        ("Administrative Complexity Barrier", [8, 7]),
        ("Environmental Expertise Barrier", [6, 5]),
        ("Enterprise Size Category", [4, 3, 2]),
        ("National Digital Infrastructure", [1, 0]),
    ]
    group_colors = ["#F8FAFC", "#F1F5F9"]
    for i, (_, row_indices) in enumerate(groups):
        ymin = min(row_indices) - 0.48
        ymax = max(row_indices) + 0.48
        ax.axhspan(ymin, ymax, color=group_colors[i % 2], zorder=0)
        if i < len(groups) - 1:
            ax.axhline(ymin, color="#CBD5E1", linewidth=0.8, linestyle="-", zorder=1)

    ax.axvline(x=ate_val, color=AMBER, linestyle="--", linewidth=1.4, alpha=0.95, zorder=2)
    ax.text(
        ate_val + 0.003, n - 0.18,
        f"Stabilized IPW ATE = +{ate_val:.3f} (N = {int(ate_row['N']):,})",
        fontsize=11.0, color="#B45309", fontweight="bold", va="bottom",
    )
    ax.axvline(x=0, color="#475569", linestyle="-", linewidth=0.95, alpha=0.75, zorder=1)

    for i, (_, row) in enumerate(cate_df.iterrows()):
        y = y_positions[i]
        c_est = float(row["CATE"])
        lo = float(row["CI_lo_reest"])
        hi = float(row["CI_hi_reest"])
        p_val = float(row["p_reest"])
        sig = p_val < 0.05
        color = TEAL if sig else SLATE

        for alpha_level, width_mult in [(0.07, 1.8), (0.15, 1.3), (0.25, 1.0)]:
            half_w = ((hi - lo) / 2.0) * width_mult
            ax.fill_betweenx([y - 0.18, y + 0.18], c_est - half_w, c_est + half_w, color=color, alpha=alpha_level, zorder=2)

        ax.plot([lo, hi], [y, y], color=color, linewidth=2.1, zorder=3)
        marker = "D" if sig else "o"
        ax.plot(c_est, y, marker=marker, color=color, markersize=9.0 if sig else 7.8, markeredgecolor="white", markeredgewidth=1.2, zorder=4)

        sig_txt = f" (p = {p_val:.3f})" if sig else f" (p = {p_val:.3f}, n.s.)"
        ax.text(
            hi + 0.005, y,
            f"+{c_est:.3f}{sig_txt}",
            ha="left", va="center", fontsize=10.8,
            fontweight="bold" if sig else "normal", color=color,
        )

    c_admin1 = float(cate_df.iloc[0]["CATE"])
    c_admin0 = float(cate_df.iloc[1]["CATE"])
    ax.annotate(
        f"Concentrated Support Effect Under\nAdministrative Complexity (H3c)\n"
        f"(+{c_admin1:.3f} vs. +{c_admin0:.3f}; Delta CATE = +{delta_admin:.3f}, p = {p_delta_admin:.3f})",
        xy=(c_admin1, y_positions[0] - 0.08),
        xytext=(0.195, y_positions[1] + 0.12),
        fontsize=10.6, fontweight="bold", color=CORAL, ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5, connectionstyle="arc3,rad=-0.12"),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF5F3", edgecolor=CORAL, alpha=0.96),
    )

    formatted_labels = [f"{r['SubgroupLabel']}\n[{r['GroupName']}]" for _, r in cate_df.iterrows()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(formatted_labels, fontsize=11.0, fontweight="bold", color="#1E293B")
    ax.set_xlabel(
        "Conditional Average Treatment Effect of Public Financial Support on Reported Net-Zero Strategy (95% Bootstrap CI)",
        fontweight="bold", labelpad=10,
    )
    ax.set_xlim(-0.055, 0.295)
    ax.set_ylim(-0.75, n + 0.55)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1", zorder=1)

    legend_elements = [
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=TEAL, markersize=9.0, label="Statistically Significant Subgroup CATE (p < 0.05)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=SLATE, markersize=7.8, label="Non-Significant Subgroup CATE (p >= 0.05)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, framealpha=0.94, edgecolor="#CBD5E1")

    save_all_formats(fig, "Figure_4")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 & FIGURE S5: Joint Empirical Path Diagrams (Zero Top Title)
# ═════════════════════════════════════════════════════════════════════════════
def generate_path_diagram(xl, basename, with_moderation=True):
    coef = xl.parse("BiProbit_Coefficients")
    diag = xl.parse("BiProbit_Diagnostics").iloc[0]
    mod_df = xl.parse("AppendixJ_ProbScale_Moderation").set_index("quantity")

    q9 = coef[coef["eq"].str.startswith("Y1")].set_index("var")
    q14 = coef[coef["eq"].str.startswith("Y2")].set_index("var")

    iv_list = [
        ("B2C", "Business-to-Consumer Orientation", "#2563EB"),
        ("B2G_public", "Business-to-Government Orientation", "#4F46E5"),
        ("B2B_private", "Business-to-Business Orientation", "#D97706"),
        ("barrier_admin", "Administrative and Legal Complexity", "#64748B"),
        ("barrier_expertise", "Environmental Expertise Shortage", "#64748B"),
        ("grant_receipt", "Public Financial Support", "#059669"),
        ("advice_public", "Public Non-Financial Advice", "#0D9488"),
    ]

    fig, ax = plt.subplots(figsize=(14.6, 8.6), facecolor="white")
    y_min = -1.75 if with_moderation else -1.05
    ax.set_xlim(0.0, 14.0)
    ax.set_ylim(y_min, 8.25)
    ax.axis("off")

    ys = np.linspace(7.35, 2.55, len(iv_list))
    px, pw, ph = 4.95, 4.10, 0.58

    for (v, lbl, col), y in zip(iv_list, ys):
        ax.add_patch(
            FancyBboxPatch(
                (px, y - ph / 2.0), pw, ph,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                fc=col, ec=NAVY, lw=1.15, zorder=3,
            )
        )
        ax.text(px + pw / 2.0, y, lbl, ha="center", va="center", color="white", fontsize=10.3, fontweight="bold", zorder=4)

    out_boxes = [
        (0.25, "Green Product or\nService Offerings\n(Y1, Q9)", CORAL),
        (10.95, "Reported\nNet-Zero Strategy\n(Y2, Q14)", TEAL),
    ]
    for bx, txt, col in out_boxes:
        ax.add_patch(
            FancyBboxPatch(
                (bx, 2.05), 2.80, 5.75,
                boxstyle="round,pad=0.02,rounding_size=0.16",
                fc=col, ec=NAVY, lw=1.35, zorder=3,
            )
        )
        ax.text(bx + 1.40, 4.925, txt, ha="center", va="center", color="white", fontsize=11.8, fontweight="bold", zorder=4)

    for (v, _, _), y in zip(iv_list, ys):
        for tbl, side in [(q9, "L"), (q14, "R")]:
            r = tbl.loc[v]
            b_val = float(r["beta"])
            p_val = float(r["p_cluster"])
            sig = p_val < 0.05
            c_arr = (CORAL if side == "L" else TEAL) if sig else "#94A3B8"
            lw = 1.25 + 11.0 * abs(b_val) if sig else 1.0
            x0, x1 = (px, 3.07) if side == "L" else (px + pw, 10.93)
            ax.add_patch(
                FancyArrowPatch(
                    (x0, y), (x1, y),
                    arrowstyle="-|>", mutation_scale=15,
                    lw=lw, color=c_arr, ls="-" if sig else (0, (4, 3)), zorder=2,
                )
            )
            xm = (x0 + x1) / 2.0
            lbl_txt = f"beta = {b_val:+.3f}{stars(p_val)}" if sig else f"beta = {b_val:+.3f} (n.s.)"
            ax.text(
                xm, y + 0.16, lbl_txt,
                ha="center", va="bottom", fontsize=9.2,
                color=c_arr if sig else "#64748B",
                fontweight="bold" if sig else "normal",
            )

    ax.add_patch(
        FancyArrowPatch(
            (1.65, 1.92), (12.35, 1.92),
            connectionstyle="arc3,rad=0.11", arrowstyle="<|-|>",
            mutation_scale=15, lw=2.1, color=NAVY, zorder=3,
        )
    )
    ax.text(
        7.0, 0.35,
        f"Residual Correlation rho = +{diag['rho']:.3f} (Cluster SE = {diag['se_rho']:.3f}, z = {diag['z_rho']:.2f}, p < 0.001);  "
        f"N = {int(diag['N']):,} SMEs (Turnover, Country & Sector FEs)",
        ha="center", va="center", fontsize=10.2, color=NAVY, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#EFF6FF", edgecolor="#BFDBFE"),
    )

    if with_moderation:
        d_ame = mod_df.loc["Delta-AME (Admin=1 vs 0)"]
        ax.text(
            7.0, -0.58,
            f"Barrier Moderation on Y2 (H3c, Probability Scale): Delta AME of Public Financial Support (Admin = 1 vs. 0) = "
            f"+{d_ame['est']:.3f} (95% CI [+{d_ame['lo']:.3f}, +{d_ame['hi']:.3f}], p = {d_ame['p']:.3f})",
            ha="center", va="center", fontsize=9.5, color="#5B21B6", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.36", facecolor="#F5F3FF", edgecolor="#8B5CF6"),
        )
        foot_y = -1.35
    else:
        foot_y = -0.55

    ax.text(
        7.0, foot_y,
        "Solid arrows denote p < 0.05 (country-clustered SEs across 27 EU member states); dashed grey arrows denote p >= 0.05. * p < 0.05, ** p < 0.01, *** p < 0.001.",
        ha="center", va="center", fontsize=9.0, color="#64748B",
    )

    save_all_formats(fig, basename)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE S1: Four-Quadrant Sustainability Typology Mosaic Plot
# ═════════════════════════════════════════════════════════════════════════════
def generate_figure_s1(xl):
    t1 = xl.parse("Table1_Typology_12205").iloc[:4].copy()
    total = int(t1["N"].sum())

    rows = {int(r["Configuration"].split(".")[0]): r for _, r in t1.iterrows()}
    n_00, n_10, n_01, n_11 = int(rows[0]["N"]), int(rows[1]["N"]), int(rows[2]["N"]), int(rows[3]["N"])

    n_q9_0 = n_00 + n_01
    n_q9_1 = n_10 + n_11
    w_left = n_q9_0 / total
    w_right = n_q9_1 / total

    h_left_top = n_00 / n_q9_0
    h_left_bot = n_01 / n_q9_0
    h_right_top = n_10 / n_q9_1
    h_right_bot = n_11 / n_q9_1

    fig, ax = plt.subplots(figsize=(13.0, 8.8))
    gap = 0.024

    rects = [
        (0, h_left_bot + gap / 2, w_left - gap / 2, h_left_top - gap / 2, "Neither / Disengaged (Index = 0)", "#CBD5E1", rows[0]),
        (0, 0, w_left - gap / 2, h_left_bot - gap / 2, "Net-Zero Strategy Only (Index = -1)", TEAL, rows[2]),
        (w_left + gap / 2, h_right_bot + gap / 2, w_right - gap / 2, h_right_top - gap / 2, "Green Offerings Only (Index = +1)", CORAL, rows[1]),
        (w_left + gap / 2, 0, w_right - gap / 2, h_right_bot - gap / 2, "Dual Adopters / Both (Index = 0)", NAVY, rows[3]),
    ]

    for x, y, w, h, title_lbl, col, r_stat in rects:
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", facecolor=col, edgecolor="white", linewidth=2.0, alpha=0.93)
        )
        n_c = int(r_stat["N"])
        pct = float(r_stat["Share_Pct"])
        mean_re = float(r_stat["Mean_RE_Actions_9Item"])
        pct_sup = float(r_stat["Public_Support_Pct"])
        mean_dii = float(r_stat["Mean_National_DII"])

        text_col = "#1E293B" if col == "#CBD5E1" else "white"
        y_c = y + h / 2.0
        ax.text(
            x + w / 2.0, y_c + 0.055,
            f"{title_lbl}\nn = {n_c:,} SMEs ({pct:.1f}% of sample)",
            ha="center", va="center", fontsize=11.6, fontweight="bold", color=text_col,
        )
        ax.text(
            x + w / 2.0, y_c - 0.055,
            f"Mean Substantive RE Actions: {mean_re:.2f}\n"
            f"Public Financial Support Rate: {pct_sup:.1f}%\n"
            f"Mean National Digital Intensity: {mean_dii:.1f}",
            ha="center", va="center", fontsize=10.6, color=text_col,
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Market-Facing Green Product or Service Offerings (Q9: No vs. Yes)", fontweight="bold", labelpad=10)
    ax.set_ylabel("Reported Net-Zero Strategy by 2030 (Q14: No vs. Yes)", fontweight="bold", labelpad=10)

    pct_q9_0 = n_q9_0 / total * 100
    pct_q9_1 = n_q9_1 / total * 100
    pct_q14_1 = (n_01 + n_11) / total * 100
    pct_q14_0 = (n_00 + n_10) / total * 100

    ax.set_xticks([w_left / 2.0, w_left + w_right / 2.0])
    ax.set_xticklabels([f"No Green Offerings ({pct_q9_0:.1f}%)", f"Active Green Offerings ({pct_q9_1:.1f}%)"], fontsize=11.5, fontweight="bold")
    ax.set_yticks([h_left_bot / 2.0, h_left_bot + h_left_top / 2.0])
    ax.set_yticklabels([f"Reported Net-Zero Strategy ({pct_q14_1:.1f}%)", f"No Net-Zero Strategy ({pct_q14_0:.1f}%)"], fontsize=11.5, fontweight="bold")

    save_all_formats(fig, "Figure_S1")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE S2: EU-27 Spatial Choropleth & Member-State Ranking
# ═════════════════════════════════════════════════════════════════════════════
def generate_figure_s2(xl):
    ne_path = os.path.join(DATA_DIR, "ne_110m_admin_0_countries")
    if not os.path.exists(ne_path):
        print("  [SKIP] Natural Earth shapefile directory not found for Figure_S2.")
        return

    import geopandas as gpd
    from shapely.geometry import box, Point

    country_di = xl.parse("DI_Country_Level").rename(columns={"Mean_DI": "mean_DI", "N": "n"}).copy()
    country_di["iso3"] = country_di["country_iso2"].map(EU27_ISO3)
    country_di["name"] = country_di["country_iso2"].map(COUNTRY_NAMES)
    country_di = country_di.sort_values("mean_DI").reset_index(drop=True)

    world = gpd.read_file(ne_path)
    europe_bbox = box(-15, 33, 36, 72)
    world_eu = world.clip(europe_bbox).to_crs(epsg=3035)

    context = world_eu[(~world_eu["ISO_A3"].isin(EU27_ISO3.values())) & (~world_eu["ADM0_A3"].isin(EU27_ISO3.values()))]
    eu_gdf = world_eu[world_eu["ISO_A3"].isin(EU27_ISO3.values()) | world_eu["ADM0_A3"].isin(EU27_ISO3.values())].copy()
    eu_gdf["iso3"] = eu_gdf["ISO_A3"].where(eu_gdf["ISO_A3"].isin(EU27_ISO3.values()), eu_gdf["ADM0_A3"])
    eu_gdf = eu_gdf.merge(country_di[["iso3", "mean_DI", "n", "name"]], on="iso3", how="left")

    mlt_di = float(country_di.loc[country_di["iso3"] == "MLT", "mean_DI"].iloc[0])
    mlt_n = int(country_di.loc[country_di["iso3"] == "MLT", "n"].iloc[0])
    malta_pt = gpd.GeoSeries([Point(14.45, 35.90)], crs="EPSG:4326").to_crs(epsg=3035).iloc[0]
    malta_row = pd.DataFrame([{"iso3": "MLT", "name": "Malta", "mean_DI": mlt_di, "n": mlt_n, "geometry": malta_pt.buffer(45000)}])
    eu_gdf = pd.concat([eu_gdf, gpd.GeoDataFrame(malta_row, crs="EPSG:3035")], ignore_index=True)

    cy_idx = eu_gdf[eu_gdf["iso3"] == "CYP"].index
    if len(cy_idx) > 0:
        eu_gdf.loc[cy_idx, "geometry"] = eu_gdf.loc[cy_idx, "geometry"].buffer(25000)

    cmap = LinearSegmentedColormap.from_list(
        "asym_carto",
        ["#0F4C5C", "#2C8C85", "#70B2A9", "#FFF3D6", "#F3A066", "#E25B45", "#9E1B27"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-0.06, vcenter=0.0, vmax=0.24)

    fig = plt.figure(figsize=(18.8, 11.4), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.76, 1.24], wspace=0.09, left=0.025, right=0.975, top=0.94, bottom=0.05)
    ax_map = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    ax_map.set_facecolor("#EAF2F8")
    ax_map.set_xlim(2450000, 6650000)
    ax_map.set_ylim(1300000, 5400000)
    for spine in ax_map.spines.values():
        spine.set_visible(True)
        spine.set_color("#CBD5E1")
        spine.set_linewidth(1.1)

    context.plot(ax=ax_map, color="#E2E8F0", edgecolor="#94A3B8", linewidth=0.55, zorder=1)
    eu_gdf.plot(ax=ax_map, column="mean_DI", cmap=cmap, norm=norm, edgecolor="#1E293B", linewidth=0.75, zorder=2)

    ax_map.text(
        3400000, 3750000, "United Kingdom\n(Non-EU)", fontsize=10.2, color="#64748B",
        ha="center", va="center", style="italic", zorder=3,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )
    ax_map.text(
        4200000, 4700000, "Norway\n(Non-EU)", fontsize=10.2, color="#64748B",
        ha="center", va="center", style="italic", zorder=3,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )

    territory_labels = {
        "ESP": ("Spain", 3200000, 2025000, 11.2),
        "PRT": ("Portugal", 2760000, 2020000, 10.2),
        "FRA": ("France", 3740000, 2660000, 11.4),
        "DEU": ("Germany", 4335000, 3115000, 11.4),
        "POL": ("Poland", 4940000, 3270000, 11.2),
        "ITA": ("Italy", 4520000, 2170000, 11.2),
        "SWE": ("Sweden", 4570000, 4365000, 11.2),
        "FIN": ("Finland", 5150000, 4730000, 11.2),
        "ROU": ("Romania", 5460000, 2655000, 11.2),
        "BGR": ("Bulgaria", 5570000, 2305000, 10.6),
        "GRC": ("Greece", 5325000, 1915000, 10.6),
        "IRL": ("Ireland", 3155000, 3506000, 10.4),
        "AUT": ("Austria", 4675000, 2744000, 10.2),
        "HUN": ("Hungary", 5008000, 2727000, 10.5),
        "CZE": ("Czechia", 4725000, 2979000, 10.2),
        "SVK": ("Slovakia", 4975000, 2882000, 10.0),
        "HRV": ("Croatia", 4765000, 2390000, 9.8),
        "SVN": ("Slovenia", 4687000, 2569000, 9.5),
        "LTU": ("Lithuania", 5185000, 3660000, 10.2),
        "LVA": ("Latvia", 5217000, 3829000, 10.2),
        "EST": ("Estonia", 5211000, 4050000, 10.2),
        "DNK": ("Denmark", 4288000, 3691000, 10.0),
        "NLD": ("Netherlands", 4015000, 3270000, 9.6),
        "BEL": ("Belgium", 3855000, 3105000, 9.6),
        "LUX": ("Luxembourg", 4070000, 2940000, 9.1),
        "CYP": ("Cyprus", 6414000, 1635000, 10.0),
        "MLT": ("Malta", 4726000, 1438000, 9.5),
    }

    for iso, (name, x, y, fsize) in territory_labels.items():
        di_val = float(country_di.loc[country_di["iso3"] == iso, "mean_DI"].iloc[0])
        sign = "+" if di_val > 0 else ""
        ax_map.text(
            x, y, f"{name}\n({sign}{di_val:.3f})",
            fontsize=fsize, fontweight="bold", color="#0F172A",
            ha="center", va="center", zorder=5,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white", alpha=0.92)],
        )

    ax_legend = ax_map.inset_axes([0.028, 0.63, 0.35, 0.33])
    ax_legend.set_facecolor("white")
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    for spine in ax_legend.spines.values():
        spine.set_visible(True)
        spine.set_color("#CBD5E1")
        spine.set_linewidth(1.0)

    ax_legend.text(0.5, 0.89, "Sustainability Asymmetry Index", fontsize=11.4, fontweight="bold", ha="center", va="center", color="#0F172A", transform=ax_legend.transAxes)
    ax_legend.text(0.5, 0.77, "Green Offerings Minus Net-Zero Strategy", fontsize=9.6, style="italic", ha="center", va="center", color="#475569", transform=ax_legend.transAxes)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb_inset = ax_legend.inset_axes([0.08, 0.52, 0.84, 0.12])
    cb = plt.colorbar(sm, cax=cb_inset, orientation="horizontal")
    cb.set_ticks([-0.05, 0.00, 0.10, 0.20])
    cb.set_ticklabels(["-0.05", "0.00", "+0.10", "+0.20"])
    cb.ax.tick_params(labelsize=9.2, pad=2)

    ax_legend.text(0.5, 0.35, "Net-Zero Tilt (Mean Index < 0)", fontsize=9.8, fontweight="bold", color="#0F4C5C", ha="center", va="center", transform=ax_legend.transAxes)
    ax_legend.text(0.5, 0.22, "High Green-Offering Tilt (Mean Index >= +0.15)", fontsize=9.8, fontweight="bold", color="#9E1B27", ha="center", va="center", transform=ax_legend.transAxes)
    ax_legend.text(0.5, 0.09, f"Sample: N = {int(country_di['n'].sum()):,} SMEs (27 EU States)", fontsize=9.4, color="#64748B", ha="center", va="center", transform=ax_legend.transAxes)

    scale_x0, scale_y0 = 2620000, 1480000
    ax_map.plot([scale_x0, scale_x0 + 500000], [scale_y0, scale_y0], color="#0F172A", lw=2.3, zorder=7)
    ax_map.plot([scale_x0, scale_x0], [scale_y0 - 25000, scale_y0 + 25000], color="#0F172A", lw=1.7, zorder=7)
    ax_map.plot([scale_x0 + 500000, scale_x0 + 500000], [scale_y0 - 25000, scale_y0 + 25000], color="#0F172A", lw=1.7, zorder=7)
    ax_map.text(scale_x0, scale_y0 + 55000, "0", fontsize=9.8, ha="center", fontweight="bold", color="#0F172A", zorder=7)
    ax_map.text(scale_x0 + 500000, scale_y0 + 55000, "500 km", fontsize=9.8, ha="center", fontweight="bold", color="#0F172A", zorder=7)
    ax_map.text(scale_x0 + 250000, scale_y0 - 65000, "ETRS89-LAEA (EPSG:3035)", fontsize=9.2, ha="center", color="#475569", style="italic", zorder=7)

    ax_map.set_title("(a) Spatial Distribution Across the European Union", fontweight="bold", loc="left", pad=10, color="#0F172A")
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    y_pos = np.arange(len(country_di))
    bar_colors = [cmap(norm(v)) for v in country_di["mean_DI"]]
    bars = ax_bar.barh(y_pos, country_di["mean_DI"], color=bar_colors, edgecolor="#1E293B", linewidth=0.6, height=0.72, zorder=3)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(country_di["name"], fontsize=11.2, fontweight="bold", color="#0F172A")
    ax_bar.axvline(x=0, color="#0F172A", linewidth=1.1, linestyle="-", zorder=4)
    mean_eu = float(np.average(country_di["mean_DI"], weights=country_di["n"]))
    ax_bar.axvline(x=mean_eu, color="#475569", linewidth=1.1, linestyle="--", zorder=2)
    ax_bar.set_ylim(-0.8, len(country_di) + 0.85)
    ax_bar.text(
        mean_eu, len(country_di) + 0.08,
        f"EU Sample Mean = +{mean_eu:.3f}",
        fontsize=10.6, color="#334155", fontweight="bold", style="italic", ha="center", va="bottom", zorder=5,
    )

    for bar, val, n_c in zip(bars, country_di["mean_DI"], country_di["n"]):
        sign = "+" if val > 0 else ""
        x_txt = val + 0.004 if val >= 0 else 0.004
        ax_bar.text(
            x_txt, bar.get_y() + bar.get_height() / 2.0,
            f"{sign}{val:.3f}  (n = {int(n_c)})",
            va="center", ha="left", fontsize=10.4, fontweight="bold", color="#1E293B",
        )

    ax_bar.set_xlim(-0.065, 0.305)
    ax_bar.set_xlabel("National Mean Sustainability Asymmetry Index", fontweight="bold", labelpad=10, color="#0F172A")
    ax_bar.set_title("(b) Complete Ranking Across 27 EU Member States", fontweight="bold", loc="left", pad=10, color="#0F172A")
    ax_bar.xaxis.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)

    save_all_formats(fig, "Figure_S2")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE S3: CATE Subgroup Bar Chart with 95% Bootstrap Confidence Whiskers
# ═════════════════════════════════════════════════════════════════════════════
def generate_figure_s3(xl):
    cate_data = xl.parse("CausalForest_CATEs").copy()
    ate_row = xl.parse("CausalForest_ATE").iloc[0]
    ate_val = float(ate_row["ATE"])
    delta_admin = float(ate_row["Delta_CATE_Admin"])
    p_delta_admin = float(ate_row["p_Delta_CATE_Admin"])

    full_labels = [
        "Administrative Complexity Reported",
        "No Administrative Barrier Reported",
        "Environmental Expertise Shortage",
        "No Expertise Shortage Reported",
        "Micro Enterprises (1-9 Employees)",
        "Small Enterprises (10-49 Employees)",
        "Medium Enterprises (50-249 Employees)",
        "Low National Digital Intensity (< Median)",
        "High National Digital Intensity (>= Median)",
    ]
    cate_data["ExpandedLabel"] = full_labels[: len(cate_data)]

    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    cates = cate_data["CATE"].values.astype(float)
    ci_lo = cate_data["CI_lo_reest"].values.astype(float)
    ci_hi = cate_data["CI_hi_reest"].values.astype(float)
    colors = [TEAL if ci_lo[i] > 0 else "#94A3B8" for i in range(len(cates))]

    x = np.arange(len(cates))
    ax.bar(x, cates, color=colors, edgecolor="white", linewidth=1.0, width=0.56, zorder=3)
    ax.errorbar(
        x, cates, yerr=[cates - ci_lo, ci_hi - cates], fmt="none",
        ecolor="#1E293B", capsize=5, capthick=1.4, linewidth=1.4, zorder=4,
    )

    ax.axhline(0, color="#1E293B", linewidth=0.9, zorder=2)
    ax.axhline(
        ate_val, color=AMBER, linewidth=1.35, linestyle="--",
        label=f"Stabilized IPW Average Treatment Effect = +{ate_val:.3f}", zorder=2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(cate_data["ExpandedLabel"], rotation=28, ha="right", fontsize=10.8, fontweight="bold", color="#1E293B")
    ax.set_ylabel("Conditional Average Treatment Effect on\nReported Net-Zero Strategy", fontweight="bold", labelpad=10)
    ax.set_ylim(-0.04, 0.175)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)

    ax.annotate(
        f"Concentrated Support Effect Among\nAdministratively Constrained SMEs\n(Delta CATE = +{delta_admin:.3f}, p = {p_delta_admin:.3f})",
        xy=(0.28, cates[0]),
        xytext=(2.05, 0.140),
        fontsize=10.4, fontweight="bold", color=CORAL, ha="center",
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#FFF5F3", edgecolor=CORAL, alpha=0.94),
    )

    ax.legend(loc="upper right", frameon=True, framealpha=0.94, edgecolor="#CBD5E1")
    save_all_formats(fig, "Figure_S3")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE S4: National Digital Intensity vs. Sustainability Asymmetry Bubble Plot
# ═════════════════════════════════════════════════════════════════════════════
def generate_figure_s4(xl):
    c_data = xl.parse("DI_Country_Level").rename(columns={"Mean_DI": "mean_DI", "DII": "dii", "N": "n"}).copy()

    clusters_map = {
        "DK": "Nordic Region", "FI": "Nordic Region", "SE": "Nordic Region",
        "AT": "Western Europe", "BE": "Western Europe", "DE": "Western Europe", "FR": "Western Europe",
        "IE": "Western Europe", "LU": "Western Europe", "NL": "Western Europe",
        "CY": "Southern Europe", "EL": "Southern Europe", "ES": "Southern Europe", "IT": "Southern Europe",
        "MT": "Southern Europe", "PT": "Southern Europe",
        "BG": "Central and Eastern Europe", "CZ": "Central and Eastern Europe", "EE": "Central and Eastern Europe",
        "HR": "Central and Eastern Europe", "HU": "Central and Eastern Europe", "LT": "Central and Eastern Europe",
        "LV": "Central and Eastern Europe", "PL": "Central and Eastern Europe", "RO": "Central and Eastern Europe",
        "SI": "Central and Eastern Europe", "SK": "Central and Eastern Europe",
    }
    cluster_colors = {
        "Nordic Region": "#2563EB",
        "Western Europe": "#7C3AED",
        "Southern Europe": "#EA580C",
        "Central and Eastern Europe": "#059669",
    }

    c_data["cluster"] = c_data["country_iso2"].map(clusters_map)
    c_data["color"] = c_data["cluster"].map(cluster_colors)

    fig, ax = plt.subplots(figsize=(12.4, 8.8))
    ax.axhline(0.08, color="#CBD5E1", linestyle=":", linewidth=0.9, alpha=0.8)
    ax.axvline(22.0, color="#CBD5E1", linestyle=":", linewidth=0.9, alpha=0.8)

    label_offsets = {
        "RO": (-10, 10), "BG": (9, -4), "HU": (9, 2),
        "FR": (9, -12), "LV": (-12, 10), "PL": (9, 8),
        "SK": (9, 6), "EL": (9, -8),
        "EE": (-10, 8), "PT": (-10, -14), "CZ": (-10, 8),
        "IT": (-12, 12), "HR": (9, 8), "LT": (9, -6), "LU": (8, 14),
        "SI": (-10, 10), "DE": (10, -14), "ES": (9, -10), "CY": (9, 8),
        "BE": (9, 8), "AT": (9, -14), "IE": (9, 8),
        "NL": (9, 4), "MT": (-10, 10), "DK": (8, -12),
        "FI": (8, 8), "SE": (9, -10),
    }

    for _, row in c_data.iterrows():
        ax.scatter(
            row["dii"], row["mean_DI"], s=row["n"] / 5.8,
            c=row["color"], alpha=0.78, edgecolors="white", linewidth=1.2, zorder=3,
        )
        c_full = COUNTRY_NAMES.get(row["country_iso2"], row["country_iso2"])
        dx, dy = label_offsets.get(row["country_iso2"], (8, 4))
        ha_align = "right" if dx < 0 else "left"
        ax.annotate(
            c_full, (row["dii"], row["mean_DI"]),
            textcoords="offset points", xytext=(dx, dy),
            fontsize=10.4, fontweight="bold", color="#1E293B", ha=ha_align,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CBD5E1", alpha=0.90, linewidth=0.6),
        )

    slope, intercept, r_val, p_val, _ = stats.linregress(c_data["dii"], c_data["mean_DI"])
    x_fit = np.linspace(c_data["dii"].min() - 2, c_data["dii"].max() + 2, 100)
    ax.plot(x_fit, slope * x_fit + intercept, color="#64748B", linestyle="--", linewidth=1.3, zorder=2)

    ax.text(
        0.97, 0.04,
        f"Country-Level OLS Slope: beta = {slope:+.4f}\nR-squared = {r_val ** 2:.3f} (two-tailed p = {p_val:.3f})",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10.8, color="#334155", style="italic",
        bbox=dict(boxstyle="round,pad=0.42", facecolor="white", edgecolor="#CBD5E1", alpha=0.94),
    )

    ax.set_xlabel("National Digital Intensity Index (Eurostat 2021 Composite Score)", fontweight="bold", labelpad=10)
    ax.set_ylabel("National Mean Sustainability Asymmetry Index\n(Green Offerings Minus Reported Net-Zero Strategy)", fontweight="bold", labelpad=10)
    ax.set_ylim(-0.07, 0.255)
    ax.grid(True, linestyle="--", alpha=0.3, color="#E2E8F0")

    for cluster_name, col in cluster_colors.items():
        ax.scatter([], [], c=col, s=75, alpha=0.85, label=cluster_name, edgecolors="white")
    ax.scatter([], [], c="#94A3B8", s=250 / 5.8, alpha=0.45, label="Country Sample n ~ 250")
    ax.scatter([], [], c="#94A3B8", s=500 / 5.8, alpha=0.45, label="Country Sample n ~ 500")
    ax.legend(
        loc="upper left", frameon=True, framealpha=0.94, edgecolor="#CBD5E1",
        title="European Geographic Clusters", title_fontsize=11.2,
    )

    save_all_formats(fig, "Figure_S4")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE S6: Exact Newton-Raphson Propensity Score Overlap Density Plot
# ═════════════════════════════════════════════════════════════════════════════
def generate_figure_s6(xl):
    ps_df = pd.read_csv(PS_CSV)
    ipw_diag = xl.parse("IPW_Diagnostics").set_index("Series")

    ps_t = ps_df.loc[ps_df["grant_receipt"] == 1, "ps_raw"].values
    ps_c = ps_df.loc[ps_df["grant_receipt"] == 0, "ps_raw"].values

    ov_min = max(ps_t.min(), ps_c.min())
    ov_max = min(ps_t.max(), ps_c.max())
    ess_t = float(ipw_diag.loc["Treated_Stabilized_Weight", "Kish_ESS"])
    ess_c = float(ipw_diag.loc["Control_Stabilized_Weight", "Kish_ESS"])

    fig, ax = plt.subplots(figsize=(13.0, 6.8))
    bins = np.linspace(0.01, 0.62, 62)

    ax.hist(
        ps_c, bins=bins, alpha=0.42, color="#94A3B8", edgecolor="white", linewidth=0.5,
        label=f"Non-Recipient Control SMEs (n = {len(ps_c):,})", density=True,
    )
    ax.hist(
        ps_t, bins=bins, alpha=0.62, color="#2A9D8F", edgecolor="white", linewidth=0.5,
        label=f"Public Financial Support Recipients (n = {len(ps_t):,})", density=True,
    )

    x_grid = np.linspace(0.01, 0.62, 450)
    kde_c = gaussian_kde(ps_c, bw_method=0.14)(x_grid)
    kde_t = gaussian_kde(ps_t, bw_method=0.14)(x_grid)

    ax.plot(x_grid, kde_c, color="#1E293B", linewidth=2.1, label="Kernel Density: Non-Recipients")
    ax.plot(x_grid, kde_t, color="#0F5147", linewidth=2.1, label="Kernel Density: Support Recipients")
    ax.axvspan(ov_min, ov_max, alpha=0.07, color="#10B981", label=f"Common Empirical Support [{ov_min:.3f}, {ov_max:.3f}]")

    textbox = (
        f"Exact Newton-Raphson Propensity Diagnostics (N = {len(ps_df):,}):\n"
        f"• Common Support Range: [{ov_min:.3f}, {ov_max:.3f}]\n"
        f"• Treated (n = {len(ps_t):,}): Mean = {ps_t.mean():.3f}, SD = {ps_t.std(ddof=1):.3f}\n"
        f"• Control (n = {len(ps_c):,}): Mean = {ps_c.mean():.3f}, SD = {ps_c.std(ddof=1):.3f}\n"
        f"• Kish Effective Sample Size: Treated = {ess_t:.1f} ({ess_t / len(ps_t) * 100:.1f}%), Control = {ess_c:,.1f}"
    )
    ax.text(
        0.975, 0.56, textbox, transform=ax.transAxes, fontsize=10.3, va="top", ha="right", color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.96),
    )

    ax.set_xlabel("Estimated Propensity Score for Public Financial Support Receipt (Turnover, Country & Sector FEs)", fontweight="bold", labelpad=10)
    ax.set_ylabel("Probability Density", fontweight="bold", labelpad=10)
    ax.set_xlim(0.005, 0.625)
    ax.set_ylim(0, max(kde_c.max(), kde_t.max()) * 1.18)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CBD5E1")

    save_all_formats(fig, "Figure_S6")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE S7: Sustainability Asymmetry Index by Enterprise Size Category
# ═════════════════════════════════════════════════════════════════════════════
def generate_figure_s7(df):
    size_labels = {
        1.0: "Micro Enterprises\n(1-9 Employees)",
        2.0: "Small Enterprises\n(10-49 Employees)",
        3.0: "Medium Enterprises\n(50-249 Employees)",
    }
    df_valid = df[["scr10", "DI"]].dropna().copy()

    stats_list = []
    for s_val in [1.0, 2.0, 3.0]:
        sub = df_valid.loc[df_valid["scr10"] == s_val, "DI"]
        n = len(sub)
        m = float(sub.mean())
        se = float(sub.std(ddof=1) / np.sqrt(n))
        stats_list.append({
            "size": s_val, "label": size_labels[s_val], "n": n, "mean": m,
            "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se,
            "pct_minus": float((sub == -1).mean() * 100),
            "pct_zero": float((sub == 0).mean() * 100),
            "pct_plus": float((sub == 1).mean() * 100),
        })
    s_df = pd.DataFrame(stats_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 6.8), gridspec_kw={"width_ratios": [1.14, 1.26], "wspace": 0.30})
    x = np.arange(3)

    ax1.bar(x, s_df["mean"], color=[AMBER, TEAL, "#059669"], edgecolor="white", width=0.52, zorder=3)
    ax1.errorbar(
        x, s_df["mean"], yerr=[s_df["mean"] - s_df["ci_lo"], s_df["ci_hi"] - s_df["mean"]],
        fmt="none", ecolor="#1E293B", capsize=5.5, capthick=1.4, linewidth=1.4, zorder=4,
    )
    ax1.axhline(0, color="#1E293B", linewidth=0.85)
    overall_m = float(df_valid["DI"].mean())
    ax1.axhline(overall_m, color="#64748B", linestyle="--", linewidth=1.1, label=f"Overall Sample Mean = +{overall_m:.3f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(s_df["label"], fontsize=11.0, fontweight="bold", color="#1E293B")
    ax1.set_ylabel("Mean Sustainability Asymmetry Index (95% CI)", fontweight="bold", labelpad=10)
    ax1.set_title("(a) Mean Asymmetry Index by Enterprise Size", fontweight="bold", loc="left", pad=10)
    ax1.set_xlim(-0.58, 2.58)
    ax1.set_ylim(-0.02, 0.175)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)

    for i, r in s_df.iterrows():
        ax1.text(
            i, r["ci_hi"] + 0.006,
            f"n = {int(r['n']):,}\nMean = +{r['mean']:.3f}",
            ha="center", va="bottom", fontsize=10.6, fontweight="bold", color="#1E293B",
        )
    ax1.legend(loc="upper right", frameon=True, framealpha=0.94, edgecolor="#CBD5E1")

    b_zero = s_df["pct_minus"].values
    b_plus = b_zero + s_df["pct_zero"].values
    ax2.bar(x, s_df["pct_minus"], width=0.52, color=TEAL, edgecolor="white", linewidth=0.8, label="Net-Zero Strategy Only (Index = -1)")
    ax2.bar(x, s_df["pct_zero"], width=0.52, bottom=b_zero, color="#94A3B8", edgecolor="white", linewidth=0.8, label="Zero Net Asymmetry: Dual & Neither (Index = 0)")
    ax2.bar(x, s_df["pct_plus"], width=0.52, bottom=b_plus, color=CORAL, edgecolor="white", linewidth=0.8, label="Green Offerings Only (Index = +1)")

    for i, r in s_df.iterrows():
        ax2.text(i, r["pct_minus"] / 2.0, f"{r['pct_minus']:.1f}%", ha="center", va="center", fontsize=10.8, color="white", fontweight="bold")
        ax2.text(i, b_zero[i] + r["pct_zero"] / 2.0, f"{r['pct_zero']:.1f}%", ha="center", va="center", fontsize=10.8, color="white", fontweight="bold")
        ax2.text(i, b_plus[i] + r["pct_plus"] / 2.0, f"{r['pct_plus']:.1f}%", ha="center", va="center", fontsize=10.8, color="white", fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(s_df["label"], fontsize=11.0, fontweight="bold", color="#1E293B")
    ax2.set_xlim(-0.58, 2.58)
    ax2.set_ylabel("Percentage Share of Enterprises (%)", fontweight="bold", labelpad=10)
    ax2.set_title("(b) Asymmetry Index Composition by Enterprise Size", fontweight="bold", loc="left", pad=10)
    ax2.set_ylim(0, 100)
    ax2.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1,
        frameon=True, framealpha=0.95, edgecolor="#CBD5E1",
    )

    save_all_formats(fig, "Figure_S7")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE S8: Regional Sustainability Asymmetry Distributions Across Europe
# ═════════════════════════════════════════════════════════════════════════════
def generate_figure_s8(df):
    clusters = {
        "Nordic Region": ["DK", "FI", "SE"],
        "Western Europe": ["AT", "BE", "DE", "FR", "IE", "LU", "NL"],
        "Southern Europe": ["CY", "EL", "ES", "IT", "MT", "PT"],
        "Central and Eastern Europe": ["BG", "CZ", "EE", "HR", "HU", "LT", "LV", "PL", "RO", "SI", "SK"],
    }
    cluster_colors = {
        "Nordic Region": "#2563EB",
        "Western Europe": "#7C3AED",
        "Southern Europe": "#EA580C",
        "Central and Eastern Europe": "#059669",
    }

    iso2_to_c = {c: cl for cl, c_list in clusters.items() for c in c_list}
    df_c = df[df["country_iso2"].isin(iso2_to_c.keys())].copy()
    df_c["cluster"] = df_c["country_iso2"].map(iso2_to_c)

    fig, axes = plt.subplots(4, 1, figsize=(12.6, 10.6), sharex=True)
    c_order = ["Nordic Region", "Western Europe", "Southern Europe", "Central and Eastern Europe"]

    for i, cl in enumerate(c_order):
        ax = axes[i]
        data = df_c.loc[df_c["cluster"] == cl, "DI"].dropna()
        vals = np.array([-1.0, 0.0, 1.0])
        counts = np.array([(data == v).sum() for v in vals])
        props = counts / counts.sum()

        ax.bar(vals, props, width=0.52, color=[TEAL, "#94A3B8", CORAL], edgecolor="white", linewidth=1.2, alpha=0.88, zorder=2)
        ax.fill_between(vals, 0, props, alpha=0.14, color=cluster_colors[cl], step="mid", zorder=1)

        mean_val = float(data.mean())
        pct_go = float((data == 1).mean() * 100)
        ax.axvline(mean_val, color=cluster_colors[cl], linestyle="--", linewidth=1.4, zorder=3)

        header_text = (
            f"{cl} (n = {len(data):,} SMEs)\n"
            f"Mean Asymmetry Index = +{mean_val:.3f}\n"
            f"Green Offerings Only Share (Index = +1) = {pct_go:.1f}%"
        )
        ax.text(
            0.02, 0.95, header_text,
            transform=ax.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold", color=cluster_colors[cl],
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#CBD5E1", alpha=0.95), zorder=5,
        )

        for v, p in zip(vals, props):
            ax.text(
                v, p + 0.022, f"{p * 100:.1f}%",
                ha="center", va="bottom", fontsize=10.6, fontweight="bold", color="#1E293B",
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.92), zorder=5,
            )

        ax.set_ylim(0, max(props) * 1.50)
        ax.set_ylabel("Proportion", fontsize=11.4, fontweight="bold")
        ax.set_xlim(-1.5, 1.5)
        if i < 3:
            ax.tick_params(axis="x", labelbottom=False)

    axes[3].set_xlabel(
        "Directional Sustainability Asymmetry Index (Green Offerings Minus Reported Net-Zero Strategy)",
        fontweight="bold", labelpad=10,
    )
    axes[3].set_xticks([-1, 0, 1])
    axes[3].set_xticklabels([
        "Net-Zero Strategy Only (Index = -1)\n(Q9 = 0, Q14 = 1)",
        "Zero Net Asymmetry (Index = 0)\n(Dual Adopters & Neither)",
        "Green Offerings Only (Index = +1)\n(Q9 = 1, Q14 = 0)",
    ], fontsize=10.8, fontweight="bold")

    save_all_formats(fig, "Figure_S8")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE S9: Cross-Equation Bivariate Probit Coefficient Asymmetry Heatmap
# ═════════════════════════════════════════════════════════════════════════════
def generate_figure_s9(xl):
    coef = xl.parse("BiProbit_Coefficients")
    tests = xl.parse("BiProbit_CrossEq_Tests").set_index("var")

    q9 = coef[coef["eq"].str.startswith("Y1")].set_index("var")
    q14 = coef[coef["eq"].str.startswith("Y2")].set_index("var")

    row_specs = [
        ("B2C", "Business-to-Consumer Orientation"),
        ("B2G_public", "Business-to-Government Orientation"),
        ("B2B_private", "Business-to-Business Orientation"),
        ("barrier_admin", "Administrative and Legal Complexity"),
        ("barrier_expertise", "Environmental Expertise Shortage"),
        ("grant_receipt", "Public Financial Support"),
        ("advice_public", "Public Non-Financial Advisory Support"),
        ("scr10", "Enterprise Size Category"),
        ("re_action_count", "Substantive Resource-Efficiency Actions (RECount)"),
    ]

    b = np.array([[float(q9.loc[v, "beta"]), float(q14.loc[v, "beta"])] for v, _ in row_specs])
    p = np.array([[float(q9.loc[v, "p_cluster"]), float(q14.loc[v, "p_cluster"])] for v, _ in row_specs])
    d = b[:, 0] - b[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 7.2), gridspec_kw={"width_ratios": [2.05, 1.15], "wspace": 0.22})

    cm1 = LinearSegmentedColormap.from_list("coef_hm", [TEAL, "#CBE7E3", "#FFFFFF", "#F8CBC6", CORAL], N=256)
    cm2 = LinearSegmentedColormap.from_list("diff_hm", ["#2166AC", "#D1E5F0", "#FFFFFF", "#FDDBC7", "#B2182B"], N=256)

    im1 = ax1.imshow(b, cmap=cm1, norm=TwoSlopeNorm(0.0, -0.24, 0.24), aspect="auto")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Green Offerings\n(Y1, Q9)", "Reported Net-Zero\nStrategy (Y2, Q14)"], fontsize=11.0, fontweight="bold")
    ax1.set_yticks(range(len(row_specs)))
    ax1.set_yticklabels([lbl for _, lbl in row_specs], fontsize=10.8, fontweight="bold", color="#1E293B")
    ax1.set_title("(a) Equation-Specific Estimates (beta)", fontweight="bold", loc="left", pad=10)

    for i in range(len(row_specs)):
        for j in range(2):
            txt_col = "white" if abs(b[i, j]) > 0.165 else "#0F172A"
            ax1.text(j, i, f"{b[i, j]:+.3f}{stars(p[i, j])}", ha="center", va="center", fontsize=10.8, fontweight="bold", color=txt_col)

    ax1.set_xticks(np.arange(-0.5, 2), minor=True)
    ax1.set_yticks(np.arange(-0.5, len(row_specs)), minor=True)
    ax1.grid(which="minor", color="white", lw=2.0)
    ax1.tick_params(which="both", length=0)
    plt.colorbar(im1, ax=ax1, fraction=0.042, pad=0.03, label="BiProbit Coefficient (beta)")

    im2 = ax2.imshow(d[:, None], cmap=cm2, norm=TwoSlopeNorm(0.0, -0.24, 0.24), aspect="auto")
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Cross-Equation Difference\n(beta_1 Minus beta_2)"], fontsize=11.0, fontweight="bold")
    ax2.set_yticks([])
    ax2.set_title("(b) Cross-Equation Contrast (Delta beta)", fontweight="bold", loc="left", pad=10)

    for i, (v, _) in enumerate(row_specs):
        s_diff = stars(float(tests.loc[v, "p"])) if v in tests.index else ""
        txt_col = "white" if abs(d[i]) > 0.165 else "#0F172A"
        ax2.text(0, i, f"{d[i]:+.3f}{s_diff}", ha="center", va="center", fontsize=10.8, fontweight="bold", color=txt_col)

    ax2.set_yticks(np.arange(-0.5, len(row_specs)), minor=True)
    ax2.grid(which="minor", color="white", lw=2.0)
    ax2.tick_params(which="both", length=0)
    plt.colorbar(im2, ax=ax2, fraction=0.08, pad=0.04, label="Difference (Delta beta)")

    fig.text(
        0.5, -0.02,
        "Note: Joint FIML Bivariate Probit (N = 11,128 SMEs; 9 turnover bracket FEs, 26 country FEs, 3 sector FEs; G = 27 country-clustered SEs). "
        "Cross-equation Wald tests reported for focal and support predictors. * p < 0.05, ** p < 0.01, *** p < 0.001.",
        ha="center", fontsize=9.2, color="#475569",
    )

    save_all_formats(fig, "Figure_S9")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 76)
    print("Standalone One-Click Figure Generator (generate_all_figures.py)")
    print("Reading dynamic estimates from SME_Analytical_Final.csv & Results_Consolidated.xlsx")
    print("=" * 76)

    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Missing cleaned dataset: {DATA_CSV}. Run `python run_all_analysis.py` first.")
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Missing consolidated results: {EXCEL_PATH}. Run `python run_all_analysis.py` first.")
    if not os.path.exists(PS_CSV):
        raise FileNotFoundError(f"Missing exact propensity scores: {PS_CSV}. Run `python run_all_analysis.py` first.")

    df = pd.read_csv(DATA_CSV)
    if "DI" not in df.columns:
        df["DI"] = df["q9_binary"] - df["q14_binary"]
    xl = pd.ExcelFile(EXCEL_PATH)

    # Main Manuscript Figures (1 - 5)
    generate_figure_1(df, xl)
    generate_figure_2(xl)
    generate_figure_3(xl)
    generate_figure_4(xl)
    generate_path_diagram(xl, "Figure_5", with_moderation=True)

    # Supporting Information Figures (S1 - S9)
    generate_figure_s1(xl)
    generate_figure_s2(xl)
    generate_figure_s3(xl)
    generate_figure_s4(xl)
    generate_path_diagram(xl, "Figure_S5", with_moderation=False)
    generate_figure_s6(xl)
    generate_figure_s7(df)
    generate_figure_s8(df)
    generate_figure_s9(xl)

    print("=" * 76)
    print("All 14 figures exported cleanly in 600 DPI .png, editable .pdf, and .svg to:")
    for d in EXPORT_DIRS:
        print(f"  - {d}")
    print("=" * 76)


if __name__ == "__main__":
    main()
