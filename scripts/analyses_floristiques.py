#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Floristic diversity analysis — Yamoussoukro, Côte d'Ivoire
Reproducible analysis script for the GitHub repository.

Expected repository structure
-----------------------------
floristic-diversity-yamoussoukro/
├── data/
│   └── Base_floristique_Yamoussoukro.xlsx
├── scripts/
│   └── analyses_floristiques.py
└── results/
    ├── figures/
    └── tables/

The script is deliberately self-contained and uses only the supplied dataset.
It does not silently invent missing variables or columns.
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2_contingency
from scipy.special import gammaln, comb
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import MDS


# ---------------------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "Base_floristique_Yamoussoukro.xlsx"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 12345

# Keep the original site labels found in the supplied workbook.
SITE_ORDER = ["golf", "FUY", "KONGHO"]

REQUIRED_COLUMNS = [
    "Site", "Classe_occupation", "Placette", "Espece", "Auteur",
    "Genre", "Famille", "Circonference", "Diametre", "Aire_basale",
    "Hauteur", "Type_bio", "Type_choro", "UICN",
    "Statut_CITES_Oui_Non", "Annexe_CITES"
]


# ---------------------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------------------

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run this script from the repository or check the repository structure."
        )


def check_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )


def clean_text_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def safe_percent(numerator, denominator):
    return np.where(
        np.asarray(denominator) == 0,
        np.nan,
        100 * np.asarray(numerator) / np.asarray(denominator)
    )


# ---------------------------------------------------------------------
# 3. Load and quality-control data
# ---------------------------------------------------------------------

require_file(DATA_FILE)

raw = pd.read_excel(DATA_FILE, sheet_name="Donnees_floristique")
check_columns(raw, REQUIRED_COLUMNS)

df = raw.copy()

for col in ["Site", "Classe_occupation", "Placette", "Espece",
            "Genre", "Famille", "Type_bio", "Type_choro", "UICN"]:
    df[col] = clean_text_series(df[col])

# Quantitative analysis excludes observations outside quantitative plots.
plots = df[
    df["Placette"].notna()
    & df["Placette"].str.lower().ne("hors placette")
].copy()

# Floristic diversity requires an identified species.
div = plots.dropna(subset=["Espece"]).copy()

# Standardize numeric fields.
for col in ["Circonference", "Diametre", "Aire_basale", "Hauteur"]:
    div[col] = pd.to_numeric(div[col], errors="coerce")

# Preserve only the three study sites.
div = div[div["Site"].isin(SITE_ORDER)].copy()

if div.empty:
    raise ValueError("No observations remain after quality-control filtering.")

# ---------------------------------------------------------------------
# 4. Basic inventory summary
# ---------------------------------------------------------------------

summary_rows = []
for site in SITE_ORDER:
    g = div[div["Site"].eq(site)]
    summary_rows.append({
        "Site": site,
        "Individuals": len(g),
        "Species": g["Espece"].nunique(),
        "Genera": g["Genre"].nunique(),
        "Families": g["Famille"].nunique(),
        "Plots": g["Placette"].nunique()
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(TABLES_DIR / "inventory_summary.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------
# 5. Community matrices
# ---------------------------------------------------------------------

# Site × species abundance
abund_site = (
    div.groupby(["Site", "Espece"])
       .size()
       .unstack(fill_value=0)
       .reindex(SITE_ORDER, fill_value=0)
)

# Plot × species abundance
abund_plot = (
    div.groupby(["Placette", "Espece"])
       .size()
       .unstack(fill_value=0)
       .fillna(0)
)

plot_site = (
    div.groupby("Placette")["Site"]
       .first()
       .reindex(abund_plot.index)
)

# ---------------------------------------------------------------------
# 6. Hill numbers
# ---------------------------------------------------------------------

def hill_number(counts, q: int) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]

    if counts.size == 0:
        return np.nan

    p = counts / counts.sum()

    if q == 0:
        return float(len(p))
    if q == 1:
        return float(np.exp(-np.sum(p * np.log(p))))
    if q == 2:
        return float(1 / np.sum(p ** 2))

    raise ValueError("Only q=0, q=1 and q=2 are supported.")


hill_rows = []
for site in SITE_ORDER:
    counts = abund_site.loc[site].values
    hill_rows.append({
        "Site": site,
        "Hill_q0": hill_number(counts, 0),
        "Hill_q1": hill_number(counts, 1),
        "Hill_q2": hill_number(counts, 2)
    })

hill = pd.DataFrame(hill_rows)
hill.to_csv(TABLES_DIR / "hill_numbers.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------
# 7. Shannon, Simpson and Pielou
# ---------------------------------------------------------------------

diversity_rows = []
for site in SITE_ORDER:
    counts = abund_site.loc[site].values.astype(float)
    counts = counts[counts > 0]
    p = counts / counts.sum()

    H = -np.sum(p * np.log(p))
    S = len(p)
    Simpson = 1 - np.sum(p ** 2)
    Pielou = H / np.log(S) if S > 1 else np.nan

    diversity_rows.append({
        "Site": site,
        "Shannon_H": H,
        "Simpson_1_minus_D": Simpson,
        "Pielou_J": Pielou
    })

diversity = pd.DataFrame(diversity_rows)
diversity.to_csv(TABLES_DIR / "diversity_indices.csv",
                 index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------
# 8. Chao1 + analytical 95% CI
# ---------------------------------------------------------------------

def chao1_classic(counts):
    """
    Classic abundance-based Chao1 estimator:
        Sobs + f1^2 / (2*f2)

    where f1 and f2 are the numbers of singleton and doubleton species.
    """
    counts = np.asarray(counts, dtype=int)
    counts = counts[counts > 0]

    Sobs = len(counts)
    f1 = int(np.sum(counts == 1))
    f2 = int(np.sum(counts == 2))

    if f2 > 0:
        estimate = Sobs + (f1 ** 2) / (2 * f2)
    else:
        estimate = np.nan

    return float(estimate), Sobs, f1, f2


def chao1_variance_classic(f1: int, f2: int) -> float:
    """
    Analytical variance used to reproduce the Chao1 uncertainty
    reported in the manuscript's historical results.
    """
    if f2 <= 0:
        return np.nan

    ratio = f1 / f2
    return float(
        f2 * (
            0.5 * ratio**2
            + ratio**3
            + 0.25 * ratio**4
        )
    )


def chao1_result(counts):
    estimate, Sobs, f1, f2 = chao1_classic(counts)
    variance = chao1_variance_classic(f1, f2)
    se = np.sqrt(variance) if np.isfinite(variance) else np.nan

    if np.isfinite(se):
        lower = estimate - 1.96 * se
        upper = estimate + 1.96 * se
    else:
        lower = upper = np.nan

    return {
        "S_obs": Sobs,
        "f1": f1,
        "f2": f2,
        "Chao1": estimate,
        "Chao1_SE": se,
        "Chao1_CI95_lower": lower,
        "Chao1_CI95_upper": upper
    }


chao_rows = []

for site in SITE_ORDER:
    counts = div.loc[div["Site"].eq(site), "Espece"].value_counts().values
    row = {"Site": site}
    row.update(chao1_result(counts))

    # First-order jackknife using incidence across plots.
    g = div[div["Site"].eq(site)]
    incidence = g.groupby("Espece")["Placette"].nunique()
    Q1 = int((incidence == 1).sum())
    n_plots = g["Placette"].nunique()
    row["Jackknife1"] = (
        row["S_obs"] + Q1 * (n_plots - 1) / n_plots
        if n_plots > 0 else np.nan
    )
    chao_rows.append(row)

# Overall pooled estimate
overall_counts = div["Espece"].value_counts().values
overall = {"Site": "Overall"}
overall.update(chao1_result(overall_counts))
overall["Jackknife1"] = np.nan
chao_rows.append(overall)

chao = pd.DataFrame(chao_rows)
chao.to_csv(TABLES_DIR / "chao1_jackknife1.csv",
            index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------
# 9. Species richness, genera and families
# ---------------------------------------------------------------------

taxonomic_rows = []
for site in SITE_ORDER:
    g = div[div["Site"].eq(site)]
    taxonomic_rows.append({
        "Site": site,
        "Species": g["Espece"].nunique(),
        "Genera": g["Genre"].nunique(),
        "Families": g["Famille"].nunique()
    })

taxonomic = pd.DataFrame(taxonomic_rows)
taxonomic.to_csv(TABLES_DIR / "taxonomic_richness.csv",
                 index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------
# 10. Bray-Curtis similarity and hierarchical clustering
# ---------------------------------------------------------------------

if len(abund_plot) >= 2:
    bc = squareform(pdist(abund_plot.values, metric="braycurtis"))
    similarity = 1 - bc

    bc_table = pd.DataFrame(
        similarity,
        index=abund_plot.index,
        columns=abund_plot.index
    )
    bc_table.to_csv(TABLES_DIR / "bray_curtis_similarity_plot.csv",
                    encoding="utf-8-sig")

    # Cluster plot-level communities.
    Z = linkage(
        pdist(abund_plot.values, metric="braycurtis"),
        method="average"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(
        Z,
        labels=[f"{s}_{p}" for p, s in plot_site.items()],
        leaf_rotation=90,
        ax=ax
    )
    ax.set_ylabel("Bray–Curtis dissimilarity")
    ax.set_title("Hierarchical clustering of study plots")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "bray_curtis_clustering.png",
        dpi=650,
        bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# 11. NMDS
# ---------------------------------------------------------------------

if len(abund_plot) >= 3:
    nmds = MDS(
        n_components=2,
        metric=False,
        dissimilarity="precomputed",
        random_state=RANDOM_SEED,
        n_init=20,
        max_iter=1000,
        normalized_stress=True
    )

    coords = nmds.fit_transform(bc)
    stress = float(nmds.stress_)

    nmds_df = pd.DataFrame(
        coords,
        index=abund_plot.index,
        columns=["NMDS1", "NMDS2"]
    )
    nmds_df["Site"] = plot_site.values
    nmds_df.to_csv(TABLES_DIR / "nmds_coordinates.csv",
                   encoding="utf-8-sig")

    with open(TABLES_DIR / "nmds_stress.txt", "w", encoding="utf-8") as f:
        f.write(f"2D NMDS stress = {stress:.6f}\n")

    fig, ax = plt.subplots(figsize=(8, 7))

    for site in SITE_ORDER:
        mask = nmds_df["Site"].eq(site)
        ax.scatter(
            nmds_df.loc[mask, "NMDS1"],
            nmds_df.loc[mask, "NMDS2"],
            label=site,
            s=45
        )

    ax.set_xlabel("NMDS1")
    ax.set_ylabel("NMDS2")
    ax.set_title(f"NMDS based on Bray–Curtis dissimilarity (stress = {stress:.3f})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "nmds_2d.png",
        dpi=650,
        bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# 12. IndVal-like site association
# ---------------------------------------------------------------------

# This implementation is explicitly labelled as an IndVal-like statistic.
# It should not be presented as an exact replacement for labdsv/indicspecies
# without confirming the original statistical implementation.

def indval_like(abundance_matrix: pd.DataFrame, site_vector: pd.Series):
    rows = []

    for species in abundance_matrix.columns:
        values = abundance_matrix[species].values

        for site in SITE_ORDER:
            mask = site_vector.values == site

            if not np.any(mask):
                continue

            mean_target = values[mask].mean()
            mean_other = values[~mask].mean() if np.any(~mask) else 0

            specificity = (
                mean_target / (mean_target + mean_other)
                if (mean_target + mean_other) > 0 else 0
            )

            fidelity = np.mean(values[mask] > 0)

            statistic = 100 * specificity * fidelity

            rows.append({
                "Site": site,
                "Species": species,
                "IndVal_like": statistic
            })

    return pd.DataFrame(rows)


indval = indval_like(abund_plot, plot_site)
indval.to_csv(TABLES_DIR / "indval_like.csv",
              index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------
# 13. Diameter classes
# ---------------------------------------------------------------------

diam = div.dropna(subset=["Diametre"]).copy()

diam_bins = [3.18, 10, 20, 30, 40, 50, np.inf]
diam_labels = ["3.18–10", "10–20", "20–30", "30–40", "40–50", ">50"]

diam["Diameter_class"] = pd.cut(
    diam["Diametre"],
    bins=diam_bins,
    labels=diam_labels,
    include_lowest=True,
    right=True
)

diam_table = (
    pd.crosstab(
        diam["Site"],
        diam["Diameter_class"],
        normalize="index"
    )
    .reindex(SITE_ORDER)
    .fillna(0)
    * 100
)

diam_table.to_csv(TABLES_DIR / "diameter_class_distribution_percent.csv",
                  encoding="utf-8-sig")


# ---------------------------------------------------------------------
# 14. Stand structure by plot
# ---------------------------------------------------------------------

plot_structure = (
    div.groupby(["Site", "Placette"])
       .agg(
           n_stems=("Espece", "size"),
           basal_area=("Aire_basale", "sum"),
           mean_diameter=("Diametre", "mean"),
           mean_height=("Hauteur", "mean"),
           dominant_height=("Hauteur", "max")
       )
       .reset_index()
)

# Density based on 0.06 ha plots, as used in the study workflow.
PLOT_AREA_HA = 0.06
plot_structure["density_stems_ha"] = (
    plot_structure["n_stems"] / PLOT_AREA_HA
)
plot_structure["basal_area_ha"] = (
    plot_structure["basal_area"] / PLOT_AREA_HA
)

plot_structure.to_csv(
    TABLES_DIR / "plot_structure.csv",
    index=False,
    encoding="utf-8-sig"
)


# ---------------------------------------------------------------------
# 15. Top species by abundance
# ---------------------------------------------------------------------

top_species_rows = []

for site in SITE_ORDER:
    counts = (
        div.loc[div["Site"].eq(site), "Espece"]
           .value_counts()
           .head(10)
           .rename_axis("Species")
           .reset_index(name="Individuals")
    )
    counts.insert(0, "Site", site)
    counts.insert(0, "Rank", np.arange(1, len(counts) + 1))
    top_species_rows.append(counts)

top_species = pd.concat(top_species_rows, ignore_index=True)
top_species.to_csv(
    TABLES_DIR / "top10_species_by_site.csv",
    index=False,
    encoding="utf-8-sig"
)


# ---------------------------------------------------------------------
# 16. Conservation-status summaries
# ---------------------------------------------------------------------

uicn_table = (
    pd.crosstab(
        div["Site"],
        div["UICN"],
        normalize="index"
    )
    .reindex(SITE_ORDER)
    .fillna(0)
    * 100
)

uicn_table.to_csv(
    TABLES_DIR / "iucn_status_percent_by_site.csv",
    encoding="utf-8-sig"
)

threatened = (
    div["UICN"].isin(["VU", "EN"])
    .groupby(div["Site"])
    .agg(["sum", "count"])
)

threatened["percent_VU_EN"] = (
    100 * threatened["sum"] / threatened["count"]
)

threatened.to_csv(
    TABLES_DIR / "threatened_species_stems.csv",
    encoding="utf-8-sig"
)


# ---------------------------------------------------------------------
# 17. Final audit report
# ---------------------------------------------------------------------

audit = {
    "input_file": str(DATA_FILE),
    "raw_rows": int(len(raw)),
    "quantitative_rows_after_hors_placette_filter": int(len(plots)),
    "identified_rows_used_for_diversity": int(len(div)),
    "species_overall": int(div["Espece"].nunique()),
    "genera_overall": int(div["Genre"].nunique()),
    "families_overall": int(div["Famille"].nunique()),
    "plots_overall": int(div["Placette"].nunique()),
    "sites": SITE_ORDER,
    "outputs_tables": len(list(TABLES_DIR.glob("*"))),
    "outputs_figures": len(list(FIGURES_DIR.glob("*")))
}

pd.DataFrame([audit]).to_json(
    TABLES_DIR / "analysis_audit.json",
    orient="records",
    indent=2
)

print("=" * 72)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 72)
print(f"Dataset: {DATA_FILE}")
print(f"Raw records: {len(raw)}")
print(f"Records after Hors placette exclusion: {len(plots)}")
print(f"Records used for diversity analyses: {len(div)}")
print(f"Overall species richness: {div['Espece'].nunique()}")
print(f"Plots: {div['Placette'].nunique()}")
print(f"Tables written to: {TABLES_DIR}")
print(f"Figures written to: {FIGURES_DIR}")
print("=" * 72)
