#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Floristic diversity and ecological-value analysis — Yamoussoukro, Côte d'Ivoire
===============================================================================
Reproducible workflow for the repository:
    floristic-diversity-yamoussoukro

Input
-----
data/Base_floristique_Yamoussoukro.xlsx
sheet: Donnees_floristique

Outputs
-------
results/tables/
results/figures/

Analyses implemented
--------------------
H1 — Floristic composition and diversity
    * QC and inventory summary
    * taxonomic richness
    * per-plot Hill q0, q1, q2
    * Shannon, Simpson and Pielou
    * Kruskal-Wallis, pairwise Mann-Whitney + Bonferroni, epsilon-squared
    * individual-based rarefaction to n=386 (500 resamplings, percentile CI)
    * transparent sample-size interpolation/extrapolation curves
    * Chao1 + analytical 95% CI
    * first-order Jackknife
    * Bray-Curtis dissimilarity/similarity
    * hierarchical clustering
    * PERMANOVA, 999 permutations
    * PERMDISP using PCoA distances to centroid + Kruskal-Wallis permutation check
    * NMDS 2D and 3D
    * Dufrêne-Legendre IndVal with permutation p-values
    * SIMPER pairwise comparisons
    * IVI (importance value index)

H2 — Stand structure
    * density
    * basal area
    * mean DBH
    * quadratic mean DBH
    * maximum DBH
    * mean height
    * dominant height = mean height of six largest-DBH trees per plot
    * Kruskal-Wallis, pairwise Mann-Whitney + Bonferroni, epsilon-squared
    * diameter classes

H3 — Heritage value
    * IUCN, CITES, biological type and chorological spectra
    * chi-square tests of independence

H4 — Integrated ecological value
    * Composite Ecological Value Index (CEVI), using the published weights
      and transparent min-max scoring relative to the best-performing site.

Important methodological choices
---------------------------------
* The statistical unit is the plot (n=26), not the individual.
* Records labelled 'Hors placette' are excluded from quantitative analyses.
* One unidentified in-plot individual, if present, is retained for structure
  (density/basal area) but excluded from species-diversity analyses.
* The study assumes 20 x 30 m plots = 0.06 ha.
* No Benjamini-Hochberg correction is applied to IndVal, consistent with the
  manuscript's stated approach; the small KONGHO sample (6 plots) is reported
  as a limitation.
* The 3D NMDS is the final ordination because the manuscript reports a lower
  stress than the 2D solution.
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import (
    chi2_contingency,
    kruskal,
    mannwhitneyu,
    shapiro,
)
from scipy.special import gammaln
from sklearn.manifold import MDS

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "Base_floristique_Yamoussoukro.xlsx"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SITE_ORDER = ["golf", "FUY", "KONGHO"]
SITE_LABELS = {"golf": "Golf", "FUY": "FUY", "KONGHO": "KONGHO"}
RANDOM_SEED = 12345
N_PERMUTATIONS = 999
N_RAREFACTION = 500
COMMON_N = 386
PLOT_WIDTH_M = 20.0
PLOT_LENGTH_M = 30.0
PLOT_AREA_HA = PLOT_WIDTH_M * PLOT_LENGTH_M / 10000.0

DIAM_BINS = [3.18, 10, 20, 30, 40, 50, np.inf]
DIAM_LABELS = ["3.18–10", "10–20", "20–30", "30–40", "40–50", ">50"]

REQUIRED_COLUMNS = [
    "Site", "Classe_occupation", "Placette", "Espece", "Auteur",
    "Genre", "Famille", "Circonference", "Diametre", "Aire_basale",
    "Hauteur", "Type_bio", "Type_choro", "UICN",
    "Statut_CITES_Oui_Non", "Annexe_CITES"
]

# Published CEVI weights: total = 100.
CEVI_WEIGHTS = {
    "Hill_q0": 10.0,
    "Hill_q1": 15.0,
    "Hill_q2": 15.0,
    "PERMDISP": 10.0,
    "IndVal": 10.0,
    "Basal_area": 15.0,
    "Dominant_height": 10.0,
    "Threatened": 10.0,
    "Native_or_nonintroduced": 5.0,
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def savefig(fig, filename):
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=650, bbox_inches="tight")
    plt.close(fig)


def write_csv(df, filename, index=False):
    df.to_csv(TABLES_DIR / filename, index=index, encoding="utf-8-sig")


def write_json(obj, filename):
    with open(TABLES_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")


def check_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def clean_text(s):
    return s.astype("string").str.strip()


def epsilon_squared(H, k, n):
    """Epsilon-squared effect size for Kruskal-Wallis."""
    if n <= k or n <= 1:
        return np.nan
    return max(0.0, (H - k + 1.0) / (n - k))


def bonferroni_pairwise(values, groups):
    rows = []
    pairs = list(combinations(SITE_ORDER, 2))
    m = len(pairs)
    for a, b in pairs:
        x = values[np.asarray(groups) == a]
        y = values[np.asarray(groups) == b]
        if len(x) == 0 or len(y) == 0:
            p_raw = np.nan
        else:
            p_raw = mannwhitneyu(x, y, alternative="two-sided", method="auto").pvalue
        rows.append({
            "Group_A": a,
            "Group_B": b,
            "n_A": len(x),
            "n_B": len(y),
            "p_raw": p_raw,
            "p_bonferroni": min(1.0, p_raw * m) if np.isfinite(p_raw) else np.nan,
        })
    return pd.DataFrame(rows)


def kruskal_by_site(df, variable):
    work = df[["Site", variable]].dropna()
    arrays = [work.loc[work["Site"] == s, variable].to_numpy() for s in SITE_ORDER]
    H, p = kruskal(*arrays)
    k = len(SITE_ORDER)
    n = len(work)
    eps = epsilon_squared(H, k, n)
    result = pd.DataFrame([{
        "Variable": variable,
        "H": H,
        "df": k - 1,
        "p_value": p,
        "epsilon_squared": eps,
        "n": n,
    }])
    pair = bonferroni_pairwise(work[variable].to_numpy(), work["Site"].to_numpy())
    pair.insert(0, "Variable", variable)
    return result, pair


def hill(counts, q):
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
        return float(1.0 / np.sum(p ** 2))
    raise ValueError("q must be 0, 1 or 2")


def chao1_classic(counts):
    counts = np.asarray(counts, dtype=int)
    counts = counts[counts > 0]
    S = len(counts)
    f1 = int(np.sum(counts == 1))
    f2 = int(np.sum(counts == 2))
    if f2 > 0:
        estimate = S + f1 ** 2 / (2 * f2)
    else:
        estimate = np.nan
    return float(estimate), S, f1, f2


def chao1_variance(f1, f2):
    if f2 <= 0:
        return np.nan
    r = f1 / f2
    return float(f2 * (0.5 * r**2 + r**3 + 0.25 * r**4))


def chao1_result(counts):
    est, S, f1, f2 = chao1_classic(counts)
    var = chao1_variance(f1, f2)
    se = np.sqrt(var) if np.isfinite(var) else np.nan
    lo = est - 1.96 * se if np.isfinite(se) else np.nan
    hi = est + 1.96 * se if np.isfinite(se) else np.nan
    return {
        "S_obs": S, "f1": f1, "f2": f2, "Chao1": est,
        "Chao1_SE": se, "Chao1_CI95_lower": lo, "Chao1_CI95_upper": hi
    }


# =============================================================================
# 2. LOAD AND QUALITY CONTROL
# =============================================================================

require_file(DATA_FILE)
raw = pd.read_excel(DATA_FILE, sheet_name="Donnees_floristique")
check_columns(raw)

df = raw.copy()
for c in ["Site", "Classe_occupation", "Placette", "Espece", "Genre", "Famille", "Type_bio", "Type_choro", "UICN", "Statut_CITES_Oui_Non", "Annexe_CITES"]:
    df[c] = clean_text(df[c])
for c in ["Circonference", "Diametre", "Aire_basale", "Hauteur"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Exclude records outside quantitative plots and retain the three target sites.
plots = df[
    df["Placette"].notna()
    & df["Placette"].str.lower().ne("hors placette")
    & df["Site"].isin(SITE_ORDER)
].copy()

div = plots.dropna(subset=["Espece"]).copy()

# =============================================================================
# 3. INVENTORY SUMMARY AND COMMUNITY MATRICES
# =============================================================================

summary_rows = []
for site in SITE_ORDER:
    g = plots[plots.Site == site]
    gd = div[div.Site == site]
    summary_rows.append({
        "Site": site,
        "Plots": g.Placette.nunique(),
        "Quantitative_records": len(g),
        "Identified_records": len(gd),
        "Unidentified_records": int(g.Espece.isna().sum()),
        "Species": gd.Espece.nunique(),
        "Genera": gd.Genre.nunique(),
        "Families": gd.Famille.nunique(),
    })
summary = pd.DataFrame(summary_rows)
write_csv(summary, "inventory_summary.csv")

abund_site = (
    div.groupby(["Site", "Espece"]).size()
       .unstack(fill_value=0)
       .reindex(SITE_ORDER, fill_value=0)
)

abund_plot = (
    div.groupby(["Placette", "Espece"]).size()
       .unstack(fill_value=0)
       .fillna(0)
)
plot_site = plots.groupby("Placette")["Site"].first().reindex(abund_plot.index)
valid = plot_site.notna()
abund_plot = abund_plot.loc[valid]
plot_site = plot_site.loc[valid]

# =============================================================================
# 4. H1 — HILL, SHANNON, SIMPSON, PIELOU
# =============================================================================

site_hill = []
for site in SITE_ORDER:
    counts = abund_site.loc[site].values
    site_hill.append({"Site": site, "Hill_q0": hill(counts, 0), "Hill_q1": hill(counts, 1), "Hill_q2": hill(counts, 2)})
site_hill = pd.DataFrame(site_hill)
write_csv(site_hill, "hill_numbers.csv")

# Per-plot diversity is the correct statistical unit for site comparison.
plot_div_rows = []
for (site, plot), g in div.groupby(["Site", "Placette"]):
    counts = g.Espece.value_counts().values.astype(float)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p))
    D = np.sum(p ** 2)
    S = len(counts)
    plot_div_rows.append({
        "Site": site, "Placette": plot,
        "Hill_q0": hill(counts, 0),
        "Hill_q1": hill(counts, 1),
        "Hill_q2": hill(counts, 2),
        "Shannon_H": H,
        "Simpson_D": D,
        "Simpson_1_minus_D": 1 - D,
        "Pielou_J": H / np.log(S) if S > 1 else np.nan,
    })
plot_div = pd.DataFrame(plot_div_rows)
write_csv(plot_div, "plot_diversity_indices.csv")

# CEVI uses mean per-plot Hill diversity (not pooled site abundance).
cevi_hill = plot_div.groupby("Site")[["Hill_q0", "Hill_q1", "Hill_q2"]].mean().reindex(SITE_ORDER)

# Site-level indices.
diversity_rows = []
for site in SITE_ORDER:
    g = div[div.Site == site]
    counts = g.Espece.value_counts().values.astype(float)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p))
    D = np.sum(p ** 2)
    S = len(counts)
    diversity_rows.append({
        "Site": site,
        "Shannon_H": H,
        "Simpson_D": D,
        "Simpson_1_minus_D": 1 - D,
        "Pielou_J": H / np.log(S) if S > 1 else np.nan,
    })
diversity = pd.DataFrame(diversity_rows)
write_csv(diversity, "diversity_indices.csv")

# Non-parametric comparisons.
kw_all, pair_all = [], []
for variable in ["Hill_q0", "Hill_q1", "Hill_q2", "Shannon_H", "Simpson_1_minus_D", "Pielou_J"]:
    kw, pair = kruskal_by_site(plot_div, variable)
    kw_all.append(kw)
    pair_all.append(pair)
write_csv(pd.concat(kw_all, ignore_index=True), "diversity_kruskal_wallis.csv")
write_csv(pd.concat(pair_all, ignore_index=True), "diversity_pairwise_mannwhitney_bonferroni.csv")

# Shapiro-Wilk diagnostics per diversity metric and site.
shapiro_rows = []
for variable in ["Hill_q0", "Hill_q1", "Hill_q2", "Shannon_H", "Simpson_1_minus_D", "Pielou_J"]:
    for site in SITE_ORDER:
        x = plot_div.loc[plot_div.Site == site, variable].dropna().to_numpy()
        if len(x) >= 3:
            W, p = shapiro(x)
        else:
            W, p = np.nan, np.nan
        shapiro_rows.append({"Variable": variable, "Site": site, "W": W, "p_value": p, "n": len(x)})
write_csv(pd.DataFrame(shapiro_rows), "diversity_shapiro_wilk.csv")

# =============================================================================
# 5. H1 — CHAO1 AND JACKKNIFE1
# =============================================================================

chao_rows = []
for site in SITE_ORDER:
    counts = div.loc[div.Site == site, "Espece"].value_counts().values
    row = {"Site": site}
    row.update(chao1_result(counts))
    g = div[div.Site == site]
    incidence = g.groupby("Espece")["Placette"].nunique()
    Q1 = int((incidence == 1).sum())
    nplots = g.Placette.nunique()
    row["Jackknife1_Q1"] = Q1
    row["Jackknife1"] = row["S_obs"] + Q1 * (nplots - 1) / nplots
    chao_rows.append(row)

overall = {"Site": "Overall"}
overall.update(chao1_result(div.Espece.value_counts().values))
overall["Jackknife1_Q1"] = np.nan
overall["Jackknife1"] = np.nan
chao_rows.append(overall)
write_csv(pd.DataFrame(chao_rows), "chao1_jackknife1.csv")

# =============================================================================
# 6. H1 — INDIVIDUAL-BASED RAREFACTION (500 RESAMPLES) AND I/E CURVES
# =============================================================================

def random_rarefaction_counts(species_labels, m, n_reps, rng):
    species_labels = np.asarray(species_labels)
    if m > len(species_labels):
        return np.full(n_reps, np.nan)
    out = np.empty(n_reps)
    for i in range(n_reps):
        sample = rng.choice(species_labels, size=m, replace=False)
        out[i] = pd.unique(sample).size
    return out

rng = np.random.default_rng(RANDOM_SEED)
rare_rows = []
for site in SITE_ORDER:
    labels = div.loc[div.Site == site, "Espece"].dropna().to_numpy()
    N = len(labels)
    if COMMON_N > N:
        raise ValueError(f"COMMON_N={COMMON_N} exceeds sample size at {site} ({N}).")
    vals = random_rarefaction_counts(labels, COMMON_N, N_RAREFACTION, rng)
    rare_rows.append({
        "Site": site,
        "Reference_n": COMMON_N,
        "Observed_N": N,
        "Rarefied_species_mean": np.mean(vals),
        "Rarefied_species_CI95_lower": np.percentile(vals, 2.5),
        "Rarefied_species_CI95_upper": np.percentile(vals, 97.5),
        "Resamplings": N_RAREFACTION,
    })
rare_df = pd.DataFrame(rare_rows)
write_csv(rare_df, "rarefaction_at_386.csv")

# Exact expected rarefaction using the hypergeometric formula.
def expected_rarefied_richness(counts, m):
    counts = np.asarray(counts, dtype=int)
    N = counts.sum()
    if m > N:
        return np.nan
    log_den = gammaln(N + 1) - gammaln(m + 1) - gammaln(N - m + 1)
    total = 0.0
    for n_i in counts[counts > 0]:
        if N - n_i < m:
            prob_absent = 0.0
        else:
            log_num = gammaln(N - n_i + 1) - gammaln(m + 1) - gammaln(N - n_i - m + 1)
            prob_absent = np.exp(log_num - log_den)
        total += 1.0 - prob_absent
    return float(total)

# Transparent Chao1-based extrapolation; labelled as such rather than iNEXT.
def chao1_extrapolation(counts, m):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    N = counts.sum()
    Sobs = len(counts)
    f1 = np.sum(counts == 1)
    f2 = np.sum(counts == 2)
    if m <= N:
        return expected_rarefied_richness(counts, int(m))
    if f2 > 0:
        S_chao = Sobs + f1**2 / (2*f2)
    else:
        S_chao = Sobs + f1*(f1-1)/2
    m_extra = m - N
    if S_chao <= Sobs or f1 <= 0:
        return float(Sobs)
    unseen = S_chao - Sobs
    p = f1 / (N * unseen + f1)
    return float(Sobs + unseen * (1 - (1 - p) ** m_extra))

curve_rows = []
for site in SITE_ORDER:
    counts = div.loc[div.Site == site, "Espece"].value_counts().values
    N = int(counts.sum())
    max_m = 2 * N
    # 25 interpolation points up to observed N + 25 extrapolation points.
    mvals = np.unique(np.round(np.linspace(1, max_m, 80)).astype(int))
    for m in mvals:
        value = chao1_extrapolation(counts, int(m))
        curve_rows.append({
            "Site": site,
            "n_individuals": int(m),
            "Observed_N": N,
            "Curve": "Interpolation" if m <= N else "Extrapolation",
            "Expected_species": value,
        })
curve_df = pd.DataFrame(curve_rows)
write_csv(curve_df, "rarefaction_extrapolation_curves.csv")

fig, ax = plt.subplots(figsize=(8, 6))
for site in SITE_ORDER:
    g = curve_df[curve_df.Site == site]
    ax.plot(g.n_individuals, g.Expected_species, label=SITE_LABELS[site], linewidth=2)
    ax.axvline(g.Observed_N.iloc[0], linestyle="--", linewidth=0.7, alpha=0.4)
ax.axvline(COMMON_N, linestyle=":", linewidth=1, label=f"Common reference n={COMMON_N}")
ax.set_xlabel("Number of individuals")
ax.set_ylabel("Expected species richness")
ax.set_title("Individual-based rarefaction and extrapolation")
ax.legend(frameon=False)
fig.tight_layout()
savefig(fig, "rarefaction_extrapolation.png")

# =============================================================================
# 7. H1 — TAXONOMIC RICHNESS
# =============================================================================

taxonomic = []
for site in SITE_ORDER:
    g = div[div.Site == site]
    taxonomic.append({"Site": site, "Species": g.Espece.nunique(), "Genera": g.Genre.nunique(), "Families": g.Famille.nunique()})
write_csv(pd.DataFrame(taxonomic), "taxonomic_richness.csv")

# =============================================================================
# 8. H1 — BRAY-CURTIS, SIMILARITY, CLUSTERING
# =============================================================================

bc_condensed = pdist(abund_plot.values, metric="braycurtis")
bc = squareform(bc_condensed)
write_csv(pd.DataFrame(1 - bc, index=abund_plot.index, columns=abund_plot.index), "bray_curtis_similarity_plot.csv", index=True)

site_abund = abund_site.loc[SITE_ORDER]
site_bc = squareform(pdist(site_abund.values, metric="braycurtis"))
write_csv(pd.DataFrame(site_bc, index=SITE_ORDER, columns=SITE_ORDER), "bray_curtis_dissimilarity_site.csv", index=True)
write_csv(pd.DataFrame(1 - site_bc, index=SITE_ORDER, columns=SITE_ORDER), "bray_curtis_similarity_site.csv", index=True)

Z = linkage(bc_condensed, method="average")
fig, ax = plt.subplots(figsize=(10, 7))
dendrogram(Z, labels=[f"{s}_{p}" for p, s in plot_site.items()], leaf_rotation=90, ax=ax)
ax.set_ylabel("Bray–Curtis dissimilarity")
ax.set_title("Hierarchical clustering of study plots")
fig.tight_layout()
savefig(fig, "bray_curtis_clustering.png")

# Jaccard presence/absence site-level similarity, as reported in manuscript.
pa_site = (site_abund > 0).astype(int)
jaccard = squareform(pdist(pa_site.values, metric="jaccard"))
write_csv(pd.DataFrame(1 - jaccard, index=SITE_ORDER, columns=SITE_ORDER), "jaccard_similarity_site.csv", index=True)

# =============================================================================
# 9. H1 — PERMANOVA
# =============================================================================

def permanova(D, groups, n_perm=999, seed=12345):
    D = np.asarray(D, dtype=float)
    groups = np.asarray(groups)
    n = len(groups)
    levels = pd.unique(groups)
    k = len(levels)
    if k < 2 or n <= k:
        raise ValueError("Invalid PERMANOVA design")

    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J @ (D ** 2) @ J
    SST = np.trace(G)

    def ss_between(labels):
        ss = 0.0
        for level in levels:
            idx = np.where(labels == level)[0]
            if len(idx):
                ss += G[np.ix_(idx, idx)].sum() / len(idx)
        return float(ss)

    SSB = ss_between(groups)
    SSW = SST - SSB
    df1 = k - 1
    df2 = n - k
    F = (SSB / df1) / (SSW / df2)
    R2 = SSB / SST

    rng = np.random.default_rng(seed)
    Fperm = np.empty(n_perm)
    for i in range(n_perm):
        pg = rng.permutation(groups)
        pSSB = ss_between(pg)
        pSSW = SST - pSSB
        Fperm[i] = (pSSB / df1) / (pSSW / df2)
    p = (1 + np.sum(Fperm >= F)) / (1 + n_perm)
    return {"Pseudo_F": F, "R2": R2, "SS_between": SSB, "SS_within": SSW, "SS_total": SST, "df_between": df1, "df_within": df2, "p_value": p, "n_permutations": n_perm}

perma = permanova(bc, plot_site.values, N_PERMUTATIONS, 42)
write_csv(pd.DataFrame([perma]), "permanova.csv")

# =============================================================================
# 10. H1 — PERMDISP (PCoA distance-to-centroid + KW)
# =============================================================================

def pcoa(D):
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    keep = eigvals > 1e-12
    eigvals, eigvecs = eigvals[keep], eigvecs[:, keep]
    return eigvecs * np.sqrt(eigvals)

coords_pcoa = pcoa(bc)
centroid_dist = np.zeros(len(plot_site))
for site in SITE_ORDER:
    idx = np.where(plot_site.values == site)[0]
    centroid = coords_pcoa[idx].mean(axis=0)
    centroid_dist[idx] = np.sqrt(((coords_pcoa[idx] - centroid) ** 2).sum(axis=1))

permdisp_df = pd.DataFrame({"Placette": abund_plot.index, "Site": plot_site.values, "Distance_to_centroid": centroid_dist})
write_csv(permdisp_df, "permdisp_distances.csv")

arrays = [permdisp_df.loc[permdisp_df.Site == s, "Distance_to_centroid"].values for s in SITE_ORDER]
H_disp, p_disp = kruskal(*arrays)
permdisp_summary = pd.DataFrame([{"H": H_disp, "df": 2, "p_value": p_disp, "method": "Kruskal-Wallis on PCoA distance-to-centroid"}])
write_csv(permdisp_summary, "permdisp.csv")

site_disp = permdisp_df.groupby("Site").Distance_to_centroid.agg(["mean", "std", "count"]).reindex(SITE_ORDER).reset_index()
write_csv(site_disp, "permdisp_site_summary.csv")

# =============================================================================
# 11. H1 — NMDS 2D and 3D
# =============================================================================

nmds_results = []
for dimensions in [2, 3]:
    best_model = None
    best_stress = np.inf
    best_seed = None

    # The original workflow searched 20 random starts and retained the
    # lowest-stress solution. This is more reproducible than relying on a
    # single arbitrary initialisation.
    for seed in range(20):
        model = MDS(
            n_components=dimensions,
            metric=False,
            dissimilarity="precomputed",
            random_state=seed,
            n_init=1,
            max_iter=3000,
            eps=1e-9,
            normalized_stress=True,
        )
        coords = model.fit_transform(bc)
        if model.stress_ < best_stress:
            best_stress = float(model.stress_)
            best_model = model
            best_seed = seed

    coords = best_model.embedding_
    stress = best_stress
    cols = [f"NMDS{i+1}" for i in range(dimensions)]
    out = pd.DataFrame(coords, index=abund_plot.index, columns=cols)
    out["Site"] = plot_site.values
    write_csv(out, f"nmds_{dimensions}d_coordinates.csv", index=True)
    with open(TABLES_DIR / f"nmds_{dimensions}d_stress.txt", "w", encoding="utf-8") as f:
        f.write(f"NMDS {dimensions}D normalized stress = {stress:.12f}\n")
        f.write(f"Best random seed among 0–19 = {best_seed}\n")
    nmds_results.append({"Dimensions": dimensions, "Stress": stress, "Best_seed": best_seed})

    if dimensions == 2:
        fig, ax = plt.subplots(figsize=(8, 7))
        for site in SITE_ORDER:
            m = out.Site == site
            ax.scatter(out.loc[m, "NMDS1"], out.loc[m, "NMDS2"], label=SITE_LABELS[site], s=45)
        ax.set_xlabel("NMDS1")
        ax.set_ylabel("NMDS2")
        ax.set_title(f"NMDS 2D (stress = {stress:.3f})")
        ax.legend(frameon=False)
        fig.tight_layout()
        savefig(fig, "nmds_2d.png")

    if dimensions == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        for site in SITE_ORDER:
            m = out.Site == site
            ax.scatter(out.loc[m, "NMDS1"], out.loc[m, "NMDS2"], out.loc[m, "NMDS3"], label=SITE_LABELS[site], s=35)
        ax.set_xlabel("NMDS1")
        ax.set_ylabel("NMDS2")
        ax.set_zlabel("NMDS3")
        ax.set_title(f"NMDS 3D (stress = {stress:.3f})")
        ax.legend(frameon=False)
        fig.tight_layout()
        savefig(fig, "nmds_3d.png")

write_csv(pd.DataFrame(nmds_results), "nmds_stress_summary.csv")

# =============================================================================
# 12. H1 — Dufrêne–Legendre IndVal
# =============================================================================

def indval_dl(X, groups, n_perm=999, seed=12345):
    """
    Dufrêne–Legendre IndVal with site-specific permutation tests.

    For each species and target site:
        A = mean abundance in target site / sum of site mean abundances
        B = frequency of occurrence in target site
        IndVal = 100 * sqrt(A * B)

    P-values are calculated for the observed target-site statistic against
    the same target-site statistic after unrestricted permutation of plot
    labels. This reproduces the logic used in the original analysis notebook
    and the supplementary IndVal table.
    """
    X = X.to_numpy(dtype=float)
    species = abund_plot.columns.tolist()
    groups = np.asarray(groups)
    levels = pd.unique(groups)
    rng = np.random.default_rng(seed)

    def calc_for_group(labels, target):
        out = []
        target_mask = labels == target
        means = {g: X[labels == g].mean(axis=0) for g in levels}
        denom = np.sum(np.vstack([means[g] for g in levels]), axis=0)
        A = np.divide(
            means[target], denom,
            out=np.zeros_like(means[target]),
            where=denom > 0
        )
        B = np.mean(X[target_mask] > 0, axis=0)
        IV = 100 * np.sqrt(A * B)
        for j, sp in enumerate(species):
            out.append((sp, A[j], B[j], IV[j]))
        return pd.DataFrame(out, columns=["Species", "Specificity_A", "Fidelity_B", "IndVal"])

    observed_by_group = {
        g: calc_for_group(groups, g).set_index("Species")
        for g in levels
    }

    exceed = {
        g: np.zeros(len(species), dtype=int)
        for g in levels
    }

    for _ in range(n_perm):
        perm_groups = rng.permutation(groups)
        for g in levels:
            perm_iv = calc_for_group(perm_groups, g).set_index("Species")["IndVal"].reindex(species).to_numpy()
            obs_iv = observed_by_group[g].reindex(species)["IndVal"].to_numpy()
            exceed[g] += perm_iv >= (obs_iv - 1e-12)

    rows = []
    for g in levels:
        obs = observed_by_group[g].copy().reset_index()
        obs["Indicator_site"] = g
        obs["p_value"] = (exceed[g] + 1) / (n_perm + 1)
        obs["n_permutations"] = n_perm
        rows.append(obs)

    out = pd.concat(rows, ignore_index=True)
    # A species can be significant for more than one group in theory; retain
    # only the group with the highest observed IndVal for the final indicator
    # table, consistent with the manuscript's one-site-per-species reporting.
    out = out.sort_values(["Species", "IndVal"], ascending=[True, False])
    out = out.drop_duplicates("Species", keep="first")
    return out.sort_values(["p_value", "IndVal"], ascending=[True, False])

indval = indval_dl(abund_plot, plot_site.values, N_PERMUTATIONS, 5)
indval["Significant_alpha_0.05"] = indval.p_value < 0.05
write_csv(indval, "indval_dufrene_legendre.csv")

# Significant indicator species by site for CEVI.
sig_counts = indval[indval["Significant_alpha_0.05"]].groupby("Indicator_site").size().reindex(SITE_ORDER, fill_value=0)
write_csv(sig_counts.rename("Significant_IndVal_species").reset_index(), "indval_significant_counts.csv")

# =============================================================================
# 13. H1 — SIMPER
# =============================================================================

def simper_pair(X, groups, a, b):
    Xa = X.loc[np.asarray(groups) == a]
    Xb = X.loc[np.asarray(groups) == b]
    species = X.columns
    contrib = {sp: [] for sp in species}
    for _, ra in Xa.iterrows():
        for _, rb in Xb.iterrows():
            denom = (ra.values + rb.values).sum()
            if denom == 0:
                continue
            c = np.abs(ra.values - rb.values) / denom
            for j, sp in enumerate(species):
                contrib[sp].append(c[j])
    rows = []
    for sp in species:
        mean_c = np.mean(contrib[sp]) if contrib[sp] else 0.0
        rows.append({
            "Comparison": f"{a} vs {b}",
            "Species": sp,
            "Mean_contribution": mean_c,
            f"Mean_abundance_{a}": Xa[sp].mean(),
            f"Mean_abundance_{b}": Xb[sp].mean(),
        })
    out = pd.DataFrame(rows)
    total = out.Mean_contribution.sum()
    out["Contribution_percent"] = 100*out.Mean_contribution/total if total > 0 else np.nan
    out = out.sort_values("Contribution_percent", ascending=False).reset_index(drop=True)
    out["Cumulative_percent"] = out.Contribution_percent.cumsum()
    return out

simper_tables = [simper_pair(abund_plot, plot_site.values, a, b) for a, b in combinations(SITE_ORDER, 2)]
simper_all = pd.concat(simper_tables, ignore_index=True)
write_csv(simper_all, "simper_all_pairwise.csv")
for comparison, g in simper_all.groupby("Comparison"):
    write_csv(g, "simper_" + comparison.replace(" ", "_") + ".csv")

# =============================================================================
# 14. H1 — IVI (Importance Value Index)
# =============================================================================

# For each site and species:
# relative density + relative frequency + relative dominance.
# Frequency = number of plots containing the species / total plots.
ivi_rows = []
for site in SITE_ORDER:
    g = plots[plots.Site == site]
    site_plots = g.Placette.nunique()
    # Use identified species for species-level IVI; structure measurements are used for dominance.
    gs = g.dropna(subset=["Espece"]).copy()
    stems_by_sp = gs.groupby("Espece").size()
    basal_by_sp = gs.groupby("Espece")["Aire_basale"].sum(min_count=1).fillna(0)
    freq_by_sp = gs.groupby("Espece")["Placette"].nunique()
    total_stems = stems_by_sp.sum()
    total_ba = basal_by_sp.sum()
    for sp in stems_by_sp.index:
        RD = 100 * stems_by_sp[sp] / total_stems if total_stems else np.nan
        RF = 100 * freq_by_sp[sp] / freq_by_sp.sum() if freq_by_sp.sum() else np.nan
        RDo = 100 * basal_by_sp[sp] / total_ba if total_ba else np.nan
        ivi_rows.append({"Site": site, "Species": sp, "Relative_density": RD, "Relative_frequency": RF, "Relative_dominance": RDo, "IVI": RD + RF + RDo})
ivi = pd.DataFrame(ivi_rows).sort_values(["Site", "IVI"], ascending=[True, False])
write_csv(ivi, "ivi_species_by_site.csv")

# =============================================================================
# 15. H2 — STAND STRUCTURE
# =============================================================================

# Structural dataset: retain all in-plot records, including unidentified stems.
# Dominant height = mean height of six largest-DBH trees per plot.
struct_rows = []
for (site, plot), g in plots.groupby(["Site", "Placette"]):
    dbh = g.Diametre.dropna()
    h = g.Hauteur.dropna()
    ba = g.Aire_basale.dropna()
    top6 = g.dropna(subset=["Diametre", "Hauteur"]).nlargest(6, "Diametre")["Hauteur"]
    struct_rows.append({
        "Site": site,
        "Placette": plot,
        "n_stems": len(g),
        "density_stems_ha": len(g) / PLOT_AREA_HA,
        "basal_area_m2_ha": g.Aire_basale.sum(skipna=True) / PLOT_AREA_HA,
        "mean_DBA_cm": dbh.mean() if len(dbh) else np.nan,
        "quadratic_mean_DBA_cm": np.sqrt(np.mean(dbh**2)) if len(dbh) else np.nan,
        "max_DBA_cm": dbh.max() if len(dbh) else np.nan,
        "mean_height_m": h.mean() if len(h) else np.nan,
        "dominant_height_top6_m": top6.mean() if len(top6) else np.nan,
    })
structure_plot = pd.DataFrame(struct_rows)
write_csv(structure_plot, "plot_structure.csv")

structure_site = structure_plot.groupby("Site").agg(
    plots=("Placette", "count"),
    density_mean=("density_stems_ha", "mean"), density_sd=("density_stems_ha", "std"),
    basal_area_mean=("basal_area_m2_ha", "mean"), basal_area_sd=("basal_area_m2_ha", "std"),
    mean_DBA_mean=("mean_DBA_cm", "mean"), mean_DBA_sd=("mean_DBA_cm", "std"),
    quadratic_mean_DBA_mean=("quadratic_mean_DBA_cm", "mean"), quadratic_mean_DBA_sd=("quadratic_mean_DBA_cm", "std"),
    max_DBA_mean=("max_DBA_cm", "mean"), max_DBA_sd=("max_DBA_cm", "std"),
    mean_height_mean=("mean_height_m", "mean"), mean_height_sd=("mean_height_m", "std"),
    dominant_height_mean=("dominant_height_top6_m", "mean"), dominant_height_sd=("dominant_height_top6_m", "std"),
).reindex(SITE_ORDER).reset_index()
write_csv(structure_site, "structure_site_summary.csv")

structure_kw, structure_pair = [], []
for variable in ["density_stems_ha", "basal_area_m2_ha", "mean_DBA_cm", "quadratic_mean_DBA_cm", "max_DBA_cm", "mean_height_m", "dominant_height_top6_m"]:
    kw, pair = kruskal_by_site(structure_plot, variable)
    structure_kw.append(kw)
    structure_pair.append(pair)
write_csv(pd.concat(structure_kw, ignore_index=True), "structure_kruskal_wallis.csv")
write_csv(pd.concat(structure_pair, ignore_index=True), "structure_pairwise_mannwhitney_bonferroni.csv")

# Diameter-class distribution.
diam = plots.dropna(subset=["Diametre"]).copy()
diam["Diameter_class"] = pd.cut(diam.Diametre, bins=DIAM_BINS, labels=DIAM_LABELS, include_lowest=True, right=True)
diam_pct = pd.crosstab(diam.Site, diam.Diameter_class, normalize="index").reindex(SITE_ORDER).fillna(0)*100
write_csv(diam_pct, "diameter_class_distribution_percent.csv")

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(DIAM_LABELS))
width = 0.25
for i, site in enumerate(SITE_ORDER):
    ax.bar(x + (i-1)*width, diam_pct.loc[site].values, width, label=SITE_LABELS[site])
ax.set_xticks(x)
ax.set_xticklabels(DIAM_LABELS)
ax.set_xlabel("Diameter class (cm)")
ax.set_ylabel("Stems (%)")
ax.set_title("Diameter-class distribution by site")
ax.legend(frameon=False)
fig.tight_layout()
savefig(fig, "diameter_class_distribution.png")

# =============================================================================
# 16. H3 — HERITAGE SPECTRA AND CHI-SQUARE TESTS
# =============================================================================

def chi_square_category(df, category):
    tab = pd.crosstab(df.Site, df[category]).reindex(SITE_ORDER, fill_value=0)
    if tab.shape[1] < 2:
        return tab, {"Category": category, "chi2": np.nan, "df": np.nan, "p_value": np.nan, "min_expected": np.nan}
    chi2, p, dfree, expected = chi2_contingency(tab.values)
    return tab, {"Category": category, "chi2": chi2, "df": dfree, "p_value": p, "min_expected": expected.min()}

heritage_results = []
for category in ["UICN", "Statut_CITES_Oui_Non", "Type_bio", "Type_choro"]:
    tab, res = chi_square_category(div.dropna(subset=[category]), category)
    tab_pct = tab.div(tab.sum(axis=1), axis=0)*100
    write_csv(tab, f"heritage_{category}_counts.csv", index=True)
    write_csv(tab_pct, f"heritage_{category}_percent.csv", index=True)
    heritage_results.append(res)
write_csv(pd.DataFrame(heritage_results), "heritage_chi_square_tests.csv")

# Threatened stems = VU + EN, as specified by the CEVI table.
threatened = div.UICN.isin(["VU", "EN"])
threatened_site = div.assign(Threatened=threatened).groupby("Site").Threatened.mean().reindex(SITE_ORDER)*100

# Native/non-introduced/non-cultivated: use chorological labels where the dataset
# explicitly identifies a non-introduced/native category; otherwise preserve the
# observed categorical data and document the mapping in the output.
chor = div[["Site", "Espece", "Type_choro"]].drop_duplicates()
write_csv(chor, "heritage_chorology_species.csv")

# The published CEVI uses a binary 'non-introduced/non-cultivated' proportion.
# If a direct binary category is unavailable, infer native status from the chorological
# field only when labels clearly indicate introduced/cultivated taxa.
introduced_tokens = {"INTROD", "INTRODUITE", "INTRODUIT", "EXOTIQUE", "CULTIVEE", "CULTIVE", "CULTIVATED", "INTRODUCED"}
def native_flag(x):
    if pd.isna(x):
        return np.nan
    u = str(x).strip().upper()
    return False if u in introduced_tokens else True

native_by_stem = div.Type_choro.map(native_flag)
native_pct = div.assign(Native_or_nonintroduced=native_by_stem).groupby("Site").Native_or_nonintroduced.mean().reindex(SITE_ORDER)*100

# CITES positive records.
cites_counts = pd.crosstab(div.Site, div.Statut_CITES_Oui_Non).reindex(SITE_ORDER, fill_value=0)
write_csv(cites_counts, "cites_status_counts.csv", index=True)

heritage_summary = pd.DataFrame({
    "Site": SITE_ORDER,
    "Threatened_VU_EN_percent": threatened_site.values,
    "Native_or_nonintroduced_percent": native_pct.values,
}).set_index("Site")
write_csv(heritage_summary.reset_index(), "heritage_summary.csv")

# =============================================================================
# 17. H4 — CEVI
# =============================================================================

def score_relative_to_best(series, higher_is_better=True):
    s = pd.Series(series, index=series.index, dtype=float)
    if higher_is_better:
        best = s.max()
        return 100*s/best if best > 0 else pd.Series(0.0, index=s.index)
    worst = s.min()
    return 100*worst/s if np.all(s > 0) and worst > 0 else pd.Series(np.nan, index=s.index)

# Published CEVI component scores use the best-performing site as 100.
cevi = cevi_hill.copy()
cevi["Hill_q0_score"] = score_relative_to_best(cevi.Hill_q0)
cevi["Hill_q1_score"] = score_relative_to_best(cevi.Hill_q1)
cevi["Hill_q2_score"] = score_relative_to_best(cevi.Hill_q2)

# PERMDISP compositional homogeneity: lower distance-to-centroid = more homogeneous.
# Therefore the best (lowest) site receives 100.
mean_disp = permdisp_df.groupby("Site").Distance_to_centroid.mean().reindex(SITE_ORDER)
cevi["PERMDISP_score"] = 100 * mean_disp.min() / mean_disp

# IndVal component: maximum significant IndVal within each site.
indval_sig = indval[indval["Significant_alpha_0.05"]].copy()
sig_n = indval_sig.groupby("Indicator_site").size().reindex(SITE_ORDER, fill_value=0)
cevi["Significant_IndVal_species"] = sig_n.values
cevi["IndVal_score"] = 100 * sig_n / sig_n.max() if sig_n.max() > 0 else 0

# Structure: site mean basal area and dominant height.
ba = structure_site.set_index("Site").basal_area_mean.reindex(SITE_ORDER)
dh = structure_site.set_index("Site").dominant_height_mean.reindex(SITE_ORDER)
cevi["Basal_area_score"] = 100*ba/ba.max()
cevi["Dominant_height_score"] = 100*dh/dh.max()

# Heritage components.
cevi["Threatened_score"] = 100*threatened_site/threatened_site.max() if threatened_site.max() > 0 else 0
cevi["Native_or_nonintroduced_score"] = 100*native_pct/native_pct.max() if native_pct.max() > 0 else 0

cevi["Floristic_value_score"] = (
    0.10*cevi.Hill_q0_score + 0.15*cevi.Hill_q1_score + 0.15*cevi.Hill_q2_score
)
cevi["Floristic_composition_score"] = (
    0.10*cevi.PERMDISP_score + 0.10*cevi.IndVal_score
)
cevi["Vegetation_structure_score"] = (
    0.15*cevi.Basal_area_score + 0.10*cevi.Dominant_height_score
)
cevi["Conservation_value_score"] = (
    0.10*cevi.Threatened_score + 0.05*cevi.Native_or_nonintroduced_score
)
cevi["CEVI"] = (
    cevi.Floristic_value_score
    + cevi.Floristic_composition_score
    + cevi.Vegetation_structure_score
    + cevi.Conservation_value_score
)

# Expected published maximum-normalised scores should sum to ~100 at the best site.
def cevi_class(x):
    if x >= 75:
        return "Very high"
    if x >= 50:
        return "Moderate"
    return "Low"

cevi["Ecological_value_class"] = cevi.CEVI.map(cevi_class)
write_csv(cevi.reset_index(), "cevi.csv")

# Long-format CEVI table for transparent reporting.
cevi_long = pd.DataFrame([
    {"Dimension": "Floristic value", "Indicator": "Hill q0", "Weight_percent": 10, **{s: cevi.loc[s, "Hill_q0_score"] for s in SITE_ORDER}},
    {"Dimension": "Floristic value", "Indicator": "Hill q1", "Weight_percent": 15, **{s: cevi.loc[s, "Hill_q1_score"] for s in SITE_ORDER}},
    {"Dimension": "Floristic value", "Indicator": "Hill q2", "Weight_percent": 15, **{s: cevi.loc[s, "Hill_q2_score"] for s in SITE_ORDER}},
    {"Dimension": "Floristic value", "Indicator": "Floristic value score", "Weight_percent": 40, **{s: cevi.loc[s, "Floristic_value_score"] for s in SITE_ORDER}},
    {"Dimension": "Floristic composition", "Indicator": "PERMDISP homogeneity", "Weight_percent": 10, **{s: cevi.loc[s, "PERMDISP_score"] for s in SITE_ORDER}},
    {"Dimension": "Floristic composition", "Indicator": "Significant IndVal", "Weight_percent": 10, **{s: cevi.loc[s, "IndVal_score"] for s in SITE_ORDER}},
    {"Dimension": "Floristic composition", "Indicator": "Floristic composition score", "Weight_percent": 20, **{s: cevi.loc[s, "Floristic_composition_score"] for s in SITE_ORDER}},
    {"Dimension": "Vegetation structure", "Indicator": "Basal area", "Weight_percent": 15, **{s: cevi.loc[s, "Basal_area_score"] for s in SITE_ORDER}},
    {"Dimension": "Vegetation structure", "Indicator": "Dominant height", "Weight_percent": 10, **{s: cevi.loc[s, "Dominant_height_score"] for s in SITE_ORDER}},
    {"Dimension": "Vegetation structure", "Indicator": "Vegetation structure score", "Weight_percent": 25, **{s: cevi.loc[s, "Vegetation_structure_score"] for s in SITE_ORDER}},
    {"Dimension": "Conservation value", "Indicator": "Threatened stems (VU+EN)", "Weight_percent": 10, **{s: cevi.loc[s, "Threatened_score"] for s in SITE_ORDER}},
    {"Dimension": "Conservation value", "Indicator": "Non-introduced/non-cultivated", "Weight_percent": 5, **{s: cevi.loc[s, "Native_or_nonintroduced_score"] for s in SITE_ORDER}},
    {"Dimension": "Conservation value", "Indicator": "Conservation value score", "Weight_percent": 15, **{s: cevi.loc[s, "Conservation_value_score"] for s in SITE_ORDER}},
    {"Dimension": "Overall", "Indicator": "CEVI", "Weight_percent": 100, **{s: cevi.loc[s, "CEVI"] for s in SITE_ORDER}},
])
write_csv(cevi_long, "cevi_components.csv")

# Published CEVI values are retained as a validation reference only. The original
# analysis notebook does not contain the CEVI calculation, and the manuscript does
# not provide a machine-readable species-level mapping for the
# "non-introduced/non-cultivated" component. Therefore the script computes that
# component transparently from Type_choro (Introd = introduced/cultivated) and
# reports any discrepancy with the published CEVI values instead of hard-coding
# the published results.
CEVI_PUBLISHED = pd.DataFrame({
    "Site": SITE_ORDER,
    "Published_CEVI": [57.84, 59.52, 100.00],
    "Published_native_component_score": [67.26, 74.51, 100.00],
    "Published_IndVal_component_score": [42.11, 52.63, 100.00],
})

cevi_check = CEVI_PUBLISHED.merge(
    cevi.reset_index()[["Site", "CEVI", "Native_or_nonintroduced_score", "IndVal_score"]],
    on="Site", how="left"
)
cevi_check["CEVI_difference"] = cevi_check["CEVI"] - cevi_check["Published_CEVI"]
cevi_check["Native_score_difference"] = cevi_check["Native_or_nonintroduced_score"] - cevi_check["Published_native_component_score"]
cevi_check["IndVal_score_difference"] = cevi_check["IndVal_score"] - cevi_check["Published_IndVal_component_score"]
write_csv(cevi_check, "cevi_reproduction_check.csv")

# =============================================================================
# 18. FINAL AUDIT
# =============================================================================

nmds_stress = {int(r.Dimensions): float(r.Stress) for r in pd.DataFrame(nmds_results).itertuples()}
audit = {
    "input_file": str(DATA_FILE),
    "raw_rows": int(len(raw)),
    "quantitative_rows": int(len(plots)),
    "identified_rows_for_diversity": int(len(div)),
    "unidentified_quantitative_rows": int(plots.Espece.isna().sum()),
    "species_overall": int(div.Espece.nunique()),
    "genera_overall": int(div.Genre.nunique()),
    "families_overall": int(div.Famille.nunique()),
    "plots_overall": int(div.Placette.nunique()),
    "site_plot_counts": {s: int(div.loc[div.Site == s, "Placette"].nunique()) for s in SITE_ORDER},
    "site_individual_counts": {s: int(len(div[div.Site == s])) for s in SITE_ORDER},
    "plot_area_ha": PLOT_AREA_HA,
    "common_rarefaction_n": COMMON_N,
    "rarefaction_resamplings": N_RAREFACTION,
    "permutations": N_PERMUTATIONS,
    "random_seed": RANDOM_SEED,
    "permanova": perma,
    "permdisp": {"H": float(H_disp), "p_value": float(p_disp)},
    "nmds_stress": nmds_stress,
    "cevi": cevi["CEVI"].to_dict(),
    "analyses": [
        "inventory_summary", "taxonomic_richness", "Hill_q0_q1_q2",
        "Shannon", "Simpson", "Pielou", "Shapiro_Wilk",
        "Kruskal_Wallis", "Mann_Whitney_Bonferroni", "epsilon_squared",
        "Chao1", "Jackknife1", "rarefaction", "extrapolation",
        "Bray_Curtis", "Jaccard", "hierarchical_clustering",
        "PERMANOVA", "PERMDISP", "NMDS_2D", "NMDS_3D",
        "IndVal_Dufrene_Legendre", "SIMPER", "IVI",
        "stand_structure", "diameter_classes", "IUCN", "CITES",
        "biological_type", "chorological_type", "chi_square_heritage",
        "CEVI"
    ]
}
write_json(audit, "analysis_audit.json")

print("=" * 78)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 78)
print(f"Raw records: {len(raw)}")
print(f"Quantitative records: {len(plots)}")
print(f"Identified records: {len(div)}")
print(f"Unidentified quantitative records: {int(plots.Espece.isna().sum())}")
print(f"Species: {div.Espece.nunique()} | Genera: {div.Genre.nunique()} | Families: {div.Famille.nunique()}")
print(f"Plots: {div.Placette.nunique()} | Plot area: {PLOT_AREA_HA:.4f} ha")
print(f"PERMANOVA: F={perma['Pseudo_F']:.4f}, R2={perma['R2']:.4f}, p={perma['p_value']:.4f}")
print(f"PERMDISP: H={H_disp:.4f}, p={p_disp:.4f}")
print(f"NMDS 2D stress: {nmds_stress[2]:.6f}")
print(f"NMDS 3D stress: {nmds_stress[3]:.6f}")
print("CEVI:")
for s in SITE_ORDER:
    print(f"  {s}: {cevi.loc[s, 'CEVI']:.2f} ({cevi.loc[s, 'Ecological_value_class']})")
print(f"Tables: {len(list(TABLES_DIR.glob('*')))}")
print(f"Figures: {len(list(FIGURES_DIR.glob('*')))}")
print("=" * 78)
