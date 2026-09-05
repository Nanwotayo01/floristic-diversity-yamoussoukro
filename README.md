# Floristic Diversity of Urban Forest Sites in Yamoussoukro, Côte d'Ivoire

## Overview

This repository contains the floristic inventory data and Python analysis workflow associated with a study of plant diversity and vegetation structure across three surveyed sites in Yamoussoukro, Côte d'Ivoire: **Golf**, **FUY**, and **KONGHO**.

The repository is intended to support **transparent, reproducible analysis** and facilitate the reuse and verification of the analytical workflow.

## Repository contents

```text
floristic-diversity-yamoussoukro/
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── data/
│   └── Base_floristique_Yamoussoukro.xlsx
├── scripts/
│   └── Script-analyses1.ipynb
├── results/
│   ├── figures/
│   └── tables/
└── docs/
    └── DATA_DICTIONARY.md
```

## Data

The main dataset is:

`data/Base_floristique_Yamoussoukro.xlsx`

The workbook contains the floristic inventory data supplied for the analysis. The principal data fields include site, plot, species, genus, family, circumference, diameter, basal area, height, biological type, chorological type, IUCN status and CITES-related fields.

**Important:** the dataset should be checked for institutional, ethical, privacy, biodiversity-sensitive-location, or other restrictions before being made publicly accessible.

## Analysis workflow

The analysis workflow is provided as:

`scripts/Script-analyses1.ipynb`

The notebook should be run from the repository root after installing the dependencies listed in `requirements.txt`.

### Reproducibility

Create an environment and install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

and open:

`scripts/Script-analyses1.ipynb`

## Data-processing note

Analytical results should always be interpreted together with the data-cleaning and filtering operations implemented in the notebook. In particular, users reproducing manuscript results should verify the treatment of observations outside quantitative plots, missing species identifications, and other quality-control decisions documented in the analysis workflow.

## Main diversity metrics

The workflow includes floristic diversity analyses and related calculations present in the supplied notebook. The repository is intended to preserve the analysis as actually implemented rather than silently replacing the original workflow.

## Chao1 confidence intervals

The manuscript analysis includes Chao1 richness estimates and confidence intervals. When updating or reproducing these results, the exact estimator and uncertainty procedure should be documented explicitly and applied consistently across sites and the pooled dataset.

## Results

The `results/` directory is reserved for derived outputs:

- `results/figures/` — publication-ready figures;
- `results/tables/` — derived tables.

No derived result is included in this initial package unless generated directly from the supplied workflow.

## Citation

If you use these data or scripts, please cite the associated scientific publication/manuscript and, where appropriate, this repository.

A machine-readable citation file is provided in `CITATION.cff`.

## Licence and data-use conditions

The code and data should not automatically be assumed to have the same licence. The repository owner should confirm the appropriate licence and any restrictions applying to the dataset before public release.

## Contact

**Nanwotayo Jacques**

Côte d'Ivoire

