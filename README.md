# Spaceflight VEN Behavioral Analysis

**Connecting the Fast Lane Hypothesis to NASA Open Science Data**

> Companion analysis to [The Fast Lane Hypothesis](https://github.com/esila-keskin/fast-lane-hypothesis)  
> using NASA OSD-618 open-access behavioral data

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Data](https://img.shields.io/badge/data-NASA%20OSD--618-orange.svg)](https://osdr.nasa.gov/bio/repo/data/study/OSD-618)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![AWG](https://img.shields.io/badge/NASA-Brain%20AWG-blue.svg)](https://awg.osdr.space)

---

## Overview

The **Fast Lane Hypothesis** proposes that Von Economo Neurons (VENs) implement a biological speed-accuracy tradeoff in social decision circuits. This repository tests that prediction against open NASA behavioral data from **OSD-618** - exposing mice to simultaneous spaceflight stressors (galactic cosmic radiation, hindlimb unloading, social isolation) and measuring cognition across six behavioral assays.

The combined stressor condition is treated as a proxy for the **FTD-like VEN ablation** condition from the computational model.

---

## Hypothesis Mapping

| Behavioral Assay | VEN-Relevant Domain | Prediction |
|-----------------|--------------------|--------------------|
| Three Chamber Social Test | Social approach speed | Reduced social preference |
| Novel Object Recognition | Rapid novelty detection | Lower discrimination index |
| Radial Arm Water Maze | Spatial speed-accuracy | More errors, slower learning |
| Open Field | Locomotor baseline | No deficit (confound control) |
| Balance Beam | Sensorimotor | No deficit (confound control) |

---

## Dataset

**NASA OSD-618** - Rienecker et al. (2023)  
DOI: [10.1038/s41598-023-28508-0](https://doi.org/10.1038/s41598-023-28508-0)  
OSDR: [osdr.nasa.gov/bio/repo/data/study/OSD-618](https://osdr.nasa.gov/bio/repo/data/study/OSD-618)

590 C57BL/6J mice · 4 conditions · 6 behavioral assays

---

## Repository Structure
```
spaceflight-ven-analysis/
├── data/                    ← place TRANSFORMED.csv files here
├── analysis/
│   └── ven_spaceflight_analysis.py
├── figures/                 ← generated PDF figures
├── results/                 ← JSON summary statistics
├── requirements.txt
└── README.md
```
---

## Data Download

1. Go to [NASA OSDR OSD-618](https://osdr.nasa.gov/bio/repo/data/study/OSD-618)
2. Click **Files** tab
3. Download the `_TRANSFORMED.csv` from each folder:
   - Three Chamber Social Test
   - Novel Object Recognition
   - Radial Arm Water Maze
   - Open Field
   - Balance Beam
4. Place all five in `data/`

---

## Usage
```bash
pip install -r requirements.txt
python analysis/ven_spaceflight_analysis.py
```

Outputs:
- `figures/fig_ven_spaceflight_overview.pdf`
- `figures/fig_deficit_heatmap.pdf`
- `results/ven_spaceflight_results.json`

---

## Relationship to Fast Lane Model

| Computational condition | This behavioral analog |
|------------------------|----------------------|
| Typical (2% VENs) | Control mice |
| Autism-like (0.4% VENs) | Radiation-only or isolation-only |
| FTD-like (VEN ablation) | Combined stressor condition |

---

## Paper

Companion to: **The Fast Lane Hypothesis: Von Economo Neurons Implement a Biological Speed-Accuracy Tradeoff** - Esila Keskin, UWE Bristol (2026)

---

## NASA AWG

Open project under the **Brain AWG**, NASA OSDR Analysis Working Groups.

---

## License

MIT
