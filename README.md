# Data and Code Repository

**Paper:** Systematic underestimation of ambient dose rate by in-situ NaI(Tl) gamma spectrometry using the G-factor method: field validation and Cs-137 spatial mapping at an agricultural site

**Journal:** Journal of Environmental Radioactivity (Short Communication, submitted 2026)

**Authors:** H. You, W. Bak, G. Choi, M. Shin, J. Yoo, J. Kim\*, K. Kang\*

**Corresponding:** jokim@inje.ac.kr · bisu9082@gmail.com

---

## Repository structure

```
upload/
├── data/
│   ├── raw_results.csv          # Per-point dose rate and ratio results (EXP1)
│   ├── experiment_summary.json  # Aggregated statistics for all experiments
│   └── results_table.tex        # LaTeX-formatted results table
└── code/
    ├── analysis_main.py         # Main statistical analysis (EXP1–3)
    └── make_figures_jer.py      # Figure generation for all JER figures (Fig 1–4)
```

Raw SPE spectrum files (10 × 1200 s acquisitions) are available upon reasonable request to the corresponding authors. They are not included here due to proprietary instrument format constraints.

---

## Data description

### `data/raw_results.csv`
Per-point measurement results for all 10 locations (P3 cluster and P1 cluster). Columns:

| Column | Description | Unit |
|---|---|---|
| `point_id` | Measurement point label (e.g. P3, P3E5m) | — |
| `fh40g_1m` | FH40G ionisation-chamber reading at 1 m height | µSv/h |
| `fh40g_floor` | FH40G reading at floor level | µSv/h |
| `gf_calc` | G-factor-calculated H*(10) dose rate | µSv/h |
| `ratio` | GF\_calc / FH40G\_1m | dimensionless |
| `cs137_net` | Cs-137 net count rate (661.7 keV peak, 1200 s) | counts |

### `data/experiment_summary.json`
Machine-readable summary of key statistics for EXP1 (dose rate agreement) and EXP3 (Cs-137 hotspot), including Wilcoxon test results, Cohen's d, Bland–Altman bias, and hotspot ratio.

### `data/results_table.tex`
LaTeX source for the results table included in the manuscript.

---

## Code description

### `code/analysis_main.py`
Runs the full statistical analysis pipeline:
- EXP1: G-factor vs FH40G ratio analysis (Wilcoxon signed-rank, Cohen's d, Bland–Altman)
- EXP3: Cs-137 spatial heterogeneity and hotspot quantification

**Usage:**
```bash
python analysis_main.py
```
Outputs: `raw_results.csv`, `experiment_summary.json`, `results_table.tex`

**Dependencies:** Python ≥ 3.9, numpy, scipy, pandas

### `code/make_figures_jer.py`
Generates all four manuscript figures (JER\_Fig1.png – JER\_Fig4.png) from the processed data embedded in the script.

**Usage:**
```bash
python make_figures_jer.py
```
Output directory: `../figure/` (relative to script location)

**Dependencies:** Python ≥ 3.9, numpy, scipy, matplotlib, cartopy (Fig 4 only)

---

## Reproducing the results

```bash
# 1. Run analysis
cd code/
python analysis_main.py

# 2. Regenerate figures
python make_figures_jer.py
```

All key results (ratio = 0.399, Wilcoxon p = 0.002, Cohen's d = −6.3, hotspot ratio = 20.9×) are printed to stdout by `analysis_main.py` and saved to `data/experiment_summary.json`.

---

## Field measurement conditions

| Parameter | Value |
|---|---|
| Site | Jeju Nanjichuksan Agricultural Research Center, Republic of Korea |
| Date | 29 April 2026 |
| Instrument (spectrometer) | NaI(Tl) 3×3 inch, 1024-channel MCA |
| Instrument (reference) | FH40G ionisation-chamber survey meter |
| Acquisition time | 1200 s live-time per point |
| Measurement height | 1.0 m above ground |
| Points | 10 (P3 cluster × 5, P1 cluster × 5) |
| Cs-137 hotspot | P3E5m: 1960 counts/1200 s (20.9× site background) |

---

## License

Data and code are released under the **MIT License**. When using this repository, please cite the associated manuscript.
