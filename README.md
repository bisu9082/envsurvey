# Data and code: in situ NaI(Tl) G(E) dose rates vs FH40G, and Cs-137 spectral screening

**Manuscript:** Site-specific field comparison of in situ NaI(Tl) gamma-ray spectrometry for ambient dose equivalent rate estimation and Cs-137 spectral screening
**Journal:** Applied Radiation and Isotopes (ARI-D-26-00647, under revision)
**Corresponding authors:** M.-K. Kim (m.k.kim@mju.ac.kr), J. Kim (jokim@inje.ac.kr), K. Kang (bisu9082@gmail.com)

This repository contains the raw spectra, the Monte Carlo outputs and the four scripts that generate every number reported in the revised manuscript.

## Structure

```
data/spe/          10 field spectra (ORTEC MAESTRO .Spe, ASCII), 1200 s real / 1198 s live time each
data/mcnp_3inch/   20 MCNP 6.2 output files: 3-inch NaI(Tl) F8 pulse-height response, 0.05-3.0 MeV
code/dose_r2.py               G(E) x ICRP 74 H*(10)/Ka dose rates, comparison statistics,
                              uncertainty-term bases, bias-structure tests
code/cs137_consistent_roi.py  Cs-137 photopeak extraction with one procedure for all ten spectra
code/cosmic_r2.py             cosmic-ray ionizing component at the site elevation (UNSCEAR 2000)
                              and the cosmic-corrected G(E)/FH40G comparison
code/cosmic_in_window.py      upper bound of cosmic signal already inside the 50 keV-3.1 MeV window
logs/              output of the two scripts (as cited in the manuscript)
legacy_jer/        superseded files from the earlier (JER) submission - see note below
```

## Spectrum files

| File | Point | | File | Point |
|---|---|---|---|---|
| 3원점.Spe | P3 | | 1원점.Spe | P1 |
| 3북쪽.Spe | P3N5m | | 1북쪽.Spe | P1N5m |
| 3동쪽.Spe | P3E5m | | 1동쪽.Spe | P1E5m |
| 3서쪽.Spe | P3W5m | | 1서쪽.Spe | P1W5m |
| 3남쪽.Spe | P3S5m | | 1남쪽.Spe | P1S5m |

Site: Jeju, Republic of Korea, 664-691 m above sea level, 29 April 2026. Detector: 3 x 3 inch NaI(Tl), ORTEC digiBASE-RH, 1024 channels, 1 m above ground. The energy calibration (E = -26.44 + 3.0734 x channel keV) is in each file header. FH40G (FH 40 G-L10) readings at 1 m are listed in `code/dose_r2.py`. SHA-1 hashes of the spectrum files are printed in `logs/cs137_consistent_roi.log`.

## Reproducing the results

```
pip install numpy scipy
cd code
python3 dose_r2.py               # -> logs/dose_r2.log
python3 cs137_consistent_roi.py  # -> logs/cs137_consistent_roi.log, .csv
python3 cosmic_r2.py             # -> logs/cosmic_r2.log
python3 cosmic_in_window.py      # -> logs/cosmic_in_window.log
```

| Manuscript item | Log section |
|---|---|
| G(E)/FH40G ratio, Wilcoxon, d_z, Bland-Altman, Pearson | `dose_r2.log` [1] |
| k(E) = H*(10)/Ka values and channel mapping | `dose_r2.log` [2], [6] |
| Additive vs proportional bias tests | `dose_r2.log` [3] |
| Table 2 uncertainty terms | `dose_r2.log` [4], [7] |
| Cluster comparison (Fig. 4b) | `dose_r2.log` [8] |
| Cosmic-ray component, corrected ratio, Table 2 cosmic term | `cosmic_r2.log` [1]-[3] |
| Cosmic signal inside the G(E) window (upper bound) | `cosmic_in_window.log` |
| Table 1, Cs-137 fitted areas | `cs137_consistent_roi.log` [0], [B] |
| Fixed-window comparison, Currie levels | `cs137_consistent_roi.log` [A] |

## Note on the earlier version (legacy_jer/)

`legacy_jer/` holds the files deposited with the earlier submission. They are kept for transparency and **should not be used**:

- The G(E) dose rates there omit the ICRP 74 H*(10)/Ka conversion (ratio 0.399; corrected value 0.493).
- The Cs-137 net counts there were transcribed from regions of interest set manually at each point in the acquisition software, with widths that differed between points. The apparent 20.9x "hotspot" at P3E5m was an artefact of those settings and has been withdrawn. With one procedure applied to all spectra, Cs-137 is present at comparable levels at all ten points (`logs/cs137_consistent_roi.log`).
- The FH40G was described there as an ionisation chamber; it is a proportional-counter instrument.

## License

MIT. Please cite the associated article when using these data or code.
