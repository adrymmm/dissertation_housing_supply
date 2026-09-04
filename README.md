# Dissertation: England Housing Starts VECM

This repo implements the empirical pipeline for the dissertation: a Johansen
VECM of England private housing starts (with ARDL/NARDL and MacroRF
horse-race comparisons), used to forecast starts against the government's
1.5m-homes target.

## Repo structure

```
R/                  Core econometric pipeline (numbered scripts, run in order)
  functions/         Small shared helpers sourced by the pipeline scripts
  diagnostics/        Supplementary diagnostic scripts (X_ prefix), not part of the numbered run order
  models/             .rds/.csv model outputs written by the pipeline (gitignored, regenerated on run)
  vendor/             Third-party R code sourced by the pipeline (Lee-Strazicich unit root test, MacroRF)
  marshall_zhang/    Standalone replication of Marshall & Zhang; not sourced by the numbered pipeline
python/
  core/              Data cleaning, Chronos/LSTM forecasts, OBR scenario build, starts-to-completions bridge
  functions/         Shared helpers (bridge.py, forecast.py) imported by python/core notebooks/scripts
  extension/         Planning/social-VAR/reform-scenario extension notebooks
  misc/              One-off exploratory notebooks, not part of the reproduction path
data/
  raw/               Vendored source files (tracked in git) — see Data sources below
  python_master/      Cleaned per-series CSVs and the combined england_master.csv, built by python/core/02
  processed/          Intermediate artefacts (e.g. completions_bridge.pkl)
  outputs/            Forecasts, figures and diagnostics written by the pipeline
team_files/          Legacy/superseded notebooks from earlier team drafts (GTVP, ARRF, LSTM) — not the
                     reporting pipeline; see R/07_MacroRF.R and python/core for the current versions
requirements.txt    Python environment (pip freeze)
R/sessionInfo.txt   R environment (sessionInfo() with package versions)
```

## Environment setup

- **Python**: `pip install -r requirements.txt`
- **R**: package versions used are recorded in `R/sessionInfo.txt`. There is no
  R package manifest (renv/packrat) — install the packages listed there
  (dplyr, readr, zoo, urca, vars, ARDL, sandwich, lmtest, car, strucchange,
  tseries, tempdisagg, pracma) plus ggplot2/tidyr for plotting.

Run everything from the repo root (`R/`, `data/` paths in the scripts are
relative to repo root, not to `R/`).

## How to reproduce

`R/models/` is gitignored — none of the `.rds`/`.csv` files in it are
checked in, so the pipeline has to actually be run to regenerate them.
`data/raw/`, `data/python_master/`, `data/processed/` and most of
`data/outputs/` (excluding `.png` figures) are checked in.

**Python (`python/core/`, in order):**

1. `01_data_exploration.ipynb` — exploratory read of the raw MHCLG starts
   series; no persisted output
2. `02_data_cleaning.ipynb` — cleans everything under `data/raw/` into
   `data/python_master/*.csv` and assembles `data/python_master/england_master.csv`
   — **must run before `R/01_data_loading.R`**
3. `03_chronos.py` — Chronos foundation-model forecast, writes
   `data/outputs/forecasts/chronos_forecasts.csv`, `chronos_forward.csv`
4. `04_LSTM.py` — LSTM forecast + CV/robustness diagnostics, writes to
   `data/outputs/lstm/`
5. `05_aggr_forecast.py` — aggregates R and Python forecasts into a single
   horse-race summary, writes `data/outputs/forecasts/horse_race_summary.csv`
   — **needs `data/outputs/forecasts/h1_model_roles.csv` from `R/08_horse_race.R` first**
6. `06_starts_to_completions.py` — builds the starts-to-completions bridge,
   writes `data/processed/completions_bridge.pkl`
7. `07_obr_data.ipynb` — builds forward covariate paths under the OBR
   scenario, writes `data/python_master/OBR/obr_scenario.csv` — **live-scrapes
   the construction cost index from costmodelling.com at runtime (needs
   internet access); needed before `R/09_foreARDL.R`**
8. `08_net_additions.ipynb` — bridges starts forecasts to net additional
   dwellings against the 1.5m target — needs outputs of (6) and `R/09_foreARDL.R`

**R (`R/`, in order):**

1. `01_data_loading.R` — builds the transformed variable matrix from
   `data/python_master/england_master.csv`, writes `R/models/eng_tf.rds`, `eng_df.rds`
2. `02_stationarity.R` — ADF/KPSS/Zivot-Andrews/Lee-Strazicich unit root
   diagnostics (console/plot output only)
3. `03_vecm_core.R` — headline Johansen VECM (K=5, rank test), writes
   `R/models/jo_eng.rds`, `var_eng.rds`, `final_dummies.rds`
4. `04_vecm_results.R` — long-run elasticities and short-run ECM from the
   headline VECM (console output only)
5. `05_ARDL.R` — ARDL bounds test/ECM, writes `R/models/ardl_mod.rds`,
   `ardl_best.rds`, `ardl_ecm.rds`, `ardl_coefs_full.csv`
6. `06_NARDL.R` — nonlinear ARDL, writes `R/models/nardl_eng_zoo.rds`,
   `nardl_dum_full.rds`, `nardl_fits_full.rds`, `nardl_summary.csv`
7. `07_MacroRF.R` — ARRF/GTVP MacroRF fits, writes `R/models/mrf_h1.rds`,
   `mrf_gtvp.rds`, and `data/outputs/forecasts/mrf_gtvp_betas.csv`, `mrf_gtvp_vi.csv`
8. `08_horse_race.R` — pseudo-out-of-sample horse race across
   VECM/ARDL/NARDL/MacroRF, writes `data/outputs/forecasts/h1_forecasts.csv`,
   `h1_model_roles.csv`, `h1_vecm_rank_trace.csv`
9. `09_foreARDL.R` — ARDL/NARDL forward forecast under the OBR scenario
   (reads `data/python_master/OBR/obr_scenario.csv` from Python step 7),
   writes `data/outputs/forecasts/obr_scenario_forecasts.csv`
10. `10_foreVECM.R` — VECM unconditional forward forecast, writes
    `data/outputs/forecasts/vecm_unconditional_forecast.csv`, `vecm_implied_covariates.csv`

`R/diagnostics/X_*.R` are supplementary robustness/diagnostic scripts (serial
correlation scans, symmetry tests, rate/planning-deficit sensitivity, etc.),
each run against the model objects its corresponding numbered stage produces.

## Data sources

All raw inputs under `data/raw/` are vendored and tracked in git — no manual
download is needed to reproduce the pipeline as checked in:

| File | Source (as named/used in code) |
|---|---|
| `OBR/efo-march-2026-detailed-forecast-tables-economy.xlsx` | OBR Economic and Fiscal Outlook, March 2026, detailed forecast tables (economy) |
| `starts/indicatorsofukhousebuilding.xlsx`, `housing_stock/indicatorsofukhousebuilding.xlsx` | MHCLG house building live tables (starts/completions, by nation) |
| `housing_stock/LiveTable104.ods` | MHCLG Live Table 104 (dwelling stock) |
| `net_additions/Live_Table_120.ods` | MHCLG Live Table 120 (net additional dwellings) |
| `planning_applications/PS2_data_-_open_data_table__202512_.csv` | MHCLG planning applications live table PS2 |
| `transactions/Table_584.xlsx` | MHCLG/DLUHC live table 584 (property transactions, England & Wales) |
| `nominal_house_price/Average-prices-2026-03.csv`, `transactions/Sales-2026-03.csv` | ONS/Land Registry UK House Price Index |
| `construction_cost/14-p157e-output-price-indices-2014Q2_rev1.xls`, `bulletindataset9.xlsx` | ONS construction output price indices |
| `gdp_deflator/series-290526.csv`, `income_deflator/series-290526.xls`, `NRJR.csv`, `RPHQ.csv` | ONS time series (GDP/income deflators) |
| `rate/Bank Rate history and data  Bank of England Database.csv` | Bank of England Bank Rate database |

Exact download URLs are not recorded in the code or comments for any of the
above, so treat the file-to-agency mapping as inferred from filenames/variable
names rather than confirmed provenance.

The one input that is **not** vendored: `python/core/07_obr_data.ipynb`
scrapes the construction cost index (BCI) live from
`https://costmodelling.com/construction-indices` at runtime. Reproducing that
notebook requires internet access; there is no cached copy of that series in
the repo.

## Known limitations

- **Serial correlation in the VECM/ARDL residuals does not clear at any lag
  order tested** (`R/diagnostics/X_vecm_diagnostics.R` scans K, `X_ardl_diagnostics.R`
  runs Breusch-Godfrey at multiple orders). This is treated as a documented
  feature of the data, not a bug to be fixed by further lag search.
- **The headline Johansen rank decision (r=1) holds by a narrow margin** —
  see `trace_rank_decision()` output in `R/03_vecm_core.R`.
- Model objects and figures are not checked into git (`R/models/`, `*.png`
  are gitignored); the pipeline must be run to regenerate them before any
  downstream diagnostics or plotting scripts will find their inputs.
