"""
Aggregate per-model one-step-ahead forecasts and run Diebold-Mariano tests.

Run this AFTER 04_chronos.py, 05_ARRF.py, 06_LSTM.py (Python) and the R
VECM/ARDL/NARDL script have each produced their forecast output in
../data/outputs/forecasts/. It merges every model's forecasts onto a common
Quarter key, checks the windows actually overlap before computing anything,
cross-checks that the R-side and Python-side RW benchmarks agree (they're
both just "next quarter = this quarter's actual" on the same series -- if
they don't match, something is misaligned between the two halves of the
pipeline), then prints an accuracy table and runs HLN-corrected
Diebold-Mariano tests: every model against RW, plus a head-to-head between
your best structural model and your best ML model.

Two source formats are handled:
  - Python scripts: one narrow CSV per model, columns (Quarter, <pred>, actual),
    Quarter already a "2010Q1"-style string.
  - R script: one wide CSV (h1_forecasts.csv), columns (date, actual, rw, ar,
    vecm, ardl, nardl), date an R Date string ("2010-01-01") that needs
    converting to the same "2010Q1" key before it can be merged with the rest.
"""
import numpy as np
import pandas as pd
from scipy.stats import t as tdist

FORECASTS_DIR = "data/outputs/forecasts"

# (display_name, filename, quarter_col, pred_col, actual_col, date_to_quarter)
SOURCES = [
    ("RW_py",   "chronos_forecasts.csv", "Quarter", "rw",      "actual", False),
    ("Chronos", "chronos_forecasts.csv", "Quarter", "chronos", "actual", False),
    ("ARRF",    "rf_forecasts.csv",      "Quarter", "rf",      "actual", False),
    ("LSTM",    "lstm_forecasts.csv",    "Quarter", "lstm",    "actual", False),
    ("RW_R",    "h1_forecasts.csv",      "date",    "rw",      "actual", True),
    ("AR",      "h1_forecasts.csv",      "date",    "ar",      "actual", True),
    ("VECM",    "h1_forecasts.csv",      "date",    "vecm",    "actual", True),
    ("ARDL",    "h1_forecasts.csv",      "date",    "ardl",    "actual", True),
    ("NARDL",   "h1_forecasts.csv",      "date",    "nardl",   "actual", True),
]

# RW_py exists only as a cross-check against RW_R (see below); RW_R is the
# canonical persistence benchmark used in the accuracy table and DM tests.
RW_BENCHMARK = "RW_R"

# Update these once you know which model actually has the lowest RMSE in
# each camp -- this pairing is just a starting guess.
STRUCTURAL_BEST = "VECM"
ML_BEST = "LSTM"


def dm_test(e1, e2, h=1, power=2):
    """H0: equal accuracy. Loss = |e|^power. stat>0 => model-1 worse.
    HLN small-sample correction; compared to t(n-1). Identical to the
    function in 04_chronos.py so results are directly comparable."""
    d = np.abs(e1) ** power - np.abs(e2) ** power
    nn = len(d)
    dbar = d.mean()
    var = np.sum((d - dbar) ** 2) / nn
    for lag in range(1, h):
        var += 2 * np.sum((d[lag:] - dbar) * (d[:-lag] - dbar)) / nn
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var / nn)
    stat *= np.sqrt((nn + 1 - 2 * h + h * (h - 1) / nn) / nn)
    return stat, 2 * tdist.cdf(-abs(stat), df=nn - 1)


def quarter_str_from_dates(date_series):
    dt = pd.to_datetime(date_series)
    return dt.dt.year.astype(str) + "Q" + dt.dt.quarter.astype(str)


# --- load each source ---
frames = {}
_raw_cache = {}  # avoid re-reading h1_forecasts.csv five times
for name, fname, qcol, pcol, acol, conv in SOURCES:
    path = f"{FORECASTS_DIR}/{fname}"
    if path not in _raw_cache:
        try:
            _raw_cache[path] = pd.read_csv(path)
        except FileNotFoundError:
            _raw_cache[path] = None
    raw = _raw_cache[path]
    if raw is None:
        print(f"SKIPPING {name}: {path} not found yet")
        continue

    df = raw[[qcol, pcol, acol]].copy()
    if conv:
        df["Quarter"] = quarter_str_from_dates(df[qcol])
    else:
        df = df.rename(columns={qcol: "Quarter"})
    frames[name] = df[["Quarter", pcol, acol]].rename(
        columns={pcol: name, acol: f"actual_{name}"}
    )

if len(frames) < 2:
    raise SystemExit("Need at least 2 models' forecast files to compare -- "
                      "check FORECASTS_DIR and that the upstream scripts ran.")

# --- merge on Quarter, inner join so we only keep fully-overlapping rows ---
merged = None
for name, df in frames.items():
    merged = df if merged is None else merged.merge(df, on="Quarter", how="inner")

n_rows = len(merged)
print(f"Merged on {n_rows} quarters: {merged['Quarter'].min()} to {merged['Quarter'].max()}\n")

# --- alignment checks BEFORE trusting any RMSE/DM number below ---
problems = False
for name, df in frames.items():
    if len(df) != n_rows:
        problems = True
        print(f"  WARNING: {name}'s file has {len(df)} rows but only {n_rows} "
              f"survived the merge -- its window doesn't fully overlap with "
              f"the others (check date range / path before trusting results)")

actual_cols = [c for c in merged.columns if c.startswith("actual_")]
ref = merged[actual_cols[0]].to_numpy()
for c in actual_cols[1:]:
    if not np.allclose(merged[c].to_numpy(), ref, atol=1e-6):
        problems = True
        print(f"  WARNING: actual values disagree between models on {c} -- "
              f"the merge likely produced misaligned rows, check Quarter labels")

# R and Python both compute the same naive RW forecast independently -- if
# they don't match, the two halves of the pipeline are looking at different
# windows or different target transforms, and nothing else here is trustworthy.
if "RW_py" in merged.columns and "RW_R" in merged.columns:
    if not np.allclose(merged["RW_py"].to_numpy(), merged["RW_R"].to_numpy(), atol=1e-6):
        problems = True
        max_diff = np.max(np.abs(merged["RW_py"] - merged["RW_R"]))
        print(f"  WARNING: Python's RW (chronos_forecasts.csv) and R's RW "
              f"(h1_forecasts.csv) disagree, max abs diff={max_diff:.4f} -- "
              f"the R and Python pipelines are NOT looking at the same "
              f"window/series. Fix this before trusting anything below.")
    else:
        print("  OK: Python RW and R RW agree exactly -- the two pipelines "
              "are aligned on the same window and target series.\n")

if problems:
    print("\n-> Fix the warnings above before trusting the table below.\n")

actual = ref

# --- accuracy table ---
model_names = [n for n in frames if n != "RW_py"]  # RW_py was just the cross-check
print(f"{'Model':10s} {'RMSE':>8s} {'MAE':>8s}")
errors = {}
for name in model_names:
    e = actual - merged[name].to_numpy()
    errors[name] = e
    rmse = np.sqrt(np.mean(e ** 2))
    mae = np.mean(np.abs(e))
    print(f"{name:10s} {rmse:8.4f} {mae:8.4f}")

# --- DM tests: every model vs RW persistence benchmark ---
if RW_BENCHMARK in errors:
    print(f"\nDiebold-Mariano vs {RW_BENCHMARK} (stat > 0 => model worse than RW):")
    for name in model_names:
        if name == RW_BENCHMARK:
            continue
        s, pv = dm_test(errors[name], errors[RW_BENCHMARK])
        flag = "  *significant at 5%" if pv < 0.05 else ""
        print(f"  {name:10s} stat={s:+.3f}  p={pv:.3f}{flag}")

# --- DM test: best structural model vs best ML model ---
if STRUCTURAL_BEST in errors and ML_BEST in errors:
    s, pv = dm_test(errors[ML_BEST], errors[STRUCTURAL_BEST])
    flag = "  *significant at 5%" if pv < 0.05 else ""
    print(f"\nDM {ML_BEST} vs {STRUCTURAL_BEST}: stat={s:+.3f}  p={pv:.3f}"
          f"  (stat>0 => {ML_BEST} worse){flag}")
else:
    print(f"\n(Skipping structural-vs-ML head-to-head -- '{STRUCTURAL_BEST}' or "
          f"'{ML_BEST}' not in {list(errors.keys())}.)")
