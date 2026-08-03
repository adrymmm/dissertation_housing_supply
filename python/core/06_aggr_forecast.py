import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from python.functions.forecast import quarter_str_from_dates, dm_test, plot_rmse_bar

# IMPORTANT: RUN 08_horse_race.R first for h1_forecasts.csv

# --- CONFIG ---
FORECASTS_DIR = ROOT / "data" / "outputs" / "forecasts"

# (display_name, filename, quarter_col, pred_col, actual_col, date_to_quarter)
SOURCES = [
    ("RW_py",   "chronos_forecasts.csv", "Quarter", "rw",      "actual", False),
    ("Chronos", "chronos_forecasts.csv", "Quarter", "chronos", "actual", False),
    ("ARRF",    "rf_forecasts.csv",      "Quarter", "rf",      "actual", False),
    ("LSTM",    "lstm_forecasts.csv",    "Quarter", "lstm",    "actual", False),
    ("RW",      "h1_forecasts.csv",      "date",    "rw",      "actual", True),  # R impl
    ("SNAIVE",  "h1_forecasts.csv",      "date",    "snaive",  "actual", True),
    ("AR",      "h1_forecasts.csv",      "date",    "ar",      "actual", True),
    ("TSLM",    "h1_forecasts.csv",      "date",    "tslm",    "actual", True),
    ("TSLM_s",  "h1_forecasts.csv",      "date",    "tslm_s",  "actual", True),
    ("VECM",    "h1_forecasts.csv",      "date",    "vecm",    "actual", True),
    ("ARDL",    "h1_forecasts.csv",      "date",    "ardl",    "actual", True),
    ("NARDL",   "h1_forecasts.csv",      "date",    "nardl",   "actual", True),
]

# ARDL/NARDL receive realised target-quarter covariates at lag 0, so their accuracy
# is conditional on correct inputs and not comparable to models forecasting from own
# history alone. Scored in separate panels; DM vs naive runs on UNCOND only.
UNCOND = ["RW", "SNAIVE", "AR", "TSLM", "TSLM_s", "VECM"]
COND = ["ARDL", "NARDL", "Ensemble_avg", "Ensemble_invmse"]
ML = ["ARRF", "LSTM", "Chronos"]

# SNAIVE is the binding benchmark: lhstarts is strongly seasonal, so a non-seasonal
# RW is a soft bar (tslm sd 0.033 against actual sd 0.285).
BENCHMARKS = ["RW", "SNAIVE"]

# Chronos excluded: disqualified from the forward projection on multi-step grounds,
# so a blend containing it would not be deployable.
BLEND_MODELS = ["VECM", "ARDL", "NARDL", "ARRF", "LSTM"]
WARMUP = 8

EXCLUDE_QUARTERS = ["2020Q2", "2020Q3"]


# --- Loading loop ---
frames = {}
for name, fname, qcol, pcol, acol, dtc in SOURCES:
    df = pd.read_csv(f"{FORECASTS_DIR}/{fname}")[[qcol, pcol, acol]].copy()
    if dtc:
        df["Quarter"] = quarter_str_from_dates(df[qcol])
    else:
        df = df.rename(columns={qcol: "Quarter"})
    frames[name] = df[["Quarter", pcol, acol]].rename(
        columns={pcol: name, acol: f"actual_{name}"}
    )

# --- Merge on Quarter, inner join so we only keep fully-overlapping rows ---
merged = None
for name, df in frames.items():
    merged = df if merged is None else merged.merge(df, on="Quarter", how="inner")

n_rows = len(merged)
print(f"Merged on {n_rows} quarters: {merged['Quarter'].min()} to {merged['Quarter'].max()}\n")

# --- Alignment checks ---
for name, df in frames.items():
    if len(df) != n_rows:
        print(f"  WARNING: {name}'s file has {len(df)} rows but only {n_rows} merged")

actual_cols = [c for c in merged.columns if c.startswith("actual_")]
ref = merged[actual_cols[0]].to_numpy()
for c in actual_cols[1:]:
    if not np.allclose(merged[c].to_numpy(), ref, atol=1e-6):
        print(f"  WARNING: actual values disagree between models on {c}")

if not np.allclose(merged["RW_py"].to_numpy(), merged["RW"].to_numpy(), atol=1e-6):
    max_diff = np.max(np.abs(merged["RW_py"] - merged["RW"]))
    print(f"  WARNING: Python's RW (chronos_forecasts.csv) and R's RW "
          f"(h1_forecasts.csv) disagree, max abs diff={max_diff:.4f} -- the R and "
          f"Python pipelines are NOT looking at the same window/series. Fix this "
          f"before trusting anything below.")
else:
    print("  OK: Python RW and R RW agree exactly -- the two pipelines are aligned "
          "on the same window and target series.\n")

# --- Sort chronologically (required for recursive ensemble weighting) ---
merged["_q"] = pd.PeriodIndex(merged["Quarter"], freq="Q")
merged = merged.sort_values("_q").reset_index(drop=True)
actual = merged[actual_cols[0]].to_numpy()

# --- Ensemble: simple average + recursive inverse-MSE ---
ens_avg = np.full(n_rows, np.nan)
ens_invmse = np.full(n_rows, np.nan)

for t in range(n_rows):
    preds_t = merged.loc[t, BLEND_MODELS].to_numpy(dtype=float)
    ens_avg[t] = np.nanmean(preds_t)
    if t < WARMUP:
        ens_invmse[t] = ens_avg[t]
        continue
    mses = np.array([
        np.mean((actual[:t] - merged[m].to_numpy()[:t]) ** 2) for m in BLEND_MODELS
    ])
    w = (1 / mses) / np.sum(1 / mses)
    ens_invmse[t] = np.sum(w * preds_t)

merged["Ensemble_avg"] = ens_avg
merged["Ensemble_invmse"] = ens_invmse


# --- Accuracy table ---
def best_of(errs, cands):
    return min(cands, key=lambda m: np.sqrt(np.mean(errs[m] ** 2)))


def report(mask, label):
    print(f"\n=== {label} ({mask.sum()} quarters) ===")
    errs = {}
    for name in UNCOND + COND + ML:
        e = (actual - merged[name].to_numpy(dtype=float))[mask]
        if np.isnan(e).any():
            print(f"  WARNING: {name} has {np.isnan(e).sum()} NaN in this mask")
        errs[name] = e

    # ME alongside RMSE/MAE: both of those are magnitude-only, and systematic bias
    # compounds over the 13-quarter projection where symmetric noise does not.
    for panel, members in [("Unconditional", UNCOND), ("Conditional", COND), ("ML", ML)]:
        print(f"\n  [{panel}]")
        print(f"  {'Model':18s} {'RMSE':>8s} {'MAE':>8s} {'ME':>9s}")
        for name in members:
            e = errs[name]
            print(f"  {name:18s} {np.sqrt(np.mean(e ** 2)):8.4f} "
                  f"{np.mean(np.abs(e)):8.4f} {np.mean(e):+9.4f}")

    for bench in BENCHMARKS:
        print(f"\n  DM vs {bench} (stat>0 => worse than {bench}; UNCOND only):")
        for name in UNCOND:
            if name == bench:
                continue
            s, pv = dm_test(errs[name], errs[bench])
            flag = "  *sig 5%" if pv < 0.05 else ""
            print(f"    {name:18s} stat={s:+.3f}  p={pv:.3f}{flag}")

    struct_best = best_of(errs, ["VECM", "ARDL", "NARDL"])
    ml_best = best_of(errs, ML)
    print(f"\n  Best structural = {struct_best}, best ML = {ml_best}")
    for a, b in [(ml_best, struct_best),
                 ("Ensemble_avg", struct_best),
                 ("Ensemble_invmse", struct_best)]:
        if a == b:
            continue
        s, pv = dm_test(errs[a], errs[b])
        flag = "  *sig 5%" if pv < 0.05 else ""
        print(f"    DM {a} vs {b}: stat={s:+.3f}  p={pv:.3f} "
              f"(stat>0 => {a} worse){flag}")

    return errs


full_mask = np.ones(n_rows, dtype=bool)
excov_mask = ~merged["Quarter"].isin(EXCLUDE_QUARTERS).to_numpy()

errors = report(full_mask, "Full sample")
excov_errors = report(excov_mask, "Excluding COVID (2020Q2-Q3)")

plot_rmse_bar(errors, title="Forecast RMSE - Full Sample")
plot_rmse_bar(excov_errors, title="Forecast RMSE - Excluding COVID")