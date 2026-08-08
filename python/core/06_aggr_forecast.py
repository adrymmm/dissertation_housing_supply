import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
from python.functions.forecast import quarter_str_from_dates, plot_rmse_bar, report

# IMPORTANT: RUN 08_horse_race.R first for h1_forecasts.csv

# --- CONFIG ---
FORECASTS_DIR = ROOT / "data" / "outputs" / "forecasts"
FIGURES_DIR = ROOT / "data" / "outputs" / "figures"

UNCOND = ["RW", "SNAIVE", "AR", "TSLM", "TSLM_s", "VECM"]
COND = ["ARDL", "NARDL", "Ensemble_avg", "Ensemble_invmse"]
ML = ["ARRF", "LSTM", "Chronos"]
BENCHMARKS = ["RW", "SNAIVE"]

# Chronos out since it cant predict forward
MCS_MODELS = [m for m in UNCOND + ML if m != "Chronos"]

BLEND_MODELS = ["VECM", "ARDL", "NARDL", "ARRF", "LSTM"]
WARMUP = 8
EXCLUDE_QUARTERS = ["2020Q2", "2020Q3"]

# (display_name, filename, quarter_col, pred_col, actual_col, date_to_quarter)
SOURCES = [
    ("RW_py", "chronos_forecasts.csv", "Quarter", "rw", "actual", False),
    ("Chronos", "chronos_forecasts.csv", "Quarter", "chronos", "actual", False),
    ("ARRF", "rf_forecasts.csv", "Quarter", "rf", "actual", False),
    ("LSTM", "lstm_spec4_forecast.csv", "date", "lstm", "actual", True),
    ("RW", "h1_forecasts.csv", "date", "rw", "actual", True),
    ("SNAIVE", "h1_forecasts.csv", "date", "snaive", "actual", True),
    ("AR", "h1_forecasts.csv", "date", "ar", "actual", True),
    ("TSLM", "h1_forecasts.csv", "date", "tslm", "actual", True),
    ("TSLM_s", "h1_forecasts.csv", "date", "tslm_s", "actual", True),
    ("VECM", "h1_forecasts.csv", "date", "vecm", "actual", True),
    ("ARDL", "h1_forecasts.csv", "date", "ardl", "actual", True),
    ("NARDL", "h1_forecasts.csv", "date", "nardl", "actual", True),
]

# Load, merge and chronologize
merged = None
for name, fname, qcol, pcol, acol, dtc in SOURCES:
    df = pd.read_csv(FORECASTS_DIR / fname)[[qcol, pcol, acol]]
    df["Quarter"] = quarter_str_from_dates(df[qcol]) if dtc else df[qcol]
    df = df[["Quarter", pcol, acol]].rename(columns={pcol: name, acol: f"actual_{name}"})
    merged = df if merged is None else merged.merge(df, on="Quarter", how="inner")

merged = merged.assign(_q=pd.PeriodIndex(merged["Quarter"], freq="Q")).sort_values("_q").reset_index(drop=True)

actual_cols = [c for c in merged.columns if c.startswith("actual_")]
actual = merged[actual_cols[0]].to_numpy(dtype=float)

# Test actuals across files for vintages
for c in actual_cols[1:]:
    assert np.allclose(merged[c], actual, atol=1e-6, equal_nan=True), \
        f"Actuals disagree between {actual_cols[0]} and {c}"

# Check if RW in python aligns with RW in R
assert np.allclose(merged["RW_py"], merged["RW"], atol=1e-6), "Python and R RW series misaligned!"


def invmse_ensemble(preds, warmup, weight_mask):
    """Expanding-window inverse-MSE weights. nanmean, so a model with a NaN
    in its history is skipped at that t rather than poisoning every later
    weight. weight_mask restricts the weighting history to the evaluation
    subsample, so the ex-COVID panel isn't weighted on COVID quarters."""
    out = np.nanmean(preds, axis=1)
    for t in range(warmup, len(preds)):
        hist = weight_mask[:t]
        mses = np.nanmean((actual[:t, None] - preds[:t, :])[hist] ** 2, axis=0)
        ok = np.isfinite(mses) & (mses > 0) & np.isfinite(preds[t])
        if ok.sum():
            w = (1 / mses[ok]) / np.sum(1 / mses[ok])
            out[t] = np.sum(w * preds[t, ok])
    return out


blend_preds = merged[BLEND_MODELS].to_numpy(dtype=float)
merged["Ensemble_avg"] = np.nanmean(blend_preds, axis=1)

full_mask = np.ones(len(merged), dtype=bool)
excov_mask = ~merged["Quarter"].isin(EXCLUDE_QUARTERS).to_numpy()

results = {}
for label, mask, fig_title in [
    ("Full sample", full_mask, "Forecast RMSE - Full Sample"),
    ("Excluding COVID (2020Q2-Q3)", excov_mask, "Forecast RMSE - Excluding COVID"),
]:
    m = merged.copy()
    m["Ensemble_invmse"] = invmse_ensemble(blend_preds, WARMUP, mask)
    res = report(actual, m, mask, label, uncond=UNCOND, cond=COND, ml=ML,
                 benchmarks=BENCHMARKS, mcs_models=MCS_MODELS)
    plot_rmse_bar(res["errs"], FIGURES_DIR, title=fig_title)
    results[label] = res