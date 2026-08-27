import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
from python.functions.forecast import quarter_str_from_dates, plot_rmse_bar, report, run_ljungbox, run_mcs, run_spa


# IMPORTANT: RUN 08_horse_race.R first for h1_forecasts.csv + h1_model_roles.csv

# --- CONFIG ---
FORECASTS_DIR = ROOT / "data" / "outputs" / "forecasts"
FIGURES_DIR = ROOT / "data" / "outputs" / "figures"

# Column roles (matching R): *_rw = frozen, full-sample-selected structural spec
# (unconditional); VECM carries no suffix but is the same frozen spec.
BENCH = ["RW", "SNAIVE", "AR", "TSLM", "TSLM_s"]
# ARRF/GTVP are the two MRF specs from R/07_MacroRF.R.
ML_TESTED = ["ARRF", "GTVP", "LSTM", "Chronos"]
ML_ALL = ML_TESTED
# Chronos-Bolt gives a clean forward path (unlike t5, which collapsed onto a token-bin lattice), so it's now tested and blended; set ENSEMBLE_ML = ["ARRF", "LSTM"] to test Chronos without blending it.
ENSEMBLE_ML = ML_TESTED
ENSEMBLE = ["Ensemble_avg", "Ensemble_invmse"]

# Frozen keeps the full-sample-selected choices (lag orders / K / rank); its spec saw
# the evaluation window, so results are disclosure-only, not a real-time out-of-sample test.
struct = ["VECM", "ARDL_rw", "NARDL_rw"]
FROZEN_DISCLOSURE = ["VECM", "ARDL_rw", "NARDL_rw"]

SPA_BENCHMARKS = ["RW"]
WARMUP = 8

# (display_name, filename, quarter_col, pred_col, actual_col, date_to_quarter)
SOURCES = [
    ("RW_py", "chronos_forecasts.csv", "Quarter", "rw", "actual", False),
    ("Chronos", "chronos_forecasts.csv", "Quarter", "chronos", "actual", False),
    ("LSTM", "lstm_headline_forecast.csv", "date", "lstm", "actual", True),
    ("RW", "h1_forecasts.csv", "date", "rw", "actual", True),
    ("SNAIVE", "h1_forecasts.csv", "date", "snaive", "actual", True),
    ("AR", "h1_forecasts.csv", "date", "ar", "actual", True),
    ("TSLM", "h1_forecasts.csv", "date", "tslm", "actual", True),
    ("TSLM_s", "h1_forecasts.csv", "date", "tslm_s", "actual", True),
    ("VECM", "h1_forecasts.csv", "date", "vecm", "actual", True),
    ("ARDL_rw", "h1_forecasts.csv", "date", "ardl_rw", "actual", True),
    ("NARDL_rw", "h1_forecasts.csv", "date", "nardl_rw", "actual", True),
    ("ARRF", "h1_forecasts.csv", "date", "arrf", "actual", True),
    ("GTVP", "h1_forecasts.csv", "date", "gtvp", "actual", True),
]


roles = pd.read_csv(FORECASTS_DIR / "h1_model_roles.csv").set_index("model")["role"]
role_of = {name: roles.get(pcol)
           for (name, fname, _, pcol, _, _) in SOURCES
           if fname == "h1_forecasts.csv"}
# Frozen-spec columns must be tagged frozen_spec in R; this assertion catches any
# VARIANTS edit that would silently swap in an unconditional-selection column instead.
assert all(role_of.get(m) == "frozen_spec" for m in FROZEN_DISCLOSURE), \
    f"frozen-spec columns are not tagged frozen_spec: {role_of}"

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


def invmse_ensemble(preds, actual, warmup, weight_mask):
    """Expanding-window inverse-MSE weights; nanmean skips a NaN in a model's history instead of poisoning later weights, and weight_mask restricts weighting history to the evaluation subsample so ex-COVID isn't weighted on COVID quarters."""
    out = np.nanmean(preds, axis=1)
    for t in range(warmup, len(preds)):
        hist = weight_mask[:t]
        mses = np.nanmean((actual[:t, None] - preds[:t, :])[hist] ** 2, axis=0)
        ok = np.isfinite(mses) & (mses > 0) & np.isfinite(preds[t])
        if ok.sum():
            w = (1 / mses[ok]) / np.sum(1 / mses[ok])
            out[t] = np.sum(w * preds[t, ok])
    return out


full_mask = np.ones(len(merged), dtype=bool)

PLOT_NOTE = ("Structural models use the frozen, full-sample-selected lag orders / K / "
             "rank; results are disclosure-only, not a real-time out-of-sample test.")

results = {}

blend = struct + ENSEMBLE_ML
blend_preds = merged[blend].to_numpy(dtype=float)

mcs_models = BENCH + struct + ML_TESTED
spa_models = mcs_models + ENSEMBLE
print("mcs_models:", mcs_models)
print("spa_models:", spa_models)

panels = [
    ("Unconditional benchmarks", BENCH),
    ("Structural (frozen full-sample spec)", struct),
    ("ML", ML_ALL),
    ("Ensembles", ENSEMBLE),
]

m = merged.copy()
m["Ensemble_avg"] = np.nanmean(blend_preds, axis=1)
m["Ensemble_invmse"] = invmse_ensemble(blend_preds, actual, WARMUP, full_mask)

res = report(actual, m, full_mask, "Full sample",
             panels=panels, mcs_models=mcs_models,
             spa_benchmarks=SPA_BENCHMARKS, spa_models=spa_models)
results["Full sample"] = res

lb = run_ljungbox(res["errs"], mcs_models + ENSEMBLE)
print("\n  [Ljung-Box, Full sample]")
print(lb.to_string(index=False))
plot_rmse_bar(res["errs"], FIGURES_DIR, title="Forecast RMSE - Full Sample", note=PLOT_NOTE)

# Testing block 4
for loss in ("sq", "abs"):
    spa4 = run_spa(res["errs"], "RW", spa_models, loss=loss, block_size=4)
    mcs4 = run_mcs(res["errs"], mcs_models, loss=loss, block_size=4)
    name = "MSE" if loss == "sq" else "MAE"
    print(f"\n  [block_size=4, {name}] p(consistent)={spa4['pvalues']['consistent']:.3f}"
        f"  better than RW: {', '.join(spa4['better']) or 'none'}")
    print(mcs4.to_string(index=False))
