import numpy as np
import pandas as pd
from scipy.stats import t as tdist
import matplotlib.pyplot as plt

# IMPORTANT: RUN 07_forecast.R first for h1_forecasts.csv

# --- CONFIG ---
FORECASTS_DIR = f"data/outputs/forecasts"   # for some reason this starts at project root

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

# Cross-check benchmark for R and Python random walk
RW_BENCHMARK = "RW_R"
 
STRUCTURAL_BEST = "NARDL"
ML_BEST = "ARRF"


def dm_test(e1, e2, h=1, power=2):
    """H0: equal accuracy. Loss = |e|^power. stat>0 => model-1 worse.
    HLN small-sample correction; compared to t(n-1)."""
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

# Convert R-style date column to "YYYYQ#"
def quarter_str_from_dates(date_series):
    dt = pd.to_datetime(date_series)
    return dt.dt.year.astype(str) + "Q" + dt.dt.quarter.astype(str)


# --- Loading Loop ---
frames = {}
for name, fname, qcol, pcol, acol, dtc in SOURCES:
    df = pd.read_csv(f"{FORECASTS_DIR}/{fname}")[[qcol, pcol, acol]].copy()
    # R date conversion
    if dtc:
        df["Quarter"] = quarter_str_from_dates(df[qcol])
    else:
        df = df.rename(columns={qcol: "Quarter"})
    frames[name] = df[["Quarter", pcol, acol]].rename(
        columns={pcol: name, acol: f"actual_{name}"}
    )


# --- Merging on Quarter, inner join so we only keep fully-overlapping rows ---
merged = None
for name, df in frames.items():
    if merged is None:
       merged = df
    else:
        merged = merged.merge(df, on='Quarter', how='inner')

n_rows = len(merged)
print(f"Merged on {n_rows} quarters: {merged['Quarter'].min()} to {merged['Quarter'].max()}\n")

# --- Alignment Checks ---
for name, df in frames.items():
    if len(df) != n_rows:
        print(f"  WARNING: {name}'s file has {len(df)} rows but only {n_rows}")

actual_cols = [c for c in merged.columns if c.startswith("actual_")]
ref = merged[actual_cols[0]].to_numpy()
for c in actual_cols[1:]:
    if not np.allclose(merged[c].to_numpy(), ref, atol=1e-6):
        print(f"  WARNING: actual values disagree between models on {c}")

# Check if python RW and R RW match
if "RW_py" in merged.columns and "RW_R" in merged.columns:
    if not np.allclose(merged["RW_py"].to_numpy(), merged["RW_R"].to_numpy(), atol=1e-6):
        max_diff = np.max(np.abs(merged["RW_py"] - merged["RW_R"]))
        print(f"  WARNING: Python's RW (chronos_forecasts.csv) and R's RW "
              f"(h1_forecasts.csv) disagree, max abs diff={max_diff:.4f} -- "
              f"the R and Python pipelines are NOT looking at the same "
              f"window/series. Fix this before trusting anything below.")
    else:
        print("  OK: Python RW and R RW agree exactly -- the two pipelines "
              "are aligned on the same window and target series.\n")

# --- sort chronologically (required for recursive ensemble weighting) ---
merged["_q"] = pd.PeriodIndex(merged["Quarter"], freq="Q")
merged = merged.sort_values("_q").reset_index(drop=True)

# --- ensemble: simple average + recursive inverse-MSE ---
BLEND_MODELS = ["VECM", "ARDL", "NARDL", "ARRF", "LSTM", "Chronos"]
WARMUP = 8
actual_chrono = merged[actual_cols[0]].to_numpy()

ens_avg = np.full(n_rows, np.nan)
ens_invmse = np.full(n_rows, np.nan)

for t in range(n_rows):
    preds_t = merged.loc[t, BLEND_MODELS].to_numpy(dtype=float)
    ens_avg[t] = np.nanmean(preds_t)
    if t < WARMUP:
        ens_invmse[t] = ens_avg[t]
        continue
    mses = np.array([
        np.mean((actual_chrono[:t] - merged[m].to_numpy()[:t]) ** 2)
        for m in BLEND_MODELS
    ])
    w = (1 / mses) / np.sum(1 / mses)
    ens_invmse[t] = np.sum(w * preds_t)

merged["Ensemble_avg"] = ens_avg
merged["Ensemble_invmse"] = ens_invmse

actual = merged[actual_cols[0]].to_numpy()

# --- Accuracy table ---
model_names = [n for n in frames if n != "RW_py"] + ['Ensemble_avg', 'Ensemble_invmse']

def report(mask, label):
    print(f"\n=== {label} ({mask.sum()} quarters) ===")
    print(f"{'Model':18s} {'RMSE':>8s} {'MAE':>8s}")
    errs = {}
    for name in model_names:
        e = (actual - merged[name].to_numpy())[mask]
        errs[name] = e
        rmse = np.sqrt(np.mean(e ** 2))
        mae = np.mean(np.abs(e))
        print(f"{name:18s} {rmse:8.4f} {mae:8.4f}")

    if RW_BENCHMARK in errs:
        print(f"\nDM vs {RW_BENCHMARK} (stat>0 => model worse than RW):")
        for name in model_names:
            if name == RW_BENCHMARK:
                continue
            s, pv = dm_test(errs[name], errs[RW_BENCHMARK])
            flag = "  *significant at 5%" if pv < 0.05 else ""
            print(f"  {name:18s} stat={s:+.3f}  p={pv:.3f}{flag}")

    if STRUCTURAL_BEST in errs and ML_BEST in errs:
        s, pv = dm_test(errs[ML_BEST], errs[STRUCTURAL_BEST])
        flag = "  *significant at 5%" if pv < 0.05 else ""
        print(f"\nDM {ML_BEST} vs {STRUCTURAL_BEST}: stat={s:+.3f}  p={pv:.3f}"
              f"  (stat>0 => {ML_BEST} worse){flag}")

    if "Ensemble_avg" in errs and STRUCTURAL_BEST in errs:
        s, pv = dm_test(errs["Ensemble_avg"], errs[STRUCTURAL_BEST])
        flag = "  *significant at 5%" if pv < 0.05 else ""
        print(f"\nDM Ensemble_avg vs {STRUCTURAL_BEST}: stat={s:+.3f}  p={pv:.3f}"
              f"  (stat>0 => ensemble worse){flag}")

    return errs


EXCLUDE_QUARTERS = ["2020Q2", "2020Q3"]
full_mask = np.ones(n_rows, dtype=bool)
excov_mask = ~merged["Quarter"].isin(EXCLUDE_QUARTERS).to_numpy()

errors = report(full_mask, "Full sample")
excov_errors = report(excov_mask, "Excluding COVID (2020Q2-Q3)")

def plot_rmse_bar(errs, benchmark=RW_BENCHMARK, title=""):
    rmse = {n: np.sqrt(np.mean(e ** 2)) for n, e in errs.items()}
    order = sorted(rmse, key=rmse.get)  # ascending, best first

    pvals = {}
    for n in order:
        if n == benchmark:
            continue
        _, pv = dm_test(errs[n], errs[benchmark])
        pvals[n] = pv

    colors = ['#2E7D32' if n.startswith('Ensemble') else '#9E9E9E' for n in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(order, [rmse[n] for n in order], color=colors)

    for bar, n in zip(bars, order):
        h = bar.get_height()
        star = "*" if pvals.get(n, 1) < 0.05 else ""
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.4f}{star}",
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.set_xticklabels(order, rotation=30, ha='right')
    fig.tight_layout()
    fig.savefig(f"data/outputs/figures/{title.replace(' ', '_').lower()}.png", dpi=200)
    plt.show()


plot_rmse_bar(errors, title="Forecast RMSE - Full Sample")
plot_rmse_bar(excov_errors, title="Forecast RMSE - Excluding COVID")