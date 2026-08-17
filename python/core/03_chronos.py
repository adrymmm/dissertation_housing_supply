"""
Chronos-Bolt zero-shot baseline for the housing-starts forecast horse race.

Design: expanding-window, 1-step-ahead pseudo-OOS forecasts of lhstarts.
Chronos sees only the past of lhstarts (univariate, zero-shot), so the
comparison is exactly structural-model-with-covariates vs pure pattern
matcher.  Forecasts are produced in log units to match the rest of the race.

This script does NOT score the model.  It emits chronos_forecasts.csv and
06_aggr_forecast.py does the ranking and the testing (MCS / SPA over the whole
model set).  The RMSE/MAE printed below are a run-time sanity check only --
the authoritative numbers come from the aggregator, which applies the COVID
exclusion mask and the ensemble warmup.  Pairwise Diebold-Mariano was dropped
from the project in favour of MCS/SPA, which correct for multiple testing
across the ~20 models in the panel.

    pip install chronos-forecasting

Model note: amazon/chronos-bolt-base is a direct multi-quantile regressor, not
the sampling/tokenising t5 architecture used earlier.  Three consequences that
matter here: it is deterministic (no sample median, so no seed-dependent
RMSE), its output is not confined to a token-bin lattice (the t5 forward path
collapsed onto three distinct values across 13 quarters), and it is roughly
two orders of magnitude faster -- the whole backtest is a few batched forward
passes and runs on CPU in seconds, so this no longer needs Kaggle.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

# --- config ---
FORECASTS_DIR = ROOT / "data" / "outputs" / "forecasts"
CSV         = ROOT / "data" / "python_master" / "england_master.csv"
TARGET      = "starts"            # lhstarts = log(TARGET)
H           = 1                   # forecast horizon
H_FWD       = 13                  # forward path, 2026Q1 .. 2029Q1
MODEL       = "amazon/chronos-bolt-base"
BATCH       = 32                  # origins per forward pass
POINT       = "median"            # point forecast: "median" or "mean"

# Bolt emits quantiles at these levels
Q_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# No direct mean, manual calculation
Q_MED, Q_LO, Q_HI = Q_LEVELS.index(0.5), Q_LEVELS.index(0.1), Q_LEVELS.index(0.9)

# --- data ---
df = pd.read_csv(CSV)
# Parse Quarter labels before dropping rows, so they stay row-aligned with df
df['Quarter'] = pd.PeriodIndex(df['Unnamed: 0'].astype(str).str.replace(" ", ""), freq='Q')
# Dropping 2026 Q1
df = df[df['Quarter'] != pd.Period('2026Q1', freq='Q')].reset_index(drop=True)
y = np.log(df[TARGET].to_numpy(dtype=float))

# Evaluation window -- match the R horse race (2010Q1-2025Q4) so the merge in
# 06_aggr_forecast.py is an inner join over a full set of quarters.  Located by
# Period lookup rather than row arithmetic, as in 04_ARRF.py.
start, end = pd.Period('2010Q1', freq='Q'), pd.Period('2025Q4', freq='Q')
test_idx = np.where((df['Quarter'] >= start) & (df['Quarter'] <= end))[0]
N_TEST   = len(test_idx)
assert N_TEST > 0, "no rows in the 2010Q1-2025Q4 evaluation window"
assert test_idx[0] >= 40, "not enough training data before 2010Q1"
assert df['Quarter'].iloc[-1] == end, (
    f"last row is {df['Quarter'].iloc[-1]}, expected {end} -- the test window "
    "won't match 04_ARRF.py / 05_LSTM.py, which both cap at 2025Q4"
)

# --- chronos: expanding window ---
# SCHEME: zero-shot, no parameters are ever fit -- the model is frozen and
# pretrained. Only the context window (true past lhstarts up to t-1) expands
# at each origin, which is the correct way to use a frozen model and isn't
# directly comparable to "recursive" (ARRF) or "fixed" (LSTM) re-estimation
# schemes, since there's no estimation step here at all.
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map=device, dtype=dtype)
torch.manual_seed(42)   # Bolt is deterministic; set for good order, not needed

# Every origin is an independent context, so all N_TEST of them batch into a
# handful of forward passes instead of N_TEST sequential calls.
contexts = [torch.tensor(y[: t - H + 1], dtype=torch.float32) for t in test_idx]
chunks = []
for i in range(0, len(contexts), BATCH):
    q, _ = pipe.predict_quantiles(contexts[i:i + BATCH],
                                  prediction_length=H,
                                  quantile_levels=Q_LEVELS)
    chunks.append(q.float().cpu().numpy())          # (b, H, len(Q_LEVELS))
qs = np.concatenate(chunks, axis=0)[:, H - 1, :]    # (N_TEST, len(Q_LEVELS))

chronos_median = qs[:, Q_MED]
chronos_mean   = qs.mean(axis=1)             # tail-truncated quantile average
chronos_pred   = chronos_median if POINT == "median" else chronos_mean
rw_pred        = y[test_idx - H]             # naive last-value anchor
actual         = y[test_idx]


def rmse_mae(a, p):
    e = a - p
    return np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e))


print(f"model: {MODEL} on {device}")
print(f"holdout: {df['Quarter'].iloc[test_idx[0]]}-{end} "
      f"({N_TEST} quarters), h={H}, point={POINT}\n")
# Sanity check only -- 06_aggr_forecast.py produces the reported figures.
for name, p in [("naive RW", rw_pred), ("Chronos", chronos_pred)]:
    r, m = rmse_mae(actual, p)
    print(f"{name:10s}  RMSE={r:.4f}  MAE={m:.4f}")

# Point-summary sensitivity: RMSE is minimised by the conditional mean and MAE
# by the median, so the choice is not neutral on an RMSE scoreboard.  Both are
# written to the CSV; POINT decides which one the aggregator reads.
r_med, m_med = rmse_mae(actual, chronos_median)
r_avg, m_avg = rmse_mae(actual, chronos_mean)
print(f"\npoint summary   median RMSE={r_med:.4f} MAE={m_med:.4f}"
      f"   |   quantile-avg RMSE={r_avg:.4f} MAE={m_avg:.4f}")

# --- save for merge with the R forecasts and the other Python models ----------
out = pd.DataFrame({
    "Quarter": df['Quarter'].iloc[test_idx].astype(str).to_numpy(),
    "chronos": chronos_pred,
    "chronos_median": chronos_median,
    "chronos_mean": chronos_mean,
    "rw": rw_pred,          # load-bearing: 06_aggr_forecast.py asserts this
    "actual": actual,       # matches R's RW column, aligning the two pipelines
})
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
out.to_csv(FORECASTS_DIR / "chronos_forecasts.csv", index=False)
print(f"\nSaved chronos_forecasts.csv -> {FORECASTS_DIR}")

# --- forward path, consumed by 09_net_additions.ipynb -------------------------
# Levels come from exponentiating the log quantiles: quantiles are equivariant
# under a monotone transform, so exp(q_a) is the level a-quantile.  The point
# path stays the median regardless of POINT -- exp() of a log mean would be a
# median-of-levels claim dressed as a mean, and the bridge re-logs this column
# anyway, so the median keeps that round trip exact.
qf, _ = pipe.predict_quantiles([torch.tensor(y, dtype=torch.float32)],
                               prediction_length=H_FWD,
                               quantile_levels=Q_LEVELS)
qf = qf[0].float().cpu().numpy()                # (H_FWD, len(Q_LEVELS))

quarters = pd.period_range('2026Q1', periods=H_FWD, freq='Q')
fwd = pd.DataFrame({"Quarter": quarters.astype(str),
                    "starts": np.exp(qf[:, Q_MED]),
                    "lo": np.exp(qf[:, Q_LO]),
                    "hi": np.exp(qf[:, Q_HI])})
fwd.to_csv(FORECASTS_DIR / "chronos_forward.csv", index=False)
print(f"Saved chronos_forward.csv   ({fwd['starts'].nunique()} distinct "
      f"values across {H_FWD} quarters)")
