"""
Chronos zero-shot baseline for the housing-starts forecast horse race.
Adds amazon/chronos-t5-small to the existing RW / VECM / ARDL comparison.

Design: expanding-window, 1-step-ahead recursive pseudo-OOS forecasts of
lhstarts. Chronos sees only the past of lhstarts (univariate, zero-shot) so
the comparison is exactly structural-model-with-covariates vs pure pattern
matcher. Metrics in log units to match the existing RMSEs (RW .437 / VECM
.371 / ARDL .339). Diebold-Mariano (HLN-corrected) against VECM.

    pip install chronos-forecasting scipy --break-system-packages

ALIGNMENT CHECK: tune N_TEST / H until the printed RW RMSE reproduces your
0.437. Once it matches, the harness is aligned with your R horse race and the
Chronos / VECM numbers are directly comparable. Feed your R VECM forecasts via
VECM_CSV (one row per holdout quarter, same dates) for the DM test.

NOTE: I ran this on KAGGLE

"""
import os
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from scipy.stats import t as tdist

# --- config ---
MASTER_DIR = "../../data/python_master"
CSV         = f"{MASTER_DIR}/england_master.csv"
TARGET      = "starts"            # lhstarts = log(TARGET)
H           = 1                   # forecast horizon (recursive)
NUM_SAMPLES = 100
MODEL       = "amazon/chronos-t5-small"
VECM_CSV    = None               # e.g. "vecm_fcst.csv" with column lhstarts_hat

# --- data ---
df = pd.read_csv(CSV)
# Parse Quarter labels before dropping rows, so they stay row-aligned with df
df['Quarter'] = pd.PeriodIndex(df['Unnamed: 0'].astype(str).str.replace(" ", ""), freq='Q')
# Dropping 2026 Q1
df = df[df['Quarter'] != pd.Period('2026Q1', freq='Q')].reset_index(drop=True)
y = np.log(df[TARGET].to_numpy(dtype=float))
n = len(y)
eval_start = (2010 - 1975) * 4    # = 140, 0-indexed row of 2010Q1
test_idx   = np.arange(eval_start, n)   # targets 2010Q1 .. last quarter
N_TEST     = len(test_idx)              # ~64 for n=204, not 20
assert eval_start >= 40, "not enough training data before 2010Q1"
# Verify the hardcoded row arithmetic actually lands on 2010Q1 -- don't just
# trust it because the RW RMSE happens to look right.
assert df['Quarter'].iloc[eval_start] == pd.Period('2010Q1', freq='Q'), (
    f"eval_start row is {df['Quarter'].iloc[eval_start]}, expected 2010Q1 "
    "-- check the CSV's date range / row count before trusting any RMSEs below"
)
assert df['Quarter'].iloc[-1] == pd.Period('2025Q4', freq='Q'), (
    f"last row is {df['Quarter'].iloc[-1]}, expected 2025Q4 -- test window "
    "won't match 05_ARRF.py / 06_LSTM.py, which both cap at 2025Q4"
)

# --- chronos: expanding window, h-step recursive ---
# SCHEME: zero-shot, no parameters are ever fit -- the model is frozen and
# pretrained. Only the context window (true past lhstarts up to t-1) expands
# at each origin, which is the correct way to use a frozen model and isn't
# directly comparable to "recursive" (ARRF) or "fixed" (LSTM) re-estimation
# schemes, since there's no estimation step here at all.
pipe = ChronosPipeline.from_pretrained(MODEL, device_map="cpu",
                                       torch_dtype=torch.float32)

chronos_pred = np.empty(N_TEST)
rw_pred      = np.empty(N_TEST)
for i, t in enumerate(test_idx):
    ctx = torch.tensor(y[: t - H + 1], dtype=torch.float32)   # data up to origin
    fc = pipe.predict(ctx, prediction_length=H, num_samples=NUM_SAMPLES)  # [1,S,H]
    chronos_pred[i] = np.median(fc[0].numpy(), axis=0)[H - 1]  # point = sample median
    rw_pred[i]      = y[t - H]                                 # naive last-value anchor

actual = y[test_idx]


# --- metrics & DM ---
def metrics(a, p):
    e = a - p
    return np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e))


def dm_test(e1, e2, h=H, power=2):
    """H0: equal accuracy. Loss = |e|^power. stat>0 => model-1 worse.
    HLN small-sample correction; compared to t(n-1)."""
    d = np.abs(e1) ** power - np.abs(e2) ** power
    nn = len(d)
    dbar = d.mean()
    var = np.sum((d - dbar) ** 2) / nn                  # gamma0
    for lag in range(1, h):                             # HAC up to h-1 (none for h=1)
        var += 2 * np.sum((d[lag:] - dbar) * (d[:-lag] - dbar)) / nn
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var / nn)
    stat *= np.sqrt((nn + 1 - 2 * h + h * (h - 1) / nn) / nn)
    return stat, 2 * tdist.cdf(-abs(stat), df=nn - 1)


print(f"holdout: last {N_TEST} quarters, h={H}, n_train_start={n - N_TEST}\n")
for name, p in [("naive RW", rw_pred), ("Chronos", chronos_pred)]:
    rmse, mae = metrics(actual, p)
    print(f"{name:10s}  RMSE={rmse:.4f}  MAE={mae:.4f}")

s, pv = dm_test(actual - chronos_pred, actual - rw_pred)
print(f"\nDM Chronos vs RW   stat={s:+.3f}  p={pv:.3f}")

if VECM_CSV:
    vhat = pd.read_csv(VECM_CSV)["lhstarts_hat"].to_numpy(dtype=float)[-N_TEST:]
    rmse, mae = metrics(actual, vhat)
    print(f"\nVECM       RMSE={rmse:.4f}  MAE={mae:.4f}  (from {VECM_CSV})")
    s, pv = dm_test(actual - chronos_pred, actual - vhat)
    print(f"DM Chronos vs VECM stat={s:+.3f}  p={pv:.3f}   (stat>0 => Chronos worse)")

# --- save for merge with the R forecasts and the other Python models ----------
test_quarters = df['Quarter'].iloc[test_idx].astype(str).to_numpy()
out = pd.DataFrame({
    "Quarter": test_quarters,
    "chronos": chronos_pred,
    "rw": rw_pred,
    "actual": actual,
})
os.makedirs("../data/outputs/forecasts", exist_ok=True)
out.to_csv("../data/outputs/forecasts/chronos_forecasts.csv", index=False)
print("\nSaved chronos_forecasts.csv")
