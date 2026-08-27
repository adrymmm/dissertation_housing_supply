import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import breaks_cusumolsresid

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from python.functions.bridge import parse_quarter, AR_ORDER, STARTS_LAG, DUMMY_COLS

OUT = ROOT / "data/processed/completions_bridge.pkl"

FORMULA = "ln_C ~ " + " + ".join(
    [f"ln_C_lag{L}" for L in range(1, AR_ORDER + 1)]
    + [f"ln_S_lag{STARTS_LAG}"]
    + ["q2", "q3", "q4"]
    + DUMMY_COLS
)

#  load 
raw = pd.read_excel(
    ROOT / "data/raw/starts/indicatorsofukhousebuilding.xlsx", sheet_name="1b", header=5
)
df = raw.rename(
    columns={
        "Period": "period",
        "Started - Private Enterprise": "starts_private",
        "Completed - Private Enterprise": "completions_private",
    }
)[["period", "starts_private", "completions_private"]].copy()

df["starts_private"] = pd.to_numeric(df["starts_private"], errors="coerce")
df["completions_private"] = pd.to_numeric(df["completions_private"], errors="coerce")

df["quarter"] = df["period"].apply(parse_quarter)
df = df.dropna(subset=["quarter"]).set_index("quarter").sort_index().drop(columns="period")

df["ln_C"] = np.log(df["completions_private"])
df["ln_S"] = np.log(df["starts_private"])
for L in range(1, AR_ORDER + 1):
    df[f"ln_C_lag{L}"] = df["ln_C"].shift(L)
df[f"ln_S_lag{STARTS_LAG}"] = df["ln_S"].shift(STARTS_LAG)

for q in [2, 3, 4]:
    df[f"q{q}"] = (df.index.quarter == q).astype(int)
for col, per in [("d20Q2", "2020Q2"), ("d20Q3", "2020Q3"), ("d20Q4", "2020Q4")]:
    df[col] = (df.index == per).astype(int)
df["gfc_2008_2009"] = df.index.isin(pd.period_range("2008Q3", "2009Q4", freq="Q")).astype(int)

#  fit 
hist = df.dropna()
fit = smf.ols(FORMULA, data=hist).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
smearing = float(np.mean(np.exp(fit.resid)))  # Duan retransformation correction

print(fit.summary())

rho_sum = sum(fit.params[f"ln_C_lag{L}"] for L in range(1, AR_ORDER + 1))
long_run_pass_through = fit.params[f"ln_S_lag{STARTS_LAG}"] / (1 - rho_sum)
print(f"\nLong-run pass-through:  {long_run_pass_through:.3f}")
print(f"Smearing factor:        {smearing:.4f}")

stat, pval, _ = breaks_cusumolsresid(fit.resid)
print(f"OLS-CUSUM:              stat={stat:.3f}, p={pval:.3f}")

#  save -
joblib.dump(
    {"model": fit, "smearing": smearing, "hist_ln_C": hist["ln_C"], "hist_ln_S": hist["ln_S"]},
    OUT,
)
print(f"\nSaved -> {OUT}")