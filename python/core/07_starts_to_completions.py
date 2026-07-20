import sys
sys.path.append("../..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import json
from python.functions.bridge import parse_quarter


# Define the two data file paths
housebuilding_file = "data/raw/starts/indicatorsofukhousebuilding.xlsx"

# Read sheet 1b, which contains England quarterly housebuilding data
raw = pd.read_excel(
    housebuilding_file,
    sheet_name="1b",
    header=5
)

# Display the first few rows
raw.head()

# Rename the key columns for easier use
df = raw.rename(columns={
    "Period": "period",
    "Started - Private Enterprise": "starts_private",
    "Completed - Private Enterprise": "completions_private"
})

# Keep only the variables needed for the analysis
df = df[[
    "period",
    "starts_private",
    "completions_private"
]].copy()

# Convert numerical columns to numeric format
df["starts_private"] = pd.to_numeric(df["starts_private"], errors="coerce")
df["completions_private"] = pd.to_numeric(df["completions_private"], errors="coerce")

# Apply the quarter parser
df["quarter"] = df["period"].apply(parse_quarter)

# Drop rows without valid quarters and set quarter as the index
df = df.dropna(subset=["quarter"])
df = df.set_index("quarter").sort_index()
df = df.drop(columns="period")

# Display the cleaned data  
print(df.head())

df["ln_C"] = np.log(df["completions_private"])
df["ln_S"] = np.log(df["starts_private"])
df["ln_C_lag1"] = df["ln_C"].shift(1)
df["ln_S_lag4"] = df["ln_S"].shift(4)
df["quarter_num"] = df.index.quarter
# Impulse dummies
df["d08Q3"] = (df.index == "2008Q3").astype(int)
df["d20Q2"] = (df.index == "2020Q2").astype(int)
df["d20Q3"] = (df.index == "2020Q3").astype(int)
df.dropna()

model = smf.ols(
    "ln_C ~ ln_C_lag1 + ln_S_lag4 + C(quarter_num) + d08Q3 + d20Q2 + d20Q3",
    data=df
).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

print(model.summary())

rho = model.params["ln_C_lag1"]
beta = model.params["ln_S_lag4"]
print(f"Long-run pass-through: {beta / (1 - rho):.3f}")

stat, pval, crit = breaks_cusumolsresid(model.resid)
print(f"CUSUM statistic: {stat:.3f}, p-value: {pval:.3f}")

# Saving fitted coefficients
bridge_params = {
    "intercept": model.params["Intercept"],
    "rho": model.params["ln_C_lag1"],
    "beta": model.params["ln_S_lag4"],
    "q2": model.params["C(quarter_num)[T.2]"],
    "q3": model.params["C(quarter_num)[T.3]"],
    "q4": model.params["C(quarter_num)[T.4]"],
    "d08Q3": model.params["d08Q3"],
    "d20Q2": model.params["d20Q2"],
    "d20Q3": model.params["d20Q3"],
    "smearing_factor": float(np.mean(np.exp(model.resid))),
    "last_ln_C": float(df["ln_C"].iloc[-1])  # seed value for recursive forecast
}

with open("data/processed/bridge_params.json", "w") as f:
    json.dump(bridge_params, f, indent=2)