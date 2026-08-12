#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path

# Set the folder where the data files are stored
DATA_DIR = Path(r"D:\data")

# Define the two data file paths
housebuilding_file = DATA_DIR / "indicatorsofukhousebuilding.xlsx"
stock_file = DATA_DIR / "LiveTable104.ods"

# Print the file paths to check whether they are correct
print(housebuilding_file)
print(stock_file)

# Check whether the files exist in the folder
print("Housebuilding file exists:", housebuilding_file.exists())
print("Stock file exists:", stock_file.exists())


# In[82]:


# Load the Excel file and display all sheet names
xls = pd.ExcelFile(housebuilding_file)
xls.sheet_names


# In[83]:


# Read sheet 1b, which contains England quarterly housebuilding data
raw = pd.read_excel(
    housebuilding_file,
    sheet_name="1b",
    header=5
)

# Display the first few rows
raw.head()


# In[84]:


df = raw.rename(columns={
    "Period": "period",
    "Started - All Dwellings": "starts_all",
    "Completed - All Dwellings": "completions_all",
    "Started - Private Enterprise": "starts_private",
    "Completed - Private Enterprise": "completions_private"
})

df = df[[
    "period",
    "starts_all",
    "completions_all",
    "starts_private",
    "completions_private"
]].copy()

for col in [
    "starts_all",
    "completions_all",
    "starts_private",
    "completions_private"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# In[85]:


def parse_quarter(x):
    x = str(x)

    if len(x) < 4:
        return pd.NaT

    try:
        year = int(x[-4:])
    except:
        return pd.NaT

    if x.startswith("Jan - Mar"):
        q = 1
    elif x.startswith("Apr - Jun"):
        q = 2
    elif x.startswith("Jul - Sep"):
        q = 3
    elif x.startswith("Oct - Dec"):
        q = 4
    else:
        return pd.NaT

    return pd.Period(
        year=year,
        quarter=q,
        freq="Q"
    )


df["quarter"] = df["period"].apply(parse_quarter)

df = df.dropna(subset=["quarter"])
df = df.set_index("quarter").sort_index()

print(df.head())
print(df.tail())
print(df.index.min(), df.index.max())


# In[86]:


print(df.columns)


# In[87]:


# Check the start and end quarters of the dataset
print(df.index.min())
print(df.index.max())

# Display the last few rows
df.tail()


# In[88]:


# Convert the quarterly index to timestamp format for plotting
plot_df = df.copy()
plot_df["date"] = plot_df.index.to_timestamp()

# Plot starts and completions over time
plt.figure(figsize=(12, 6))
plt.plot(plot_df["date"], plot_df["starts_all"], label="Starts")
plt.plot(plot_df["date"], plot_df["completions_all"], label="Completions")
plt.title("Quarterly Housing Starts and Completions in England")
plt.xlabel("Year")
plt.ylabel("Dwellings")
plt.legend()
plt.tight_layout()
plt.show()


# In[89]:


import matplotlib.pyplot as plt

try:
    OUTPUT_DIR
except NameError:
    OUTPUT_DIR = DATA_DIR / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 5.5))
ax.axis("off")

formula_text = r"""
Net Supply Proxy

Official long-run quarterly net additions components are not available on a consistent basis.
Therefore, annual changes in the dwelling stock are used as a long-run proxy for net supply.

$ApproxNetAdditions_y = DwellingStock_y - DwellingStock_{y-1}$

Definitions:

$DwellingStock_y$ = total dwelling stock in year $y$

$DwellingStock_{y-1}$ = total dwelling stock in the previous year

$ApproxNetAdditions_y$ = annual change in dwelling stock

Important note:

This is not the same as official net additions, because it does not separately identify
new build completions, conversions, changes of use, other gains, or demolitions.
It is only used as a long-run proxy linking completions to housing stock changes.
"""

ax.text(0.02, 0.98, formula_text, fontsize=16, va="top", ha="left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "formula_2_net_supply_proxy.png", dpi=300, bbox_inches="tight")
plt.show()


# In[90]:


# Load LiveTable104 dwelling stock data
stock_raw = pd.read_excel(
    stock_file,
    sheet_name="LT_104",
    header=4,
    engine="odf"
)

stock_raw.head()


# In[91]:


# Keep year and all dwellings columns
stock = stock_raw[["Year", "All dwellings"]].copy()

# Rename columns
stock = stock.rename(columns={
    "Year": "year",
    "All dwellings": "dwelling_stock"
})

# Convert columns to numeric format
stock["year"] = pd.to_numeric(stock["year"], errors="coerce")
stock["dwelling_stock"] = pd.to_numeric(stock["dwelling_stock"], errors="coerce")

# Keep observations from 1978 onwards
stock = stock.dropna(subset=["year", "dwelling_stock"])
stock = stock[stock["year"] >= 1978].copy()

# Calculate approximate net additions as annual change in dwelling stock
stock["approx_net_additions"] = stock["dwelling_stock"].diff()

stock.head()


# In[92]:


stock.tail()


# In[93]:


# Plot dwelling stock over time
plt.figure(figsize=(12, 6))
plt.plot(stock["year"], stock["dwelling_stock"])
plt.title("Dwelling Stock in England")
plt.xlabel("Year")
plt.ylabel("Number of dwellings")
plt.tight_layout()
plt.show()


# In[94]:


# Plot approximate net additions using annual change in dwelling stock
plt.figure(figsize=(12, 6))
plt.plot(stock["year"], stock["approx_net_additions"])
plt.title("Approximate Net Additions Using Change in Dwelling Stock")
plt.xlabel("Year")
plt.ylabel("Annual change in dwelling stock")
plt.tight_layout()
plt.show()


# In[95]:


# Save dwelling stock figure
plt.figure(figsize=(12, 6))
plt.plot(stock["year"], stock["dwelling_stock"])
plt.title("Dwelling Stock in England")
plt.xlabel("Year")
plt.ylabel("Number of dwellings")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dwelling_stock_england.png", dpi=300)
plt.show()


# In[96]:


# Save approximate net additions figure
plt.figure(figsize=(12, 6))
plt.plot(stock["year"], stock["approx_net_additions"])
plt.title("Approximate Net Additions Using Change in Dwelling Stock")
plt.xlabel("Year")
plt.ylabel("Annual change in dwelling stock")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "approx_net_additions_stock_change.png", dpi=300)
plt.show()


# In[97]:


# Convert quarterly completions into annual completions
annual_completions = df.copy()

annual_completions["year"] = annual_completions.index.year

annual_completions = annual_completions.groupby("year", as_index=False).agg({
    "completions_all": "sum",
    "starts_all": "sum"
})

annual_completions = annual_completions.rename(columns={
    "completions_all": "annual_completions",
    "starts_all": "annual_starts"
})

annual_completions.head()


# In[98]:


# Merge annual completions with dwelling stock change proxy
annual_bridge = pd.merge(
    annual_completions,
    stock[["year", "dwelling_stock", "approx_net_additions"]],
    on="year",
    how="inner"
)

annual_bridge.head()


# In[99]:


# Plot annual completions and approximate net additions
plt.figure(figsize=(12, 6))
plt.plot(annual_bridge["year"], annual_bridge["annual_completions"], label="Annual completions")
plt.plot(annual_bridge["year"], annual_bridge["approx_net_additions"], label="Approximate net additions")
plt.title("Annual Completions and Approximate Net Additions in England")
plt.xlabel("Year")
plt.ylabel("Dwellings")
plt.legend()
plt.tight_layout()
plt.show()


# In[100]:


# Save annual bridge figure
plt.figure(figsize=(12, 6))
plt.plot(annual_bridge["year"], annual_bridge["annual_completions"], label="Annual completions")
plt.plot(annual_bridge["year"], annual_bridge["approx_net_additions"], label="Approximate net additions")
plt.title("Annual Completions and Approximate Net Additions in England")
plt.xlabel("Year")
plt.ylabel("Dwellings")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "annual_completions_vs_net_additions_proxy.png", dpi=300)
plt.show()


# In[101]:


# ============================================================
# Save Net Supply Proxy Datasets
# ============================================================

stock.to_excel(
    OUTPUT_DIR / "dwelling_stock_proxy_data.xlsx",
    index=False
)

annual_bridge.to_excel(
    OUTPUT_DIR / "annual_completions_net_supply_bridge.xlsx",
    index=False
)

print("Net supply proxy datasets saved successfully.")


# In[102]:


# ============================================================
# Completion Lag Selection - Common Sample
# Preferred Shock Specification
#
# Compare 1-4 completion lags using the SAME observations
# Private starts lag fixed at 1 quarter
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from statsmodels.stats.diagnostic import (
    acorr_ljungbox
)


# ============================================================
# 1. Create modelling dataset
# ============================================================

completion_df = df.copy()

completion_df = completion_df[
    (completion_df["starts_private"] > 0) &
    (completion_df["completions_private"] > 0)
].copy()


# ============================================================
# 2. Log transformations
# ============================================================

completion_df["ln_C_private"] = np.log(
    completion_df["completions_private"]
)

completion_df["ln_S_private"] = np.log(
    completion_df["starts_private"]
)


# ============================================================
# 3. Completion lags 1-4
# ============================================================

for lag in range(1, 5):

    completion_df[f"ln_C_private_lag{lag}"] = (
        completion_df["ln_C_private"].shift(lag)
    )


# ============================================================
# 4. Private starts lag = 1 quarter
# ============================================================

completion_df["ln_S_private_lag1"] = (
    completion_df["ln_S_private"].shift(1)
)


# ============================================================
# 5. Quarterly seasonality
# ============================================================

completion_df["quarter_num"] = (
    completion_df.index.quarter
)


# ============================================================
# 6. Preferred historical shock controls
# ============================================================

completion_df["gfc_2008_2009"] = (
    (completion_df.index >= pd.Period("2008Q3", freq="Q")) &
    (completion_df.index <= pd.Period("2009Q4", freq="Q"))
).astype(int)

completion_df["covid_2020Q2"] = (
    completion_df.index == pd.Period("2020Q2", freq="Q")
).astype(int)

completion_df["covid_2020Q3"] = (
    completion_df.index == pd.Period("2020Q3", freq="Q")
).astype(int)

completion_df["covid_2020Q4"] = (
    completion_df.index == pd.Period("2020Q4", freq="Q")
).astype(int)


# ============================================================
# 7. Common sample
# ============================================================

required_cols = [
    "ln_C_private",
    "ln_C_private_lag1",
    "ln_C_private_lag2",
    "ln_C_private_lag3",
    "ln_C_private_lag4",
    "ln_S_private_lag1",
    "quarter_num",
    "gfc_2008_2009",
    "covid_2020Q2",
    "covid_2020Q3",
    "covid_2020Q4"
]

completion_df = completion_df.dropna(
    subset=required_cols
).copy()


print("==========================================")
print("COMMON SAMPLE CREATED")
print("==========================================")

print("Observations:", len(completion_df))

print(
    "Sample period:",
    completion_df.index.min(),
    "to",
    completion_df.index.max()
)


# ============================================================
# 8. Compare completion lag specifications 1-4
# ============================================================

completion_lag_results_common = []

for p in range(1, 5):

    completion_terms = " + ".join(
        [
            f"ln_C_private_lag{lag}"
            for lag in range(1, p + 1)
        ]
    )

    formula = f"""
    ln_C_private ~ {completion_terms}
    + ln_S_private_lag1
    + C(quarter_num)
    + gfc_2008_2009
    + covid_2020Q2
    + covid_2020Q3
    + covid_2020Q4
    """

    model_p = smf.ols(
        formula,
        data=completion_df
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 4}
    )

    lb = acorr_ljungbox(
        model_p.resid,
        lags=[4, 8, 12],
        return_df=True
    )

    completion_lag_results_common.append({

        "Completion_Lags": p,

        "Starts_Coefficient":
            model_p.params["ln_S_private_lag1"],

        "Starts_P_value":
            model_p.pvalues["ln_S_private_lag1"],

        "Adj_R_squared":
            model_p.rsquared_adj,

        "AIC":
            model_p.aic,

        "BIC":
            model_p.bic,

        "LB_pvalue_4":
            lb.loc[4, "lb_pvalue"],

        "LB_pvalue_8":
            lb.loc[8, "lb_pvalue"],

        "LB_pvalue_12":
            lb.loc[12, "lb_pvalue"]
    })


# ============================================================
# 9. Results table
# ============================================================

completion_lag_results_common = pd.DataFrame(
    completion_lag_results_common
)

print("\n==========================================")
print("COMPLETION LAG SELECTION RESULTS")
print("==========================================")

print(
    completion_lag_results_common
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 10. Select preferred specification
# ============================================================

best_completion_adj_r2 = (
    completion_lag_results_common.loc[
        completion_lag_results_common[
            "Adj_R_squared"
        ].idxmax()
    ]
)

best_completion_aic = (
    completion_lag_results_common.loc[
        completion_lag_results_common[
            "AIC"
        ].idxmin()
    ]
)

best_completion_bic = (
    completion_lag_results_common.loc[
        completion_lag_results_common[
            "BIC"
        ].idxmin()
    ]
)


print("\n==========================================")
print("SELECTED COMPLETION-LAG SPECIFICATION")
print("==========================================")

print(
    "Completion lags:",
    int(best_completion_adj_r2["Completion_Lags"])
)

print(
    "Adjusted R-squared:",
    best_completion_adj_r2["Adj_R_squared"]
)

print(
    "AIC:",
    best_completion_adj_r2["AIC"]
)

print(
    "BIC:",
    best_completion_adj_r2["BIC"]
)

print(
    "Ljung-Box p-value (lag 4):",
    best_completion_adj_r2["LB_pvalue_4"]
)

print(
    "Ljung-Box p-value (lag 8):",
    best_completion_adj_r2["LB_pvalue_8"]
)

print(
    "Ljung-Box p-value (lag 12):",
    best_completion_adj_r2["LB_pvalue_12"]
)


# In[103]:


# ============================================================
# Robustness Check:
# Original Shock Controls vs Group Impulse Dummies
# Final dynamic structure:
# Starts lag = 1
# Completion lags = 1-4
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import acorr_ljungbox


# ============================================================
# 1. Create common comparison dataset
# ============================================================

compare_df = df.copy()

compare_df = compare_df[
    (compare_df["starts_private"] > 0) &
    (compare_df["completions_private"] > 0)
].copy()

compare_df["ln_C_private"] = np.log(
    compare_df["completions_private"]
)

compare_df["ln_S_private"] = np.log(
    compare_df["starts_private"]
)

# Completion lags 1-4
for lag in range(1, 5):
    compare_df[f"ln_C_private_lag{lag}"] = (
        compare_df["ln_C_private"].shift(lag)
    )

# Starts lag 1
compare_df["ln_S_private_lag1"] = (
    compare_df["ln_S_private"].shift(1)
)

# Seasonality
compare_df["quarter_num"] = compare_df.index.quarter


# ============================================================
# 2. Original shock controls
# ============================================================

compare_df["gfc_2008_2009"] = (
    (compare_df.index >= pd.Period("2008Q3", freq="Q")) &
    (compare_df.index <= pd.Period("2009Q4", freq="Q"))
).astype(int)

compare_df["covid_2020Q2"] = (
    compare_df.index == pd.Period("2020Q2", freq="Q")
).astype(int)

compare_df["covid_2020Q3"] = (
    compare_df.index == pd.Period("2020Q3", freq="Q")
).astype(int)

compare_df["covid_2020Q4"] = (
    compare_df.index == pd.Period("2020Q4", freq="Q")
).astype(int)


# ============================================================
# 3. Group impulse dummies
# ============================================================

compare_df["d08Q3"] = (
    compare_df.index == pd.Period("2008Q3", freq="Q")
).astype(int)

compare_df["d20Q2"] = (
    compare_df.index == pd.Period("2020Q2", freq="Q")
).astype(int)

compare_df["d20Q3"] = (
    compare_df.index == pd.Period("2020Q3", freq="Q")
).astype(int)

compare_df["d23Q2"] = (
    compare_df.index == pd.Period("2023Q2", freq="Q")
).astype(int)

compare_df["d23Q3"] = (
    compare_df.index == pd.Period("2023Q3", freq="Q")
).astype(int)


# ============================================================
# 4. Use exactly the same sample
# ============================================================

required_cols = [
    "ln_C_private",
    "ln_C_private_lag1",
    "ln_C_private_lag2",
    "ln_C_private_lag3",
    "ln_C_private_lag4",
    "ln_S_private_lag1",
    "quarter_num"
]

compare_df = compare_df.dropna(
    subset=required_cols
).copy()


# ============================================================
# 5. Original preferred specification
# ============================================================

original_formula = """
ln_C_private ~ ln_C_private_lag1
+ ln_C_private_lag2
+ ln_C_private_lag3
+ ln_C_private_lag4
+ ln_S_private_lag1
+ C(quarter_num)
+ gfc_2008_2009
+ covid_2020Q2
+ covid_2020Q3
+ covid_2020Q4
"""

original_model = smf.ols(
    original_formula,
    data=compare_df
).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 4}
)


# ============================================================
# 6. Group impulse-dummy specification
# ============================================================

group_formula = """
ln_C_private ~ ln_C_private_lag1
+ ln_C_private_lag2
+ ln_C_private_lag3
+ ln_C_private_lag4
+ ln_S_private_lag1
+ C(quarter_num)
+ d08Q3
+ d20Q2
+ d20Q3
+ d23Q2
+ d23Q3
"""

group_model = smf.ols(
    group_formula,
    data=compare_df
).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 4}
)


# ============================================================
# 7. Ljung-Box diagnostics
# ============================================================

lb_original = acorr_ljungbox(
    original_model.resid,
    lags=[4, 8, 12],
    return_df=True
)

lb_group = acorr_ljungbox(
    group_model.resid,
    lags=[4, 8, 12],
    return_df=True
)


# ============================================================
# 8. Comparison table
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Original shock controls",
        "Group impulse dummies"
    ],

    "Observations": [
        int(original_model.nobs),
        int(group_model.nobs)
    ],

    "Adj_R_squared": [
        original_model.rsquared_adj,
        group_model.rsquared_adj
    ],

    "AIC": [
        original_model.aic,
        group_model.aic
    ],

    "BIC": [
        original_model.bic,
        group_model.bic
    ],

    "Starts_Coefficient": [
        original_model.params["ln_S_private_lag1"],
        group_model.params["ln_S_private_lag1"]
    ],

    "Starts_P_value": [
        original_model.pvalues["ln_S_private_lag1"],
        group_model.pvalues["ln_S_private_lag1"]
    ],

    "LB_pvalue_4": [
        lb_original.loc[4, "lb_pvalue"],
        lb_group.loc[4, "lb_pvalue"]
    ],

    "LB_pvalue_8": [
        lb_original.loc[8, "lb_pvalue"],
        lb_group.loc[8, "lb_pvalue"]
    ],

    "LB_pvalue_12": [
        lb_original.loc[12, "lb_pvalue"],
        lb_group.loc[12, "lb_pvalue"]
    ]
})


print("==========================================")
print("SHOCK SPECIFICATION ROBUSTNESS CHECK")
print("==========================================")

print(
    comparison.to_string(index=False)
)


# In[104]:


# ============================================================
# Final Preferred Dynamic Starts-to-Completions Model
#
# Private starts lag: 1 quarter
# Private completion lags: 1-4 quarters
#
# Preferred shock specification:
# GFC 2008Q3-2009Q4
# COVID 2020Q2, 2020Q3 and 2020Q4
# Quarterly seasonality
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


# ============================================================
# 1. Add preferred historical shock controls
# ============================================================

completion_df["gfc_2008_2009"] = (
    (completion_df.index >= pd.Period("2008Q3", freq="Q")) &
    (completion_df.index <= pd.Period("2009Q4", freq="Q"))
).astype(int)

completion_df["covid_2020Q2"] = (
    completion_df.index == pd.Period("2020Q2", freq="Q")
).astype(int)

completion_df["covid_2020Q3"] = (
    completion_df.index == pd.Period("2020Q3", freq="Q")
).astype(int)

completion_df["covid_2020Q4"] = (
    completion_df.index == pd.Period("2020Q4", freq="Q")
).astype(int)


# ============================================================
# 2. Final preferred model specification
# ============================================================

preferred_formula = """
ln_C_private ~ ln_C_private_lag1
+ ln_C_private_lag2
+ ln_C_private_lag3
+ ln_C_private_lag4
+ ln_S_private_lag1
+ C(quarter_num)
+ gfc_2008_2009
+ covid_2020Q2
+ covid_2020Q3
+ covid_2020Q4
"""


# ============================================================
# 3. Estimate final model
# HAC / Newey-West standard errors
# ============================================================

preferred_model = smf.ols(
    preferred_formula,
    data=completion_df
).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 4}
)


# ============================================================
# 4. Display final model
# ============================================================

print("==========================================")
print("FINAL PREFERRED DYNAMIC BRIDGE MODEL")
print("==========================================")

print("Starts lag: 1 quarter")
print("Completion lags: 1-4 quarters")
print("Observations:", int(preferred_model.nobs))

print(
    "Sample period:",
    completion_df.index.min(),
    "to",
    completion_df.index.max()
)

print("\nShock controls:")
print("GFC: 2008Q3-2009Q4")
print("COVID: 2020Q2, 2020Q3, 2020Q4")
print("Quarterly seasonality included")

print("\n==========================================")
print("MODEL SUMMARY")
print("==========================================")

print(preferred_model.summary())


# ============================================================
# 5. Key model statistics
# ============================================================

print("\n==========================================")
print("KEY MODEL STATISTICS")
print("==========================================")

print(
    "R-squared:",
    preferred_model.rsquared
)

print(
    "Adjusted R-squared:",
    preferred_model.rsquared_adj
)

print(
    "AIC:",
    preferred_model.aic
)

print(
    "BIC:",
    preferred_model.bic
)

print(
    "Starts coefficient:",
    preferred_model.params["ln_S_private_lag1"]
)

print(
    "Starts p-value:",
    preferred_model.pvalues["ln_S_private_lag1"]
)


# In[105]:


# ============================================================
# Final Model Diagnostic Tests
# Dynamic Starts-to-Completions Bridge Model
#
# Starts lag: 1 quarter
# Completion lags: 1-4 quarters
# Preferred historical shock controls
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_breuschpagan
)

from statsmodels.stats.stattools import (
    jarque_bera,
    durbin_watson
)


# ============================================================
# 1. Residuals from final preferred model
# ============================================================

resid = preferred_model.resid


# ============================================================
# 2. Ljung-Box Test
# H0: no residual autocorrelation
# ============================================================

ljung_box = acorr_ljungbox(
    resid,
    lags=[4, 8, 12],
    return_df=True
)

print("==========================================")
print("LJUNG-BOX TEST")
print("==========================================")
print(ljung_box)


# ============================================================
# 3. Durbin-Watson Statistic
# ============================================================

dw_stat = durbin_watson(resid)

print("\n==========================================")
print("DURBIN-WATSON STATISTIC")
print("==========================================")
print("Durbin-Watson:", dw_stat)


# ============================================================
# 4. Breusch-Pagan Test
# H0: homoskedastic residuals
# ============================================================

bp_test = het_breuschpagan(
    resid,
    preferred_model.model.exog
)

bp_results = pd.Series(
    bp_test,
    index=[
        "LM statistic",
        "LM p-value",
        "F statistic",
        "F p-value"
    ]
)

print("\n==========================================")
print("BREUSCH-PAGAN TEST")
print("==========================================")
print(bp_results)


# ============================================================
# 5. Jarque-Bera Test
# H0: normally distributed residuals
# ============================================================

jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(
    resid
)

print("\n==========================================")
print("JARQUE-BERA TEST")
print("==========================================")

print("JB statistic:", jb_stat)
print("p-value:", jb_pvalue)
print("Skewness:", skew)
print("Kurtosis:", kurtosis)


# ============================================================
# 6. Ramsey RESET Test
# H0: no functional-form misspecification
# ============================================================

reset_df = completion_df.copy()

reset_df["fitted_sq"] = (
    np.asarray(preferred_model.fittedvalues) ** 2
)

reset_formula = """
ln_C_private ~ ln_C_private_lag1
+ ln_C_private_lag2
+ ln_C_private_lag3
+ ln_C_private_lag4
+ ln_S_private_lag1
+ C(quarter_num)
+ gfc_2008_2009
+ covid_2020Q2
+ covid_2020Q3
+ covid_2020Q4
+ fitted_sq
"""

reset_model = smf.ols(
    reset_formula,
    data=reset_df
).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 4}
)

reset_test = reset_model.f_test(
    "fitted_sq = 0"
)

reset_fvalue = float(
    np.asarray(reset_test.fvalue).squeeze()
)

reset_pvalue = float(
    np.asarray(reset_test.pvalue).squeeze()
)

print("\n==========================================")
print("RAMSEY RESET TEST")
print("==========================================")

print("F statistic:", reset_fvalue)
print("p-value:", reset_pvalue)


# ============================================================
# 7. Compact Diagnostic Summary
# ============================================================

diagnostic_summary = pd.DataFrame({

    "Diagnostic": [
        "Ljung-Box (lag 4)",
        "Ljung-Box (lag 8)",
        "Ljung-Box (lag 12)",
        "Breusch-Pagan LM",
        "Breusch-Pagan F",
        "Jarque-Bera",
        "Ramsey RESET",
        "Durbin-Watson"
    ],

    "Statistic_or_p_value": [
        ljung_box.loc[4, "lb_pvalue"],
        ljung_box.loc[8, "lb_pvalue"],
        ljung_box.loc[12, "lb_pvalue"],
        bp_results["LM p-value"],
        bp_results["F p-value"],
        jb_pvalue,
        reset_pvalue,
        dw_stat
    ]
})


print("\n==========================================")
print("FINAL DIAGNOSTIC SUMMARY")
print("==========================================")

print(
    diagnostic_summary.to_string(index=False)
)


# In[106]:


# ============================================================
# Multicollinearity Check
# Variance Inflation Factor (VIF)
# Final Preferred Dynamic Starts-to-Completions Model
# ============================================================

import pandas as pd

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)


# ============================================================
# 1. Explanatory-variable matrix
# ============================================================

X = preferred_model.model.exog

variable_names = preferred_model.model.exog_names


# ============================================================
# 2. Calculate VIF
# ============================================================

vif_table = pd.DataFrame({

    "Variable": variable_names,

    "VIF": [
        variance_inflation_factor(X, i)
        for i in range(X.shape[1])
    ]
})


# ============================================================
# 3. Add simple interpretation
# ============================================================

def interpret_vif(row):

    # Intercept VIF is not substantively interpreted
    if row["Variable"] == "Intercept":
        return "Ignore intercept"

    if row["VIF"] < 5:
        return "Low"

    elif row["VIF"] < 10:
        return "Moderate"

    else:
        return "High"


vif_table["Interpretation"] = (
    vif_table.apply(
        interpret_vif,
        axis=1
    )
)


# ============================================================
# 4. Display results
# ============================================================

print("==========================================")
print("VARIANCE INFLATION FACTOR (VIF)")
print("==========================================")

print(
    vif_table.to_string(
        index=False
    )
)


# ============================================================
# 5. Maximum VIF excluding intercept
# ============================================================

vif_without_intercept = vif_table[
    vif_table["Variable"] != "Intercept"
].copy()

max_vif_row = vif_without_intercept.loc[
    vif_without_intercept["VIF"].idxmax()
]


print("\n==========================================")
print("VIF SUMMARY")
print("==========================================")

print(
    "Highest non-intercept VIF:",
    max_vif_row["Variable"]
)

print(
    "VIF:",
    max_vif_row["VIF"]
)


# In[107]:


# ============================================================
# Dynamic Stability Check
# AR(4) Completion Dynamics
# Final Preferred Starts-to-Completions Model
# ============================================================

import numpy as np


# ============================================================
# 1. Extract coefficients on lagged completions
# ============================================================

phi1 = preferred_model.params[
    "ln_C_private_lag1"
]

phi2 = preferred_model.params[
    "ln_C_private_lag2"
]

phi3 = preferred_model.params[
    "ln_C_private_lag3"
]

phi4 = preferred_model.params[
    "ln_C_private_lag4"
]


# ============================================================
# 2. Display AR coefficients
# ============================================================

print("==========================================")
print("AR(4) COMPLETION COEFFICIENTS")
print("==========================================")

print("phi1 =", phi1)
print("phi2 =", phi2)
print("phi3 =", phi3)
print("phi4 =", phi4)


# ============================================================
# 3. Construct AR(4) companion matrix
# ============================================================

companion_matrix = np.array([
    [phi1, phi2, phi3, phi4],
    [1.0,  0.0,  0.0,  0.0],
    [0.0,  1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0,  0.0]
])


# ============================================================
# 4. Calculate eigenvalues
# ============================================================

eigenvalues = np.linalg.eigvals(
    companion_matrix
)

modulus = np.abs(
    eigenvalues
)


# ============================================================
# 5. Display stability results
# ============================================================

print("\n==========================================")
print("DYNAMIC STABILITY CHECK")
print("==========================================")

print("\nAR eigenvalues:")
print(eigenvalues)

print("\nModulus of eigenvalues:")
print(modulus)

print("\nMaximum modulus:")
print(modulus.max())


# ============================================================
# 6. Stability conclusion
# ============================================================

if np.all(modulus < 1):

    print("\nRESULT: Dynamic specification is STABLE.")
    print(
        "All eigenvalues lie inside the unit circle."
    )

else:

    print("\nRESULT: Dynamic specification is NOT STABLE.")
    print(
        "At least one eigenvalue lies on or outside "
        "the unit circle."
    )


# ============================================================
# 7. Compact stability summary
# ============================================================

print("\n==========================================")
print("STABILITY SUMMARY")
print("==========================================")

print(
    "Maximum eigenvalue modulus:",
    modulus.max()
)

print(
    "Stable:",
    bool(np.all(modulus < 1))
)


# In[108]:


# ============================================================
# Actual vs Fitted Private Enterprise Completions
# Final Preferred AR(4) Starts-to-Completions Model
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. Generate fitted log private completions
# ============================================================

completion_df["fitted_ln_C_private"] = (
    preferred_model.predict(completion_df)
)


# ============================================================
# 2. Smearing correction
# Convert fitted log values back to dwelling levels
# ============================================================

smearing_factor = np.mean(
    np.exp(preferred_model.resid)
)

completion_df["fitted_C_private"] = (
    np.exp(completion_df["fitted_ln_C_private"])
    * smearing_factor
)


# ============================================================
# 3. Actual private completions in levels
# ============================================================

completion_df["actual_C_private"] = (
    np.exp(completion_df["ln_C_private"])
)


# ============================================================
# 4. Convert quarterly index to dates for plotting
# ============================================================

completion_df["date"] = (
    completion_df.index.to_timestamp()
)


# ============================================================
# 5. Display smearing factor
# ============================================================

print("==========================================")
print("SMEARING CORRECTION")
print("==========================================")

print(
    "Smearing factor:",
    smearing_factor
)


# ============================================================
# 6. Plot Actual vs Fitted
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    completion_df["date"],
    completion_df["actual_C_private"],
    label="Actual private completions"
)

plt.plot(
    completion_df["date"],
    completion_df["fitted_C_private"],
    label="Fitted private completions"
)

plt.title(
    "Actual and Fitted Private Enterprise Completions"
)

plt.xlabel("Year")
plt.ylabel("Dwellings")

plt.legend()
plt.tight_layout()


# ============================================================
# 7. Save final figure
# ============================================================

OUTPUT_DIR = Path(r"D:\data\outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.savefig(
    OUTPUT_DIR / "final_actual_vs_fitted_private_completions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nFigure saved successfully.")


# In[109]:


# ============================================================
# Final Model Summary
# Final Preferred AR(4) Starts-to-Completions Bridge Model
# ============================================================

import pandas as pd


preferred_summary = pd.DataFrame({

    "Model": [
        "Dynamic Starts-to-Completions Bridge Model"
    ],

    "Dependent_variable": [
        "Log private enterprise completions"
    ],

    "Starts_lag": [
        "1 quarter"
    ],

    "Completion_lags": [
        "1-4 quarters"
    ],

    "Shock_controls": [
        "GFC 2008Q3-2009Q4; COVID 2020Q2-Q4"
    ],

    "Observations": [
        int(preferred_model.nobs)
    ],

    "R_squared": [
        preferred_model.rsquared
    ],

    "Adj_R_squared": [
        preferred_model.rsquared_adj
    ],

    "AIC": [
        preferred_model.aic
    ],

    "BIC": [
        preferred_model.bic
    ],

    "Starts_coefficient": [
        preferred_model.params["ln_S_private_lag1"]
    ],

    "Starts_p_value": [
        preferred_model.pvalues["ln_S_private_lag1"]
    ],

    "Smearing_factor": [
        smearing_factor
    ]
})


print("==========================================")
print("FINAL MODEL SUMMARY")
print("==========================================")

print(
    preferred_summary.to_string(index=False)
)


# In[110]:


# ============================================================
# Final Regression Coefficient Table
# Final Preferred AR(4) Starts-to-Completions Model
# ============================================================

from pathlib import Path
import pandas as pd


# ============================================================
# 1. Output folder
# ============================================================

OUTPUT_DIR = Path(r"D:\data\outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Create coefficient table
# ============================================================

preferred_results_table = pd.DataFrame({

    "Variable": preferred_model.params.index,

    "Coefficient": preferred_model.params.values,

    "HAC_Std_Error": preferred_model.bse.values,

    "P_Value": preferred_model.pvalues.values,

    "CI_Lower_95": preferred_model.conf_int()[0].values,

    "CI_Upper_95": preferred_model.conf_int()[1].values
})


# ============================================================
# 3. Add significance stars
# ============================================================

def significance_star(p):

    if p < 0.01:
        return "***"

    elif p < 0.05:
        return "**"

    elif p < 0.10:
        return "*"

    else:
        return ""


preferred_results_table["Significance"] = (
    preferred_results_table["P_Value"]
    .apply(significance_star)
)


# ============================================================
# 4. Round values for readable output
# ============================================================

display_table = preferred_results_table.copy()

for col in [
    "Coefficient",
    "HAC_Std_Error",
    "P_Value",
    "CI_Lower_95",
    "CI_Upper_95"
]:
    display_table[col] = display_table[col].round(6)


# ============================================================
# 5. Display final coefficient table
# ============================================================

print("==========================================")
print("FINAL REGRESSION COEFFICIENT TABLE")
print("==========================================")

print(
    display_table.to_string(index=False)
)


# ============================================================
# 6. Save full-precision results
# ============================================================

preferred_results_table.to_excel(
    OUTPUT_DIR / "final_dynamic_bridge_model_coefficients.xlsx",
    index=False
)

preferred_summary.to_excel(
    OUTPUT_DIR / "final_dynamic_bridge_model_summary.xlsx",
    index=False
)


print(
    "\nCoefficient table and model summary saved successfully."
)


# In[111]:


# ============================================================
# Final Diagnostic Summary
# Final Preferred AR(4) Starts-to-Completions Model
# ============================================================

import pandas as pd


# ============================================================
# 1. Create final diagnostic reporting table
# ============================================================

final_diagnostic_table = pd.DataFrame({

    "Diagnostic": [
        "Ljung-Box (lag 4)",
        "Ljung-Box (lag 8)",
        "Ljung-Box (lag 12)",
        "Breusch-Pagan LM",
        "Breusch-Pagan F",
        "Jarque-Bera",
        "Ramsey RESET",
        "Durbin-Watson"
    ],

    "Value": [
        ljung_box.loc[4, "lb_pvalue"],
        ljung_box.loc[8, "lb_pvalue"],
        ljung_box.loc[12, "lb_pvalue"],
        bp_results["LM p-value"],
        bp_results["F p-value"],
        jb_pvalue,
        reset_pvalue,
        dw_stat
    ],

    "Interpretation": [
        "No residual autocorrelation",
        "No residual autocorrelation",
        "No residual autocorrelation",
        "No evidence of heteroskedasticity",
        "No evidence of heteroskedasticity",
        "Residuals reject normality",
        "No major functional-form misspecification",
        "Close to 2; little first-order autocorrelation"
    ]
})


# ============================================================
# 2. Display final diagnostic summary
# ============================================================

print("==========================================")
print("FINAL DIAGNOSTIC SUMMARY")
print("==========================================")

print(
    final_diagnostic_table
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 3. Save diagnostic summary
# ============================================================

final_diagnostic_table.to_excel(
    OUTPUT_DIR / "final_model_diagnostic_tests.xlsx",
    index=False
)

print("\nDiagnostic summary saved successfully.")


# In[121]:


# ============================================================
# Final Model Equation Figure
# Final Preferred AR(4) Starts-to-Completions Model
# ============================================================

import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(r"D:\data\outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(15, 7))
ax.axis("off")


# ============================================================
# Title
# ============================================================

ax.text(
    0.5,
    0.94,
    "Final Dynamic Starts-to-Completions Bridge Model",
    fontsize=20,
    fontweight="bold",
    ha="center",
    va="top"
)


# ============================================================
# Equation - Line 1
# ============================================================

ax.text(
    0.5,
    0.76,
    r"$\ln(C_t^p)"
    r" = \alpha"
    r" + \phi_1\ln(C_{t-1}^p)"
    r" + \phi_2\ln(C_{t-2}^p)"
    r" + \phi_3\ln(C_{t-3}^p)"
    r" + \phi_4\ln(C_{t-4}^p)$",
    fontsize=20,
    ha="center"
)


# ============================================================
# Equation - Line 2
# ============================================================

ax.text(
    0.5,
    0.62,
    r"$+ \beta\ln(S_{t-1}^p)"
    r" + \gamma_2Q_{2,t}"
    r" + \gamma_3Q_{3,t}"
    r" + \gamma_4Q_{4,t}"
    r" + \delta GFC_t$",
    fontsize=20,
    ha="center"
)


# ============================================================
# Equation - Line 3
# ============================================================

ax.text(
    0.5,
    0.48,
    r"$+ \eta_1D_{2020Q2,t}"
    r" + \eta_2D_{2020Q3,t}"
    r" + \eta_3D_{2020Q4,t}"
    r" + \varepsilon_t$",
    fontsize=20,
    ha="center"
)


# ============================================================
# Definitions
# ============================================================

ax.text(
    0.07,
    0.31,
    r"$C_t^p$ = private enterprise completions in quarter $t$",
    fontsize=14
)

ax.text(
    0.07,
    0.24,
    r"$C_{t-1}^p, C_{t-2}^p, C_{t-3}^p, C_{t-4}^p$ = lagged private enterprise completions",
    fontsize=14
)

ax.text(
    0.07,
    0.17,
    r"$S_{t-1}^p$ = private enterprise starts lagged by one quarter",
    fontsize=14
)

ax.text(
    0.07,
    0.10,
    r"$Q_{2,t}, Q_{3,t}, Q_{4,t}$ = quarterly seasonal dummy variables",
    fontsize=14
)

ax.text(
    0.07,
    0.03,
    r"$GFC_t$ = 2008--2009 financial crisis dummy; "
    r"$D_{2020Q2,t}, D_{2020Q3,t}, D_{2020Q4,t}$ = COVID shock dummies",
    fontsize=13
)


# ============================================================
# Save Figure
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_dynamic_bridge_model_equation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Final model equation figure saved successfully.")


# In[113]:


# ============================================================
# Figure: Model Fit across Alternative Private Starts Lags
# Preferred Shock Specification
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path


# ============================================================
# 1. Create dataset for starts-lag comparison
# ============================================================

preferred_lag_df = df.copy()

preferred_lag_df = preferred_lag_df[
    (preferred_lag_df["starts_private"] > 0) &
    (preferred_lag_df["completions_private"] > 0)
].copy()


# Log variables
preferred_lag_df["ln_C_private"] = np.log(
    preferred_lag_df["completions_private"]
)

preferred_lag_df["ln_S_private"] = np.log(
    preferred_lag_df["starts_private"]
)


# Completion lags used during starts-lag selection
preferred_lag_df["ln_C_private_lag1"] = (
    preferred_lag_df["ln_C_private"].shift(1)
)

preferred_lag_df["ln_C_private_lag2"] = (
    preferred_lag_df["ln_C_private"].shift(2)
)


# Starts lags 1-12
for lag in range(1, 13):
    preferred_lag_df[f"ln_S_private_lag{lag}"] = (
        preferred_lag_df["ln_S_private"].shift(lag)
    )


# Quarterly seasonality
preferred_lag_df["quarter_num"] = (
    preferred_lag_df.index.quarter
)


# ============================================================
# 2. Preferred historical shock controls
# ============================================================

preferred_lag_df["gfc_2008_2009"] = (
    (preferred_lag_df.index >= pd.Period("2008Q3", freq="Q")) &
    (preferred_lag_df.index <= pd.Period("2009Q4", freq="Q"))
).astype(int)

preferred_lag_df["covid_2020Q2"] = (
    preferred_lag_df.index == pd.Period("2020Q2", freq="Q")
).astype(int)

preferred_lag_df["covid_2020Q3"] = (
    preferred_lag_df.index == pd.Period("2020Q3", freq="Q")
).astype(int)

preferred_lag_df["covid_2020Q4"] = (
    preferred_lag_df.index == pd.Period("2020Q4", freq="Q")
).astype(int)


# Common sample for all 12 models
preferred_lag_df = preferred_lag_df.dropna().copy()


# ============================================================
# 3. Compare starts lags 1-12
# ============================================================

preferred_lag_results = []

for lag in range(1, 13):

    formula = f"""
    ln_C_private ~ ln_C_private_lag1
    + ln_C_private_lag2
    + ln_S_private_lag{lag}
    + C(quarter_num)
    + gfc_2008_2009
    + covid_2020Q2
    + covid_2020Q3
    + covid_2020Q4
    """

    model = smf.ols(
        formula,
        data=preferred_lag_df
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 4}
    )

    preferred_lag_results.append({
        "Lag": lag,
        "Coefficient": model.params[f"ln_S_private_lag{lag}"],
        "P_value": model.pvalues[f"ln_S_private_lag{lag}"],
        "Adj_R_squared": model.rsquared_adj,
        "AIC": model.aic,
        "BIC": model.bic
    })


preferred_lag_results = pd.DataFrame(
    preferred_lag_results
)


# ============================================================
# 4. Identify selected lag
# ============================================================

best_row = preferred_lag_results.loc[
    preferred_lag_results["Adj_R_squared"].idxmax()
]

best_lag = int(best_row["Lag"])
best_r2 = best_row["Adj_R_squared"]


print("==========================================")
print("PREFERRED STARTS-LAG SELECTION")
print("==========================================")

print(
    preferred_lag_results
    .round(6)
    .to_string(index=False)
)

print("\nSelected starts lag:", best_lag)
print("Adjusted R-squared:", best_r2)
print("P-value:", best_row["P_value"])


# ============================================================
# 5. Plot model fit
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    preferred_lag_results["Lag"],
    preferred_lag_results["Adj_R_squared"],
    marker="o"
)

plt.scatter(
    best_lag,
    best_r2,
    s=80
)

plt.annotate(
    f"Selected lag = {best_lag}",
    (best_lag, best_r2),
    xytext=(best_lag + 0.7, best_r2 - 0.01),
    arrowprops=dict(arrowstyle="->")
)

plt.title(
    "Model Fit across Alternative Private Enterprise Starts Lags"
)

plt.xlabel(
    "Private enterprise starts lag (quarters)"
)

plt.ylabel(
    "Adjusted $R^2$"
)

plt.xticks(range(1, 13))

plt.tight_layout()


# ============================================================
# 6. Save figure
# ============================================================

OUTPUT_DIR = Path(r"D:\data\outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.savefig(
    OUTPUT_DIR / "starts_lag_selection_adj_r2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nStarts-lag selection figure saved successfully.")


# In[114]:


# ============================================================
# Future Private Completions Forecast Equation
# ============================================================

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16, 7))
ax.axis("off")

# Title
ax.text(
    0.5,
    0.92,
    "Recursive Future Private Completions Forecast",
    fontsize=20,
    fontweight="bold",
    ha="center"
)

# Forecast equation - line 1
ax.text(
    0.5,
    0.72,
    r"$\widehat{\ln(C_{T+h}^{p})}"
    r" = \widehat{\alpha}"
    r" + \widehat{\phi}_1\widehat{\ln(C_{T+h-1}^{p})}"
    r" + \widehat{\phi}_2\widehat{\ln(C_{T+h-2}^{p})}$",
    fontsize=18,
    ha="center"
)

# Forecast equation - line 2
ax.text(
    0.5,
    0.59,
    r"$+ \widehat{\phi}_3\widehat{\ln(C_{T+h-3}^{p})}"
    r" + \widehat{\phi}_4\widehat{\ln(C_{T+h-4}^{p})}"
    r" + \widehat{\beta}\ln(\widehat{S}_{T+h-1}^{p})$",
    fontsize=18,
    ha="center"
)

# Forecast equation - line 3
ax.text(
    0.5,
    0.46,
    r"$+ \widehat{\gamma}_2Q_{2,T+h}"
    r" + \widehat{\gamma}_3Q_{3,T+h}"
    r" + \widehat{\gamma}_4Q_{4,T+h}$",
    fontsize=18,
    ha="center"
)

# Definitions
ax.text(
    0.07,
    0.29,
    r"$\widehat{C}_{T+h}^{p}$ = forecast private enterprise completions at horizon $h$",
    fontsize=14
)

ax.text(
    0.07,
    0.22,
    r"$\widehat{S}_{T+h-1}^{p}$ = forecast private enterprise starts lagged by one quarter",
    fontsize=14
)

ax.text(
    0.07,
    0.15,
    "Forecasts are generated recursively: predicted completions from earlier "
    "future quarters are used as inputs for later quarters.",
    fontsize=14
)

ax.text(
    0.07,
    0.08,
    "GFC and COVID dummy variables are set to zero for normal future periods.",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "future_private_completions_forecast_equation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Future forecast equation figure saved successfully.")


# In[115]:


import matplotlib.pyplot as plt

# If OUTPUT_DIR has not been created before, uncomment the next two lines:
# from pathlib import Path
# OUTPUT_DIR = Path(r"D:\data\outputs")

plt.figure(figsize=(14, 7))
plt.axis('off')

# Title
plt.text(
    0.5, 0.93,
    "Back-Transformation from Log Forecasts to Private Completions",
    ha='center', va='top',
    fontsize=18, fontweight='bold'
)

# Intro sentence
plt.text(
    0.05, 0.84,
    "Because the forecasting model is estimated in logarithms, the predicted log completions",
    fontsize=12
)
plt.text(
    0.05, 0.80,
    "must be converted back into dwelling levels using an exponential transformation",
    fontsize=12
)
plt.text(
    0.05, 0.76,
    "with a smearing-factor correction.",
    fontsize=12
)

# Equation 1
eq1 = (
    r"$\widehat{C}_{T+h}^{p}"
    r"="
    r"\exp\left(\widehat{\ln(C_{T+h}^{p})}\right)\times SF$"
)

plt.text(
    0.5, 0.62,
    eq1,
    ha='center', va='center',
    fontsize=22
)

# Equation 2
eq2 = (
    r"$SF=\frac{1}{T}\sum_{t=1}^{T}\exp(\widehat{u}_t)$"
)

plt.text(
    0.5, 0.47,
    eq2,
    ha='center', va='center',
    fontsize=22
)

# Definitions
plt.text(
    0.08, 0.30,
    r"$\widehat{C}_{T+h}^{p}$ = forecast private enterprise completions in dwelling levels at horizon $h$",
    fontsize=12
)
plt.text(
    0.08, 0.24,
    r"$\widehat{\ln(C_{T+h}^{p})}$ = forecast log private enterprise completions from the dynamic bridge model",
    fontsize=12
)
plt.text(
    0.08, 0.18,
    r"$SF$ = smearing factor used to correct retransformation bias",
    fontsize=12
)
plt.text(
    0.08, 0.12,
    r"$\widehat{u}_t$ = fitted residual from the estimated log-completions model",
    fontsize=12
)

# Short interpretation
plt.text(
    0.05, 0.04,
    "Interpretation: the log forecast is exponentiated and adjusted by the smearing factor "
    "to recover a forecast of private completions in dwelling units.",
    fontsize=11
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "back_transformation_private_completions_equation.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("Back-transformation equation figure saved successfully.")


# In[116]:


# ============================================================
# Net Additional Dwellings Accounting Identity
# ============================================================

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")

# Title
ax.text(
    0.5,
    0.92,
    "Net Additional Dwellings Accounting Identity",
    fontsize=20,
    fontweight="bold",
    ha="center"
)

# Main equation
ax.text(
    0.5,
    0.68,
    r"$NA_t = NB_t + CU_t + CV_t + OG_t - D_t$",
    fontsize=24,
    ha="center"
)

# Definitions
ax.text(
    0.10,
    0.50,
    r"$NA_t$ = net additional dwellings",
    fontsize=15
)

ax.text(
    0.10,
    0.43,
    r"$NB_t$ = new-build completions",
    fontsize=15
)

ax.text(
    0.10,
    0.36,
    r"$CU_t$ = gains from changes of use",
    fontsize=15
)

ax.text(
    0.10,
    0.29,
    r"$CV_t$ = gains from conversions",
    fontsize=15
)

ax.text(
    0.10,
    0.22,
    r"$OG_t$ = other gains",
    fontsize=15
)

ax.text(
    0.10,
    0.15,
    r"$D_t$ = demolitions",
    fontsize=15
)

# Interpretation
ax.text(
    0.10,
    0.06,
    "Net additions are therefore broader than new-build completions alone.",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "net_additional_dwellings_identity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Net additions equation figure saved successfully.")


# In[117]:


# ============================================================
# Social/Public Housing Completions Identity
# ============================================================

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")

# Title
ax.text(
    0.5,
    0.92,
    "Social/Public Housing Completions Identity",
    fontsize=20,
    fontweight="bold",
    ha="center"
)

# Main equation
ax.text(
    0.5,
    0.68,
    r"$SPC_t = HAC_t + LAC_t$",
    fontsize=24,
    ha="center"
)

# Definitions
ax.text(
    0.10,
    0.50,
    r"$SPC_t$ = total social/public housing completions",
    fontsize=15
)

ax.text(
    0.10,
    0.41,
    r"$HAC_t$ = housing association completions",
    fontsize=15
)

ax.text(
    0.10,
    0.32,
    r"$LAC_t$ = local authority completions",
    fontsize=15
)

# Interpretation
ax.text(
    0.10,
    0.18,
    "This identity separates the policy-driven social/public supply component",
    fontsize=14
)

ax.text(
    0.10,
    0.12,
    "from the market-driven private completions forecast.",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "social_public_housing_completions_identity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Social/public housing equation figure saved successfully.")


# In[118]:


# ============================================================
# Social/Public Housing Policy Scenario
# ============================================================

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")

# Title
ax.text(
    0.5,
    0.92,
    "Social/Public Housing Policy Scenario",
    fontsize=20,
    fontweight="bold",
    ha="center"
)

# Main equation
ax.text(
    0.5,
    0.69,
    r"$\widehat{SPC}_{T+h}^{(s)} = B_q(1+s)$",
    fontsize=26,
    ha="center"
)

# Definitions
ax.text(
    0.10,
    0.52,
    r"$\widehat{SPC}_{T+h}^{(s)}$ = forecast social/public completions under scenario $s$",
    fontsize=14
)

ax.text(
    0.10,
    0.44,
    r"$B_q$ = baseline social/public completions for quarter $q$",
    fontsize=14
)

ax.text(
    0.10,
    0.36,
    r"$s$ = assumed policy adjustment",
    fontsize=14
)

# Scenario values
ax.text(
    0.5,
    0.25,
    r"$s \in \{-0.10,\;0,\;0.10,\;0.20\}$",
    fontsize=22,
    ha="center"
)

ax.text(
    0.10,
    0.15,
    "Scenarios: -10% constrained, baseline, +10% uplift, and +20% strong uplift.",
    fontsize=14
)

ax.text(
    0.10,
    0.08,
    "These are illustrative policy scenarios rather than estimated causal forecasts.",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "social_public_policy_scenario_equation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Social/public policy scenario figure saved successfully.")


# In[119]:


# ============================================================
# Total Housing Completions Framework
# ============================================================

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")

# Title
ax.text(
    0.5,
    0.92,
    "Total Housing Completions Framework",
    fontsize=20,
    fontweight="bold",
    ha="center"
)

# Main equation
ax.text(
    0.5,
    0.68,
    r"$\widehat{TC}_{T+h}"
    r"="
    r"\widehat{PC}_{T+h}"
    r"+"
    r"\widehat{SPC}_{T+h}^{(s)}$",
    fontsize=26,
    ha="center"
)

# Definitions
ax.text(
    0.10,
    0.50,
    r"$\widehat{TC}_{T+h}$ = forecast total housing completions",
    fontsize=15
)

ax.text(
    0.10,
    0.41,
    r"$\widehat{PC}_{T+h}$ = forecast private enterprise completions",
    fontsize=15
)

ax.text(
    0.10,
    0.32,
    r"$\widehat{SPC}_{T+h}^{(s)}$ = forecast social/public completions under policy scenario $s$",
    fontsize=15
)

# Interpretation
ax.text(
    0.10,
    0.18,
    "The private component is generated by the dynamic starts-to-completions bridge model,",
    fontsize=14
)

ax.text(
    0.10,
    0.12,
    "while the social/public component is determined separately under the policy scenario framework.",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "total_housing_completions_framework.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Total housing completions framework figure saved successfully.")


# In[120]:


# ============================================================
# Complete Housing Delivery Framework
# Final Preferred Framework
# ============================================================

import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. Output folder
# ============================================================

OUTPUT_DIR = Path(r"D:\data\outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(15, 9))
ax.axis("off")


# ============================================================
# Title
# ============================================================

ax.text(
    0.5,
    0.96,
    "Complete Housing Delivery Framework",
    fontsize=21,
    fontweight="bold",
    ha="center"
)


# ============================================================
# Private-sector forecasting pipeline
# ============================================================

ax.text(
    0.30,
    0.87,
    "Private Enterprise Starts Forecast",
    fontsize=16,
    fontweight="bold",
    ha="center"
)

ax.text(
    0.30,
    0.81,
    "↓",
    fontsize=25,
    ha="center"
)

ax.text(
    0.30,
    0.74,
    "Dynamic Starts-to-Completions Bridge Model",
    fontsize=16,
    fontweight="bold",
    ha="center"
)

ax.text(
    0.30,
    0.695,
    "1-quarter starts lag + 1–4 completion lags",
    fontsize=12,
    ha="center"
)

ax.text(
    0.30,
    0.64,
    "↓",
    fontsize=25,
    ha="center"
)

ax.text(
    0.30,
    0.57,
    "Private Enterprise Completions Forecast",
    fontsize=16,
    fontweight="bold",
    ha="center"
)


# ============================================================
# Optional drop-out sensitivity adjustment
# ============================================================

ax.text(
    0.30,
    0.51,
    "↓",
    fontsize=25,
    ha="center"
)

ax.text(
    0.30,
    0.44,
    "Optional Drop-Out Scenario Adjustment",
    fontsize=15,
    ha="center"
)

ax.text(
    0.30,
    0.395,
    "Sensitivity analysis only",
    fontsize=11,
    ha="center"
)

ax.text(
    0.30,
    0.34,
    "↓",
    fontsize=25,
    ha="center"
)

ax.text(
    0.30,
    0.28,
    "Adjusted Private Completions",
    fontsize=16,
    fontweight="bold",
    ha="center"
)


# ============================================================
# Social / public housing component
# ============================================================

ax.text(
    0.72,
    0.44,
    "Social/Public Housing Policy Scenario",
    fontsize=16,
    fontweight="bold",
    ha="center"
)

ax.text(
    0.72,
    0.395,
    "Housing Association + Local Authority completions",
    fontsize=11,
    ha="center"
)

ax.text(
    0.72,
    0.34,
    "↓",
    fontsize=25,
    ha="center"
)

ax.text(
    0.72,
    0.28,
    "Social/Public Completions",
    fontsize=16,
    fontweight="bold",
    ha="center"
)


# ============================================================
# Combine private and social/public completions
# ============================================================

ax.text(
    0.50,
    0.28,
    "+",
    fontsize=26,
    fontweight="bold",
    ha="center"
)


# Arrows from both components to total new-build completions
ax.annotate(
    "",
    xy=(0.47, 0.19),
    xytext=(0.30, 0.25),
    arrowprops=dict(arrowstyle="->", linewidth=1.5)
)

ax.annotate(
    "",
    xy=(0.53, 0.19),
    xytext=(0.72, 0.25),
    arrowprops=dict(arrowstyle="->", linewidth=1.5)
)


# ============================================================
# Total new-build completions
# ============================================================

ax.text(
    0.50,
    0.16,
    "Total New-Build Completions",
    fontsize=17,
    fontweight="bold",
    ha="center"
)

ax.text(
    0.50,
    0.115,
    "Private + Social/Public",
    fontsize=11,
    ha="center"
)

ax.text(
    0.50,
    0.075,
    "↓",
    fontsize=25,
    ha="center"
)


# ============================================================
# Net additions
# ============================================================

ax.text(
    0.50,
    0.025,
    "Net Additional Dwellings",
    fontsize=17,
    fontweight="bold",
    ha="center"
)

ax.text(
    0.82,
    0.10,
    "Net additions also include:\n"
    "+ changes of use\n"
    "+ conversions\n"
    "+ other gains\n"
    "− demolitions",
    fontsize=11,
    ha="left",
    va="center"
)


# ============================================================
# Save figure
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "complete_housing_delivery_framework.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Complete housing delivery framework saved successfully.")

