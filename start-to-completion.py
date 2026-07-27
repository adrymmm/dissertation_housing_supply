#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# Load the Excel file and display all sheet names
xls = pd.ExcelFile(housebuilding_file)
xls.sheet_names


# In[3]:


# Read sheet 1b, which contains England quarterly housebuilding data
raw = pd.read_excel(
    housebuilding_file,
    sheet_name="1b",
    header=5
)

# Display the first few rows
raw.head()


# In[4]:


# Rename the key columns for easier use
df = raw.rename(columns={
    "Period": "period",
    "Started - All Dwellings": "starts_all",
    "Completed - All Dwellings": "completions_all",
    "Started - Private Enterprise": "starts_private",
    "Completed - Private Enterprise": "completions_private"
})

# Keep only the variables needed for the analysis
df = df[[
    "period",
    "starts_all",
    "completions_all",
    "starts_private",
    "completions_private"
]].copy()

# Convert numerical columns to numeric format
for col in ["starts_all", "completions_all", "starts_private", "completions_private"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert the period text into a quarterly time index
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
    
    return pd.Period(year=year, quarter=q, freq="Q")

# Apply the quarter parser
df["quarter"] = df["period"].apply(parse_quarter)

# Drop rows without valid quarters and set quarter as the index
df = df.dropna(subset=["quarter"])
df = df.set_index("quarter").sort_index()

# Display the cleaned data
df.head()


# In[5]:


# Check the start and end quarters of the dataset
print(df.index.min())
print(df.index.max())

# Display the last few rows
df.tail()


# In[6]:


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


# In[7]:


# Estimate single-lag models for lags from 1 to 12 quarters
single_lag_results = []

for lag in range(1, 13):
    temp = df.copy()
    
    # Create lagged starts variable
    temp[f"starts_lag{lag}"] = temp["starts_all"].shift(lag)
    
    # Add quarter number for seasonal controls
    temp["quarter_num"] = temp.index.quarter
    
    # Drop missing observations caused by lagging
    temp = temp.dropna()

    # Define the regression formula
    formula = f"completions_all ~ starts_lag{lag} + C(quarter_num)"
    
    # Estimate OLS model with Newey-West standard errors
    model = smf.ols(formula, data=temp).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 4}
    )
    
    # Store model results
    single_lag_results.append({
        "lag": lag,
        "r_squared": model.rsquared,
        "aic": model.aic,
        "bic": model.bic,
        "beta": model.params[f"starts_lag{lag}"],
        "p_value": model.pvalues[f"starts_lag{lag}"]
    })

# Convert results into a DataFrame
single_lag_results = pd.DataFrame(single_lag_results)

# Sort models by AIC
single_lag_results.sort_values("aic")


# In[8]:


# Plot R-squared values across different lags
plt.figure(figsize=(8, 5))
plt.plot(single_lag_results["lag"], single_lag_results["r_squared"], marker="o")
plt.title("Single-Lag Model Fit: Starts to Completions")
plt.xlabel("Lag in quarters")
plt.ylabel("R-squared")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()


# In[9]:


# Create a modelling dataset
model_df = df.copy()

# Add lagged completions to capture persistence in completions
model_df["completions_lag1"] = model_df["completions_all"].shift(1)

# Add lagged starts from 4 to 8 quarters
for lag in range(4, 9):
    model_df[f"starts_lag{lag}"] = model_df["starts_all"].shift(lag)

# Add quarter number for seasonal dummy variables
model_df["quarter_num"] = model_df.index.quarter

# Drop missing observations
model_df = model_df.dropna()

# Define the distributed lag model
formula = """
completions_all ~ completions_lag1
+ starts_lag4 + starts_lag5 + starts_lag6 + starts_lag7 + starts_lag8
+ C(quarter_num)
"""

# Estimate the model with Newey-West standard errors
completion_model = smf.ols(formula, data=model_df).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 4}
)

# Print the regression summary
print(completion_model.summary())


# In[12]:


# Create a simplified model using the most significant lag
simple_df = df.copy()

# Add lagged completions to capture persistence
simple_df["completions_lag1"] = simple_df["completions_all"].shift(1)

# Add starts lagged by 4 quarters
simple_df["starts_lag4"] = simple_df["starts_all"].shift(4)

# Add quarter number for seasonal dummy variables
simple_df["quarter_num"] = simple_df.index.quarter

# Drop missing observations
simple_df = simple_df.dropna()

# Define the simplified time-to-build model
simple_formula = """
completions_all ~ completions_lag1
+ starts_lag4
+ C(quarter_num)
"""

# Estimate the simplified model with Newey-West standard errors
simple_model = smf.ols(simple_formula, data=simple_df).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 4}
)

# Print the regression summary
print(simple_model.summary())


# In[14]:


# Create a new dataset for the more advanced model
complex_df = df.copy()

# Keep only positive observations because log transformation requires positive values
complex_df = complex_df[
    (complex_df["starts_all"] > 0) &
    (complex_df["completions_all"] > 0)
].copy()

# Create log variables
complex_df["ln_C"] = np.log(complex_df["completions_all"])
complex_df["ln_S"] = np.log(complex_df["starts_all"])

# Add lagged completions
complex_df["ln_C_lag1"] = complex_df["ln_C"].shift(1)

# Add lagged starts from 4 to 8 quarters
for lag in range(4, 9):
    complex_df[f"ln_S_lag{lag}"] = complex_df["ln_S"].shift(lag)

# Add quarter number for seasonal controls
complex_df["quarter_num"] = complex_df.index.quarter

# Drop missing observations caused by lagging
complex_df = complex_df.dropna()

complex_df.head()


# In[15]:


# Log lag-4 benchmark model
log_simple_formula = """
ln_C ~ ln_C_lag1
+ ln_S_lag4
+ C(quarter_num)
"""

log_simple_model = smf.ols(log_simple_formula, data=complex_df).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 4}
)

print(log_simple_model.summary())


# In[16]:


# Unrestricted log distributed lag model
log_full_formula = """
ln_C ~ ln_C_lag1
+ ln_S_lag4 + ln_S_lag5 + ln_S_lag6 + ln_S_lag7 + ln_S_lag8
+ C(quarter_num)
"""

log_full_model = smf.ols(log_full_formula, data=complex_df).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 4}
)

print(log_full_model.summary())


# In[60]:


import matplotlib.pyplot as plt

try:
    OUTPUT_DIR
except NameError:
    OUTPUT_DIR = DATA_DIR / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")

formula_text = r"""
Main Model: Dynamic Log Almon Distributed Lag Model

The model estimates how housing starts are translated into completions over time.

$\ln(C_t) = \alpha + \rho \ln(C_{t-1}) + \theta_0 Z_{0t} + \theta_1 Z_{1t} + \theta_2 Z_{2t} + Q_t + \varepsilon_t$

Almon lag terms:

$Z_{0t} = \sum_{j=4}^{8} \ln(S_{t-j})$

$Z_{1t} = \sum_{j=4}^{8} j \ln(S_{t-j})$

$Z_{2t} = \sum_{j=4}^{8} j^2 \ln(S_{t-j})$

Definitions:

$C_t$ = housing completions in quarter $t$

$S_{t-j}$ = housing starts lagged by $j$ quarters

$\ln(C_{t-1})$ = lagged completions, capturing persistence

$j = 4,\ldots,8$ = construction lag from starts to completions

$Q_t$ = quarterly seasonal dummy variables

$\varepsilon_t$ = error term
"""

ax.text(0.02, 0.98, formula_text, fontsize=16, va="top", ha="left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "formula_1_main_almon_model.png", dpi=300, bbox_inches="tight")
plt.show()


# In[17]:


# Almon polynomial distributed lag model
# The idea is to model the lag coefficients as a smooth polynomial function of the lag length

lags = range(4, 9)

# Create polynomial distributed lag variables
complex_df["Z0"] = sum(complex_df[f"ln_S_lag{lag}"] for lag in lags)
complex_df["Z1"] = sum(lag * complex_df[f"ln_S_lag{lag}"] for lag in lags)
complex_df["Z2"] = sum((lag ** 2) * complex_df[f"ln_S_lag{lag}"] for lag in lags)

# Define the Almon polynomial distributed lag model
almon_formula = """
ln_C ~ ln_C_lag1
+ Z0 + Z1 + Z2
+ C(quarter_num)
"""

almon_model = smf.ols(almon_formula, data=complex_df).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 4}
)

print(almon_model.summary())


# In[18]:


# Recover implied lag coefficients from the Almon polynomial model
theta0 = almon_model.params["Z0"]
theta1 = almon_model.params["Z1"]
theta2 = almon_model.params["Z2"]

lag_weights = pd.DataFrame({
    "lag": list(lags)
})

lag_weights["implied_beta"] = (
    theta0
    + theta1 * lag_weights["lag"]
    + theta2 * lag_weights["lag"] ** 2
)

lag_weights


# In[19]:


# Plot implied lag coefficients
plt.figure(figsize=(8, 5))
plt.plot(lag_weights["lag"], lag_weights["implied_beta"], marker="o")
plt.axhline(0, linestyle="--")
plt.title("Implied Starts-to-Completions Lag Coefficients")
plt.xlabel("Lag in quarters")
plt.ylabel("Implied elasticity")
plt.xticks(list(lags))
plt.tight_layout()
plt.show()


# In[20]:


# Compare the three log models
log_model_comparison = pd.DataFrame({
    "Model": [
        "Log lag-4 benchmark",
        "Unrestricted log distributed lag",
        "Almon polynomial distributed lag"
    ],
    "Observations": [
        int(log_simple_model.nobs),
        int(log_full_model.nobs),
        int(almon_model.nobs)
    ],
    "R_squared": [
        log_simple_model.rsquared,
        log_full_model.rsquared,
        almon_model.rsquared
    ],
    "Adj_R_squared": [
        log_simple_model.rsquared_adj,
        log_full_model.rsquared_adj,
        almon_model.rsquared_adj
    ],
    "AIC": [
        log_simple_model.aic,
        log_full_model.aic,
        almon_model.aic
    ],
    "BIC": [
        log_simple_model.bic,
        log_full_model.bic,
        almon_model.bic
    ]
})

log_model_comparison


# In[21]:


# Generate predicted log completions
complex_df["predicted_ln_C_almon"] = almon_model.predict(complex_df)

# Convert predicted log completions back to levels
complex_df["predicted_C_almon"] = np.exp(complex_df["predicted_ln_C_almon"])

# Convert actual log completions back to levels
complex_df["actual_C"] = np.exp(complex_df["ln_C"])

# Convert quarterly index to timestamp for plotting
complex_df["date"] = complex_df.index.to_timestamp()

# Plot actual and predicted completions
plt.figure(figsize=(12, 6))
plt.plot(complex_df["date"], complex_df["actual_C"], label="Actual completions")
plt.plot(complex_df["date"], complex_df["predicted_C_almon"], label="Predicted completions")
plt.title("Actual and Predicted Housing Completions: Almon Distributed Lag Model")
plt.xlabel("Year")
plt.ylabel("Dwellings")
plt.legend()
plt.tight_layout()
plt.show()


# In[22]:


# Extract regression results from the Almon polynomial distributed lag model
almon_results_table = pd.DataFrame({
    "Variable": almon_model.params.index,
    "Coefficient": almon_model.params.values,
    "Std_Error": almon_model.bse.values,
    "P_Value": almon_model.pvalues.values
})

almon_results_table


# In[23]:


# Plot implied starts-to-completions lag coefficients
plt.figure(figsize=(8, 5))
plt.plot(lag_weights["lag"], lag_weights["implied_beta"], marker="o")
plt.axhline(0, linestyle="--")
plt.title("Implied Starts-to-Completions Lag Coefficients")
plt.xlabel("Lag in quarters")
plt.ylabel("Implied elasticity")
plt.xticks(lag_weights["lag"])
plt.tight_layout()
plt.show()


# In[24]:


# Generate predicted log completions from the Almon model
complex_df["predicted_ln_C_almon"] = almon_model.predict(complex_df)

# Apply smearing correction when converting log predictions back to levels
smearing_factor = np.mean(np.exp(almon_model.resid))

complex_df["predicted_C_almon"] = np.exp(complex_df["predicted_ln_C_almon"]) * smearing_factor
complex_df["actual_C"] = np.exp(complex_df["ln_C"])

# Convert quarterly index to timestamp for plotting
complex_df["date"] = complex_df.index.to_timestamp()

# Plot actual and predicted completions
plt.figure(figsize=(12, 6))
plt.plot(complex_df["date"], complex_df["actual_C"], label="Actual completions")
plt.plot(complex_df["date"], complex_df["predicted_C_almon"], label="Predicted completions")
plt.title("Actual and Predicted Housing Completions: Almon Distributed Lag Model")
plt.xlabel("Year")
plt.ylabel("Dwellings")
plt.legend()
plt.tight_layout()
plt.show()


# In[25]:


# Create an output folder for tables and figures
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_DIR


# In[26]:


# Save model comparison table
log_model_comparison.to_excel(
    OUTPUT_DIR / "log_model_comparison.xlsx",
    index=False
)

# Save Almon regression results
almon_results_table.to_excel(
    OUTPUT_DIR / "almon_model_results.xlsx",
    index=False
)

# Save implied lag coefficients
lag_weights.to_excel(
    OUTPUT_DIR / "almon_implied_lag_coefficients.xlsx",
    index=False
)


# In[27]:


# Save implied lag coefficients figure
plt.figure(figsize=(8, 5))
plt.plot(lag_weights["lag"], lag_weights["implied_beta"], marker="o")
plt.axhline(0, linestyle="--")
plt.title("Implied Starts-to-Completions Lag Coefficients")
plt.xlabel("Lag in quarters")
plt.ylabel("Implied elasticity")
plt.xticks(lag_weights["lag"])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "implied_lag_coefficients.png", dpi=300)
plt.show()


# In[28]:


# Save actual vs predicted completions figure
plt.figure(figsize=(12, 6))
plt.plot(complex_df["date"], complex_df["actual_C"], label="Actual completions")
plt.plot(complex_df["date"], complex_df["predicted_C_almon"], label="Predicted completions")
plt.title("Actual and Predicted Housing Completions: Almon Distributed Lag Model")
plt.xlabel("Year")
plt.ylabel("Dwellings")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted_completions_almon.png", dpi=300)
plt.show()


# In[61]:


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


# In[29]:


# Load LiveTable104 dwelling stock data
stock_raw = pd.read_excel(
    stock_file,
    sheet_name="LT_104",
    header=4,
    engine="odf"
)

stock_raw.head()


# In[30]:


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


# In[31]:


stock.tail()


# In[32]:


# Plot dwelling stock over time
plt.figure(figsize=(12, 6))
plt.plot(stock["year"], stock["dwelling_stock"])
plt.title("Dwelling Stock in England")
plt.xlabel("Year")
plt.ylabel("Number of dwellings")
plt.tight_layout()
plt.show()


# In[33]:


# Plot approximate net additions using annual change in dwelling stock
plt.figure(figsize=(12, 6))
plt.plot(stock["year"], stock["approx_net_additions"])
plt.title("Approximate Net Additions Using Change in Dwelling Stock")
plt.xlabel("Year")
plt.ylabel("Annual change in dwelling stock")
plt.tight_layout()
plt.show()


# In[34]:


# Save dwelling stock figure
plt.figure(figsize=(12, 6))
plt.plot(stock["year"], stock["dwelling_stock"])
plt.title("Dwelling Stock in England")
plt.xlabel("Year")
plt.ylabel("Number of dwellings")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dwelling_stock_england.png", dpi=300)
plt.show()


# In[35]:


# Save approximate net additions figure
plt.figure(figsize=(12, 6))
plt.plot(stock["year"], stock["approx_net_additions"])
plt.title("Approximate Net Additions Using Change in Dwelling Stock")
plt.xlabel("Year")
plt.ylabel("Annual change in dwelling stock")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "approx_net_additions_stock_change.png", dpi=300)
plt.show()


# In[36]:


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


# In[37]:


# Merge annual completions with dwelling stock change proxy
annual_bridge = pd.merge(
    annual_completions,
    stock[["year", "dwelling_stock", "approx_net_additions"]],
    on="year",
    how="inner"
)

annual_bridge.head()


# In[38]:


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


# In[39]:


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


# In[40]:


# Save cleaned datasets
complex_df.to_excel(
    OUTPUT_DIR / "cleaned_quarterly_model_data.xlsx"
)

stock.to_excel(
    OUTPUT_DIR / "dwelling_stock_proxy_data.xlsx",
    index=False
)

annual_bridge.to_excel(
    OUTPUT_DIR / "annual_completions_net_supply_bridge.xlsx",
    index=False
)


# In[62]:


import matplotlib.pyplot as plt

try:
    OUTPUT_DIR
except NameError:
    OUTPUT_DIR = DATA_DIR / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")

formula_text = r"""
Private Enterprise Starts-to-Completions Model

Because the main group model forecasts private enterprise starts,
the same Almon distributed lag framework is estimated for private enterprise housing.

$\ln(C^{private}_t) = \alpha + \rho \ln(C^{private}_{t-1}) + \theta_0 Z^{private}_{0t} + \theta_1 Z^{private}_{1t} + \theta_2 Z^{private}_{2t} + Q_t + \varepsilon_t$

Private Almon lag terms:

$Z^{private}_{0t} = \sum_{j=4}^{8} \ln(S^{private}_{t-j})$

$Z^{private}_{1t} = \sum_{j=4}^{8} j \ln(S^{private}_{t-j})$

$Z^{private}_{2t} = \sum_{j=4}^{8} j^2 \ln(S^{private}_{t-j})$

Definitions:

$C^{private}_t$ = private enterprise completions in quarter $t$

$S^{private}_{t-j}$ = private enterprise starts lagged by $j$ quarters

$j = 4,\ldots,8$ = construction lag from starts to completions

$Q_t$ = quarterly seasonal dummy variables
"""

ax.text(0.02, 0.98, formula_text, fontsize=15.5, va="top", ha="left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "formula_3_private_enterprise_model.png", dpi=300, bbox_inches="tight")
plt.show()


# In[41]:


# Create a dataset for private enterprise starts and completions
private_df = df.copy()

# Keep only positive observations because log transformation requires positive values
private_df = private_df[
    (private_df["starts_private"] > 0) &
    (private_df["completions_private"] > 0)
].copy()

# Create log variables
private_df["ln_C_private"] = np.log(private_df["completions_private"])
private_df["ln_S_private"] = np.log(private_df["starts_private"])

# Add lagged private completions
private_df["ln_C_private_lag1"] = private_df["ln_C_private"].shift(1)

# Add lagged private starts from 4 to 8 quarters
for lag in range(4, 9):
    private_df[f"ln_S_private_lag{lag}"] = private_df["ln_S_private"].shift(lag)

# Add quarter number for seasonal controls
private_df["quarter_num"] = private_df.index.quarter

# Drop missing observations
private_df = private_df.dropna()

# Define lags
private_lags = range(4, 9)

# Create Almon polynomial distributed lag variables
private_df["Z0_private"] = sum(private_df[f"ln_S_private_lag{lag}"] for lag in private_lags)
private_df["Z1_private"] = sum(lag * private_df[f"ln_S_private_lag{lag}"] for lag in private_lags)
private_df["Z2_private"] = sum((lag ** 2) * private_df[f"ln_S_private_lag{lag}"] for lag in private_lags)

# Define the private enterprise Almon model
private_almon_formula = """
ln_C_private ~ ln_C_private_lag1
+ Z0_private + Z1_private + Z2_private
+ C(quarter_num)
"""

# Estimate the model with Newey-West standard errors
private_almon_model = smf.ols(private_almon_formula, data=private_df).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 4}
)

# Print the regression summary
print(private_almon_model.summary())


# In[42]:


# Recover implied lag coefficients from the private enterprise Almon model
theta0_p = private_almon_model.params["Z0_private"]
theta1_p = private_almon_model.params["Z1_private"]
theta2_p = private_almon_model.params["Z2_private"]

private_lag_weights = pd.DataFrame({
    "lag": list(private_lags)
})

private_lag_weights["implied_beta"] = (
    theta0_p
    + theta1_p * private_lag_weights["lag"]
    + theta2_p * private_lag_weights["lag"] ** 2
)

private_lag_weights


# In[63]:


import matplotlib.pyplot as plt

try:
    OUTPUT_DIR
except NameError:
    OUTPUT_DIR = DATA_DIR / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")

formula_text = r"""
Recovering Implied Lag Coefficients

In the Almon distributed lag model, each starts lag coefficient is recovered as:

$\beta_j = \theta_0 + \theta_1 j + \theta_2 j^2$

for:

$j = 4,5,6,7,8$

Definitions:

$\beta_j$ = implied elasticity of completions with respect to starts lagged by $j$ quarters

$\theta_0, \theta_1, \theta_2$ = estimated Almon polynomial parameters

The implied coefficients show how the effect of housing starts is distributed across construction lags.
"""

ax.text(0.02, 0.98, formula_text, fontsize=16, va="top", ha="left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "formula_4_implied_lag_coefficients.png", dpi=300, bbox_inches="tight")
plt.show()


# In[43]:


# Plot implied private starts-to-completions lag coefficients
plt.figure(figsize=(8, 5))
plt.plot(private_lag_weights["lag"], private_lag_weights["implied_beta"], marker="o")
plt.axhline(0, linestyle="--")
plt.title("Implied Private Starts-to-Completions Lag Coefficients")
plt.xlabel("Lag in quarters")
plt.ylabel("Implied elasticity")
plt.xticks(private_lag_weights["lag"])
plt.tight_layout()
plt.show()


# In[44]:


# Generate predicted log private completions
private_df["predicted_ln_C_private"] = private_almon_model.predict(private_df)

# Apply smearing correction
private_smearing_factor = np.mean(np.exp(private_almon_model.resid))

# Convert predicted log completions back to levels
private_df["predicted_C_private"] = np.exp(private_df["predicted_ln_C_private"]) * private_smearing_factor
private_df["actual_C_private"] = np.exp(private_df["ln_C_private"])

# Convert quarterly index to timestamp
private_df["date"] = private_df.index.to_timestamp()

# Plot actual and predicted private completions
plt.figure(figsize=(12, 6))
plt.plot(private_df["date"], private_df["actual_C_private"], label="Actual private completions")
plt.plot(private_df["date"], private_df["predicted_C_private"], label="Predicted private completions")
plt.title("Actual and Predicted Private Enterprise Completions: Almon Distributed Lag Model")
plt.xlabel("Year")
plt.ylabel("Dwellings")
plt.legend()
plt.tight_layout()
plt.show()


# In[45]:


# Extract regression results from the private enterprise Almon model
private_almon_results_table = pd.DataFrame({
    "Variable": private_almon_model.params.index,
    "Coefficient": private_almon_model.params.values,
    "Std_Error": private_almon_model.bse.values,
    "P_Value": private_almon_model.pvalues.values
})

private_almon_results_table


# In[46]:


# Save private enterprise Almon model results
private_almon_results_table.to_excel(
    OUTPUT_DIR / "private_almon_model_results.xlsx",
    index=False
)

# Save private implied lag coefficients
private_lag_weights.to_excel(
    OUTPUT_DIR / "private_implied_lag_coefficients.xlsx",
    index=False
)


# In[47]:


# Combine all-dwellings and private enterprise lag weights
combined_lag_weights = pd.merge(
    lag_weights.rename(columns={"implied_beta": "all_dwellings_beta"}),
    private_lag_weights.rename(columns={"implied_beta": "private_enterprise_beta"}),
    on="lag",
    how="inner"
)

combined_lag_weights


# In[48]:


# Plot implied lag coefficients for all dwellings and private enterprise
plt.figure(figsize=(8, 5))
plt.plot(
    combined_lag_weights["lag"],
    combined_lag_weights["all_dwellings_beta"],
    marker="o",
    label="All dwellings"
)
plt.plot(
    combined_lag_weights["lag"],
    combined_lag_weights["private_enterprise_beta"],
    marker="o",
    label="Private enterprise"
)

plt.axhline(0, linestyle="--")
plt.title("Implied Starts-to-Completions Lag Coefficients")
plt.xlabel("Lag in quarters")
plt.ylabel("Implied elasticity")
plt.xticks(combined_lag_weights["lag"])
plt.legend()
plt.tight_layout()
plt.show()


# In[50]:


# Save combined lag coefficients figure
plt.figure(figsize=(8, 5))
plt.plot(
    combined_lag_weights["lag"],
    combined_lag_weights["all_dwellings_beta"],
    marker="o",
    label="All dwellings"
)
plt.plot(
    combined_lag_weights["lag"],
    combined_lag_weights["private_enterprise_beta"],
    marker="o",
    label="Private enterprise"
)

plt.axhline(0, linestyle="--")
plt.title("Implied Starts-to-Completions Lag Coefficients")
plt.xlabel("Lag in quarters")
plt.ylabel("Implied elasticity")
plt.xticks(combined_lag_weights["lag"])
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "combined_implied_lag_coefficients.png", dpi=300)
plt.show()


# In[51]:


# Save actual vs predicted private enterprise completions figure
plt.figure(figsize=(12, 6))
plt.plot(private_df["date"], private_df["actual_C_private"], label="Actual private completions")
plt.plot(private_df["date"], private_df["predicted_C_private"], label="Predicted private completions")
plt.title("Actual and Predicted Private Enterprise Completions: Almon Distributed Lag Model")
plt.xlabel("Year")
plt.ylabel("Dwellings")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted_private_completions_almon.png", dpi=300)
plt.show()


# In[66]:


# Drop-out rate scenario adjustment
# This adjusts predicted private completions under different drop-out assumptions

dropout_scenarios = {
    "Low drop-out (2%)": 0.02,
    "Baseline drop-out (5%)": 0.05,
    "High drop-out (8%)": 0.08
}

dropout_df = private_df[[
    "date",
    "predicted_C_private"
]].copy()

for scenario, rate in dropout_scenarios.items():
    dropout_df[scenario] = dropout_df["predicted_C_private"] * (1 - rate)

dropout_df.tail()


# In[67]:


# Plot drop-out adjusted completions scenarios

plt.figure(figsize=(12, 6))

plt.plot(
    dropout_df["date"],
    dropout_df["predicted_C_private"],
    label="Predicted private completions before drop-out adjustment"
)

for scenario in dropout_scenarios.keys():
    plt.plot(
        dropout_df["date"],
        dropout_df[scenario],
        label=scenario
    )

plt.title("Drop-out Adjusted Private Enterprise Completions")
plt.xlabel("Year")
plt.ylabel("Dwellings")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dropout_adjusted_private_completions.png", dpi=300)
plt.show()


# In[68]:


dropout_df.to_excel(
    OUTPUT_DIR / "dropout_adjusted_private_completions.xlsx",
    index=False
)


# In[52]:


# Create a final summary table for the main and private models
final_model_summary = pd.DataFrame({
    "Model": [
        "All dwellings Almon model",
        "Private enterprise Almon model"
    ],
    "Dependent_variable": [
        "log all completions",
        "log private enterprise completions"
    ],
    "Observations": [
        int(almon_model.nobs),
        int(private_almon_model.nobs)
    ],
    "R_squared": [
        almon_model.rsquared,
        private_almon_model.rsquared
    ],
    "Adj_R_squared": [
        almon_model.rsquared_adj,
        private_almon_model.rsquared_adj
    ],
    "AIC": [
        almon_model.aic,
        private_almon_model.aic
    ],
    "BIC": [
        almon_model.bic,
        private_almon_model.bic
    ],
    "Lag_4_beta": [
        lag_weights.loc[lag_weights["lag"] == 4, "implied_beta"].iloc[0],
        private_lag_weights.loc[private_lag_weights["lag"] == 4, "implied_beta"].iloc[0]
    ],
    "Lag_8_beta": [
        lag_weights.loc[lag_weights["lag"] == 8, "implied_beta"].iloc[0],
        private_lag_weights.loc[private_lag_weights["lag"] == 8, "implied_beta"].iloc[0]
    ]
})

final_model_summary


# In[53]:


# Save final model summary
final_model_summary.to_excel(
    OUTPUT_DIR / "final_model_summary.xlsx",
    index=False
)


# In[65]:


import matplotlib.pyplot as plt

try:
    OUTPUT_DIR
except NameError:
    OUTPUT_DIR = DATA_DIR / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")

# Title
ax.text(
    0.02, 0.96,
    "Forecast Bridge from Starts to Completions",
    fontsize=20,
    fontweight="bold",
    va="top"
)

# Explanation
ax.text(
    0.02, 0.88,
    "The main group model forecasts future private enterprise starts. "
    "This bridge converts predicted starts into expected private enterprise completions.",
    fontsize=15,
    va="top"
)

# Define notation
ax.text(
    0.02, 0.78,
    r"Let $\ell_t^p = \ln(C_t^p)$ and $x_t^p = \ln(S_t^p)$",
    fontsize=18,
    va="top"
)

# Main forecast equation
ax.text(
    0.02, 0.68,
    r"$\widehat{\ell}_{t+h}^p"
    r"= \alpha + \rho \ell_{t+h-1}^p"
    r"+ \theta_0 Z_{0,t+h}^p"
    r"+ \theta_1 Z_{1,t+h}^p"
    r"+ \theta_2 Z_{2,t+h}^p"
    r"+ Q_{t+h}$",
    fontsize=19,
    va="top"
)

# Z terms
ax.text(
    0.02, 0.55,
    r"$Z_{0,t+h}^p = \sum_{j=4}^{8} \widehat{x}_{t+h-j}^p$",
    fontsize=18,
    va="top"
)

ax.text(
    0.02, 0.46,
    r"$Z_{1,t+h}^p = \sum_{j=4}^{8} j \widehat{x}_{t+h-j}^p$",
    fontsize=18,
    va="top"
)

ax.text(
    0.02, 0.37,
    r"$Z_{2,t+h}^p = \sum_{j=4}^{8} j^2 \widehat{x}_{t+h-j}^p$",
    fontsize=18,
    va="top"
)

# Convert back to levels
ax.text(
    0.02, 0.26,
    r"$\widehat{C}_{t+h}^p = \exp(\widehat{\ell}_{t+h}^p) \times \mathrm{smearing\ factor}$",
    fontsize=18,
    va="top"
)

# Interpretation
ax.text(
    0.02, 0.15,
    "Interpretation: a forecast of private starts can be translated into a forecast of private completions. "
    "The estimated lag profile suggests that the strongest effect occurs around four quarters later.",
    fontsize=15,
    va="top"
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "formula_5_forecast_bridge_clear.png", dpi=300, bbox_inches="tight")
plt.show()


# In[3]:


from pathlib import Path
from IPython.display import display, Image

folders = [
    Path(r"D:\data\outputs"),
    Path(r"D:\data\outputs_part6")
]

for folder in folders:
    print("\nFolder:", folder)
    print("=" * 80)
    
    for file in sorted(folder.glob("*.png")):
        print(file.name)
        display(Image(filename=str(file), width=600))
    


# In[2]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

DATA_DIR = Path(r"D:\data")
housebuilding_file = DATA_DIR / "indicatorsofukhousebuilding.xlsx"

# Read sheet 1b
raw = pd.read_excel(
    housebuilding_file,
    sheet_name="1b",
    header=5
)

# Rename columns
df = raw.rename(columns={
    "Period": "period",
    "Started - All Dwellings": "starts_all",
    "Completed - All Dwellings": "completions_all",
    "Started - Private Enterprise": "starts_private",
    "Completed - Private Enterprise": "completions_private"
})

# Keep needed columns
df = df[[
    "period",
    "starts_all",
    "completions_all",
    "starts_private",
    "completions_private"
]].copy()

# Convert numeric columns
for col in ["starts_all", "completions_all", "starts_private", "completions_private"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert period to quarter
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

    return pd.Period(year=year, quarter=q, freq="Q")

df["quarter"] = df["period"].apply(parse_quarter)

# Drop invalid rows and set index
df = df.dropna(subset=["quarter"])
df = df.set_index("quarter").sort_index()

# Check data
print(df.head())
print(df.tail())
print(df.index.min(), df.index.max())


# In[3]:


# ==============================
# New Model: Dynamic ARDL Bridge Model
# Private Starts to Private Completions
# ==============================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Create modelling dataset from cleaned quarterly data
new_df = df.copy()

# Keep only positive values because we use logs
new_df = new_df[
    (new_df["starts_private"] > 0) &
    (new_df["completions_private"] > 0)
].copy()

# Log variables
new_df["ln_C_private"] = np.log(new_df["completions_private"])
new_df["ln_S_private"] = np.log(new_df["starts_private"])

# Lagged completions: dynamic persistence
new_df["ln_C_private_lag1"] = new_df["ln_C_private"].shift(1)
new_df["ln_C_private_lag2"] = new_df["ln_C_private"].shift(2)

# Starts lagged by 5 quarters: construction delay
new_df["ln_S_private_lag5"] = new_df["ln_S_private"].shift(5)

# Seasonal controls
new_df["quarter_num"] = new_df.index.quarter

# Structural shock controls: Global Financial Crisis
new_df["gfc_2008_2009"] = (
    (new_df.index >= pd.Period("2008Q3", freq="Q")) &
    (new_df.index <= pd.Period("2009Q4", freq="Q"))
).astype(int)

# Structural shock controls: COVID period
new_df["covid_2020Q2"] = (new_df.index == pd.Period("2020Q2", freq="Q")).astype(int)
new_df["covid_2020Q3"] = (new_df.index == pd.Period("2020Q3", freq="Q")).astype(int)
new_df["covid_2020Q4"] = (new_df.index == pd.Period("2020Q4", freq="Q")).astype(int)

# Drop missing observations caused by lags
new_df = new_df.dropna()

# Check the dataset
print(new_df.head())
print(new_df.tail())
print("Number of observations:", len(new_df))


# In[4]:


# Define the preferred dynamic bridge model
preferred_formula = """
ln_C_private ~ ln_C_private_lag1
+ ln_C_private_lag2
+ ln_S_private_lag5
+ C(quarter_num)
+ gfc_2008_2009
+ covid_2020Q2 + covid_2020Q3 + covid_2020Q4
"""

# Estimate the model with Newey-West / HAC standard errors
preferred_model = smf.ols(
    preferred_formula,
    data=new_df
).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 4}
)

print(preferred_model.summary())


# In[7]:


from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

resid = preferred_model.resid

# ==============================
# 1. Ljung-Box test: residual autocorrelation
# ==============================
ljung_box = acorr_ljungbox(
    resid,
    lags=[4, 8, 12],
    return_df=True
)

print("Ljung-Box Test:")
print(ljung_box)

# ==============================
# 2. Breusch-Pagan test: heteroskedasticity
# ==============================
bp_test = het_breuschpagan(
    resid,
    preferred_model.model.exog
)

bp_results = pd.Series(
    bp_test,
    index=["LM statistic", "LM p-value", "F statistic", "F p-value"]
)

print("\nBreusch-Pagan Test:")
print(bp_results)

# ==============================
# 3. Jarque-Bera test: residual normality
# ==============================
jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(resid)

print("\nJarque-Bera Test:")
print("JB statistic:", jb_stat)
print("p-value:", jb_pvalue)
print("skew:", skew)
print("kurtosis:", kurtosis)

# ==============================
# 4. Manual Ramsey RESET test
# ==============================

reset_df = new_df.copy()

# Add squared fitted values manually
reset_df["fitted_sq"] = np.asarray(preferred_model.fittedvalues) ** 2

reset_formula = """
ln_C_private ~ ln_C_private_lag1
+ ln_C_private_lag2
+ ln_S_private_lag5
+ C(quarter_num)
+ gfc_2008_2009
+ covid_2020Q2 + covid_2020Q3 + covid_2020Q4
+ fitted_sq
"""

reset_model = smf.ols(
    reset_formula,
    data=reset_df
).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 4}
)

reset_test = reset_model.f_test("fitted_sq = 0")

print("\nManual Ramsey RESET Test:")
print(reset_test)


# In[8]:


import matplotlib.pyplot as plt
import numpy as np

# Generate predicted log completions
new_df["predicted_ln_C_private"] = preferred_model.predict(new_df)

# Smearing correction for converting log predictions back to levels
smearing_factor = np.mean(np.exp(preferred_model.resid))

# Convert back to levels
new_df["predicted_C_private"] = np.exp(new_df["predicted_ln_C_private"]) * smearing_factor
new_df["actual_C_private"] = np.exp(new_df["ln_C_private"])

# Date for plotting
new_df["date"] = new_df.index.to_timestamp()

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(new_df["date"], new_df["actual_C_private"], label="Actual private completions")
plt.plot(new_df["date"], new_df["predicted_C_private"], label="Predicted private completions")
plt.title("Actual and Predicted Private Enterprise Completions")
plt.xlabel("Year")
plt.ylabel("Dwellings")
plt.legend()
plt.tight_layout()
plt.show()


# In[9]:


plt.figure(figsize=(12, 6))
plt.plot(new_df["date"], new_df["actual_C_private"], label="Actual private completions")
plt.plot(new_df["date"], new_df["predicted_C_private"], label="Predicted private completions")
plt.title("Actual and Predicted Private Enterprise Completions")
plt.xlabel("Year")
plt.ylabel("Dwellings")
plt.legend()
plt.tight_layout()
plt.savefig(r"D:\data\outputs\preferred_actual_vs_predicted_private_completions.png", dpi=300)
plt.show()


# In[10]:


preferred_summary = pd.DataFrame({
    "Model": ["Dynamic ARDL bridge model"],
    "Dependent variable": ["log private enterprise completions"],
    "Observations": [int(preferred_model.nobs)],
    "R_squared": [preferred_model.rsquared],
    "Adj_R_squared": [preferred_model.rsquared_adj],
    "AIC": [preferred_model.aic],
    "BIC": [preferred_model.bic]
})

preferred_summary


# In[11]:


from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path(r"D:\data\outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Save coefficient table
preferred_results_table = pd.DataFrame({
    "Variable": preferred_model.params.index,
    "Coefficient": preferred_model.params.values,
    "Std_Error": preferred_model.bse.values,
    "P_Value": preferred_model.pvalues.values
})

preferred_results_table.to_excel(
    OUTPUT_DIR / "preferred_dynamic_bridge_model_results.xlsx",
    index=False
)

# Save model summary table
preferred_summary = pd.DataFrame({
    "Model": ["Dynamic ARDL bridge model"],
    "Dependent variable": ["log private enterprise completions"],
    "Observations": [int(preferred_model.nobs)],
    "R_squared": [preferred_model.rsquared],
    "Adj_R_squared": [preferred_model.rsquared_adj],
    "AIC": [preferred_model.aic],
    "BIC": [preferred_model.bic]
})

preferred_summary.to_excel(
    OUTPUT_DIR / "preferred_dynamic_bridge_model_summary.xlsx",
    index=False
)

preferred_results_table, preferred_summary


# In[12]:


diagnostic_summary = pd.DataFrame({
    "Test": [
        "Ljung-Box lag 4",
        "Ljung-Box lag 8",
        "Ljung-Box lag 12",
        "Breusch-Pagan LM",
        "Breusch-Pagan F",
        "Jarque-Bera",
        "Ramsey RESET"
    ],
    "P_value": [
        0.194130,
        0.085900,
        0.125622,
        0.918539,
        0.925131,
        0.922546,
        0.895120
    ],
    "Conclusion": [
        "Pass: no strong autocorrelation",
        "Pass / borderline acceptable",
        "Pass: no strong autocorrelation",
        "Pass: no heteroskedasticity",
        "Pass: no heteroskedasticity",
        "Pass: residuals approximately normal",
        "Pass: no major misspecification"
    ]
})

diagnostic_summary.to_excel(
    OUTPUT_DIR / "preferred_model_diagnostic_tests.xlsx",
    index=False
)

diagnostic_summary


# In[13]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(new_df["date"], new_df["actual_C_private"], label="Actual private completions")
plt.plot(new_df["date"], new_df["predicted_C_private"], label="Predicted private completions")
plt.title("Actual and Predicted Private Enterprise Completions")
plt.xlabel("Year")
plt.ylabel("Dwellings")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "preferred_actual_vs_predicted_private_completions.png", dpi=300)
plt.show()


# In[14]:


import matplotlib.pyplot as plt

try:
    OUTPUT_DIR
except NameError:
    OUTPUT_DIR = DATA_DIR / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")

# Title
ax.text(
    0.02, 0.96,
    "Forecast Bridge from Starts to Completions",
    fontsize=20,
    fontweight="bold",
    va="top"
)

# Purpose
ax.text(
    0.02, 0.88,
    "The main group model forecasts future private enterprise starts. "
    "This dynamic bridge model converts predicted starts into expected private enterprise completions.",
    fontsize=15,
    va="top"
)

# Main model equation
ax.text(
    0.02, 0.76,
    r"$\ln(C_t^p) = \alpha + \phi_1 \ln(C_{t-1}^p) + \phi_2 \ln(C_{t-2}^p)"
    r" + \beta \ln(S_{t-5}^p) + \gamma Q_t + \delta GFC_t$",
    fontsize=18,
    va="top"
)

ax.text(
    0.02, 0.67,
    r"$+ \eta_1 D_{2020Q2} + \eta_2 D_{2020Q3} + \eta_3 D_{2020Q4} + \varepsilon_t$",
    fontsize=18,
    va="top"
)

# Definitions
ax.text(
    0.02, 0.56,
    r"$C_t^p$ = private enterprise completions in quarter $t$",
    fontsize=16,
    va="top"
)

ax.text(
    0.02, 0.49,
    r"$C_{t-1}^p$ and $C_{t-2}^p$ = lagged private completions, capturing persistence",
    fontsize=16,
    va="top"
)

ax.text(
    0.02, 0.42,
    r"$S_{t-5}^p$ = private enterprise starts lagged by five quarters",
    fontsize=16,
    va="top"
)

ax.text(
    0.02, 0.35,
    r"$Q_t$ = quarterly seasonal dummy variables",
    fontsize=16,
    va="top"
)

ax.text(
    0.02, 0.28,
    r"$GFC_t$ = 2008--2009 financial crisis dummy",
    fontsize=16,
    va="top"
)

ax.text(
    0.02, 0.21,
    r"$D_{2020Q2}$, $D_{2020Q3}$, $D_{2020Q4}$ = COVID shock dummy variables",
    fontsize=16,
    va="top"
)

# Interpretation
ax.text(
    0.02, 0.12,
    "Interpretation: private starts are converted into private completions through a construction delay. "
    "The model also controls for completion persistence, seasonality, and major structural shocks.",
    fontsize=15,
    va="top"
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "formula_5_preferred_dynamic_bridge_model.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# In[16]:


# Re-create the old private enterprise Almon model for comparison

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Create private enterprise dataset
private_df = df.copy()

# Keep only positive observations because log transformation requires positive values
private_df = private_df[
    (private_df["starts_private"] > 0) &
    (private_df["completions_private"] > 0)
].copy()

# Create log variables
private_df["ln_C_private"] = np.log(private_df["completions_private"])
private_df["ln_S_private"] = np.log(private_df["starts_private"])

# Add lagged private completions
private_df["ln_C_private_lag1"] = private_df["ln_C_private"].shift(1)

# Add lagged private starts from 4 to 8 quarters
for lag in range(4, 9):
    private_df[f"ln_S_private_lag{lag}"] = private_df["ln_S_private"].shift(lag)

# Add quarter number for seasonal controls
private_df["quarter_num"] = private_df.index.quarter

# Drop missing observations
private_df = private_df.dropna()

# Define lags
private_lags = range(4, 9)

# Create old Almon polynomial distributed lag variables
private_df["Z0_private"] = sum(
    private_df[f"ln_S_private_lag{lag}"] for lag in private_lags
)

private_df["Z1_private"] = sum(
    lag * private_df[f"ln_S_private_lag{lag}"] for lag in private_lags
)

private_df["Z2_private"] = sum(
    (lag ** 2) * private_df[f"ln_S_private_lag{lag}"] for lag in private_lags
)

# Define the old private enterprise Almon model
private_almon_formula = """
ln_C_private ~ ln_C_private_lag1
+ Z0_private + Z1_private + Z2_private
+ C(quarter_num)
"""

# Estimate the old Almon model with Newey-West standard errors
private_almon_model = smf.ols(
    private_almon_formula,
    data=private_df
).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 4}
)

print(private_almon_model.summary())


# In[17]:


preferred_model_comparison = pd.DataFrame({
    "Model": [
        "Private enterprise Almon model",
        "Preferred dynamic ARDL bridge model"
    ],
    "Observations": [
        int(private_almon_model.nobs),
        int(preferred_model.nobs)
    ],
    "R_squared": [
        private_almon_model.rsquared,
        preferred_model.rsquared
    ],
    "Adj_R_squared": [
        private_almon_model.rsquared_adj,
        preferred_model.rsquared_adj
    ],
    "AIC": [
        private_almon_model.aic,
        preferred_model.aic
    ],
    "BIC": [
        private_almon_model.bic,
        preferred_model.bic
    ]
})

preferred_model_comparison


# In[18]:


preferred_model_comparison.to_excel(
    OUTPUT_DIR / "preferred_model_comparison.xlsx",
    index=False
)


# In[ ]:




