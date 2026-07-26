import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

MASTER_DIR = "../../data/python_master"
df = pd.read_csv(f'{MASTER_DIR}/england_master.csv')
df.rename(columns={'Unnamed: 0': 'Quarter'}, inplace=True)
df['Quarter'] = pd.PeriodIndex(df['Quarter'], freq='Q')
df.set_index('Quarter', inplace=True)

# Transformations -- single deflator (gdp_def) for both real series, matching
# the rest of the horse race.
df['log_starts'] = np.log(df['starts'])
df['log_hprice'] = np.log(df['hprice'] / df['gdp_def'])
df['log_cc']     = np.log(df['cc'] / df['gdp_def'])
df['log_vol']    = np.log(df['vol'])

df['log_starts_lag1'] = df['log_starts'].shift(1)
df['log_starts_lag4'] = df['log_starts'].shift(4)
df['log_hprice_lag1'] = df['log_hprice'].shift(1)
df['log_cc_lag1']     = df['log_cc'].shift(1)
df['log_vol_lag1']    = df['log_vol'].shift(1)
df['rate_lag1']       = df['rate'].shift(1)
df['rate_lag4']       = df['rate'].shift(4)

df_clean = df.dropna()

target = 'log_starts'
features = [
    'log_starts_lag1', 'log_starts_lag4',
    'log_hprice_lag1', 'log_cc_lag1',
    'rate_lag1', 'rate_lag4', 'log_vol_lag1'
]

X = df_clean[features]
y = df_clean[target]

# Evaluation window -- match the R horse race (2010Q1-2025Q4) so RMSEs are
# directly comparable. test_idx are positional indices into df_clean.
start = pd.Period('2010Q1', freq='Q')
end   = pd.Period('2025Q4', freq='Q')
test_mask = (df_clean.index >= start) & (df_clean.index <= end)
test_idx  = np.where(test_mask)[0]

# --- tune hyperparameters ONCE on the pre-2010Q1 (training) data, then freeze them
# (refitting GridSearchCV per origin would pick a different model each step and
#  is needlessly slow; fixing the config mirrors auto_ardl choosing lag orders
#  once, then refitting those fixed orders per origin in the R script.)
pre_eval = df_clean.index < start
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [2, 5],
}
grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1,
)
grid.fit(X[pre_eval], y[pre_eval])
best_params = grid.best_params_
print("Frozen hyperparameters:", best_params)

# --- expanding window: refit the frozen config at every origin, 1-step ahead --
# SCHEME: "recursive" (West, 1996) -- the model is re-estimated from scratch at
# every test origin using all true data strictly before it. This differs from
# 06_LSTM.py, which uses a "fixed" scheme (trained once on pre-2010Q1 data, then
# evaluated across the whole test window without re-fitting). Both are valid
# choices; the asymmetry is disclosed in the methods section rather than hidden.
preds = np.empty(len(test_idx))
for i, t in enumerate(test_idx):
    rf = RandomForestRegressor(random_state=42, **best_params)
    rf.fit(X.iloc[:t], y.iloc[:t])          # all data strictly before target t
    preds[i] = rf.predict(X.iloc[[t]])[0]   # features at t are realised (lag1/lag4 known)

actual = y.iloc[test_idx].values

rmse = np.sqrt(mean_squared_error(actual, preds))
mae  = mean_absolute_error(actual, preds)
print(f"\nExpanding-window OOS ({start}-{end}):")
print(f"Log-Scale RMSE: {rmse:.4f}")
print(f"MAE:            {mae:.4f}")

# --- save for merge with the R forecasts and the other Python models ----------
out = pd.DataFrame({
    'Quarter': df_clean.index[test_idx].astype(str),
    'rf': preds,
    'actual': actual,
})
out.to_csv('data/outputs/forecasts/rf_forecasts.csv', index=False)
print("\nSaved rf_forecasts.csv")
print(out.to_string(index=False))
