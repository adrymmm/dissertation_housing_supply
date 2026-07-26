import os
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import keras
from keras import layers, callbacks

# Set random seeds for reproducibility
np.random.seed(42)
keras.utils.set_random_seed(42)

# --- load + transform: single gdp_def deflator for both real series ---
MASTER_DIR = "../../data/python_master"
df_raw = pd.read_csv(f"{MASTER_DIR}/england_master.csv")
print(df_raw.columns.tolist())
# Rename columns to match R script
df_raw = df_raw.rename(columns={
    "Unnamed: 0": "date", "starts": "lhstarts", "hprice": "lrprc", 
    "cc": "lrcc", "vol": "lvol", "hstock": "lstock", "rate": "r3"
})

# Convert date column to datetime and sort
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw.sort_values('date').reset_index(drop=True)

# Cap the sample at 2025Q4 to match 04_chronos.py (drops 2026Q1) and
# 05_ARRF.py (explicit Period('2025Q4') boundary). Without this, an extra
# quarter in the raw CSV would silently make this test window longer than
# the other two models' and break the RMSE/DM comparison.
test_end_date = pd.to_datetime("2026-01-01")  # exclusive: drops 2026Q1 onward
df_raw = df_raw[df_raw['date'] < test_end_date].reset_index(drop=True)

# Feature engineering (logs, deflators, and seasonal dummies)
df_raw['lhstarts'] = np.log(df_raw['lhstarts'])
df_raw['lrprc']    = np.log((df_raw['lrprc'] / df_raw['gdp_def']) * 100)
df_raw['lrcc']     = np.log((df_raw['lrcc'] / df_raw['gdp_def']) * 100)
df_raw['lvol']     = np.log(df_raw['lvol'])
df_raw['lstock']   = np.log(df_raw['lstock'])

# Extract quarters and create seasonal dummies
quarters = df_raw['date'].dt.quarter
df_raw['d_Q1'] = (quarters == 1).astype(int)
df_raw['d_Q2'] = (quarters == 2).astype(int)
df_raw['d_Q3'] = (quarters == 3).astype(int)

target_var = "lhstarts"
continuous_vars = ["lrprc", "lvol", "r3", "lstock", "lrcc"]
dummy_vars = ["d_Q1", "d_Q2", "d_Q3"]
all_vars = [target_var] + continuous_vars + dummy_vars

# --- Gap-fill continuous series only (zoo::na.approx + na.locf equivalent) ---
# 1. Linear interpolation for internal NAs
# 2. Forward fill, then backward fill for remaining edge NAs
df_cont = df_raw[continuous_vars + [target_var]].interpolate(method='linear')
df_cont = df_cont.ffill().bfill()

# Recombine continuous features with exact dummy variables
df_full = pd.concat([df_cont, df_raw[dummy_vars]], axis=1)
dates = df_raw['date']

# --- Train/test split ---
lookback = 8
test_start_date = pd.to_datetime("2010-01-01")

# Train index mask
train_mask = dates < test_start_date
n_train = train_mask.sum()

# Calculate scaling parameters (mean and SD) ONLY from the training set
scale_params = {}
df_scaled = df_full.copy()

for v in [target_var] + continuous_vars:
    mean_val = df_full.loc[train_mask, v].mean()
    sd_val = df_full.loc[train_mask, v].std(ddof=1) # ddof=1 matches R's default sample SD
    scale_params[v] = {'mean': mean_val, 'sd': sd_val}
    df_scaled[v] = (df_full[v] - mean_val) / sd_val

# --- Build 3D sequences for LSTM ---
mat = df_scaled[all_vars].to_numpy()
tcol = all_vars.index(target_var)
n_seq = len(mat) - lookback

X = np.zeros((n_seq, lookback, mat.shape[1]))
y = np.zeros(n_seq)

for i in range(n_seq):
    X[i, :, :] = mat[i : i + lookback, :]
    y[i] = mat[i + lookback, tcol]

# Track dates corresponding to the targets
seq_dates = dates.iloc[lookback:].reset_index(drop=True)

# Split sequences into train and test sets
train_seq_mask = seq_dates < test_start_date
n_train_seq = train_seq_mask.sum()

X_train, y_train = X[:n_train_seq], y[:n_train_seq]
X_test, y_test = X[n_train_seq:], y[n_train_seq:]
test_dates = seq_dates.iloc[n_train_seq:].reset_index(drop=True)

print(f"Train seqs: {X_train.shape[0]} | Test seqs: {X_test.shape[0]} "
      f"| Test window: {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')}")

# --- Model Architecture ---
model = keras.Sequential([
    layers.LSTM(units=64, input_shape=(lookback, X_train.shape[2]), 
                dropout=0.1, recurrent_dropout=0.1, return_sequences=False),
    layers.Dense(units=1, activation="linear")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001), 
    loss="mean_squared_error"
)

# Callbacks
es = callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, verbose=0)
rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=0)

# Fit model
model.fit(
    X_train, y_train, 
    epochs=300, 
    batch_size=16, 
    validation_split=0.15,
    callbacks=[es, rl], 
    shuffle=False, 
    verbose=0
)

# --- Predictions and Inversion ---
pred_scaled = model.predict(X_test, verbose=0).flatten()

def inv_scale(x):
    return x * scale_params[target_var]['sd'] + scale_params[target_var]['mean']

pred = inv_scale(pred_scaled)
actual = inv_scale(y_test)

# --- Calculate Metrics ---
rmse_val = root_mean_squared_error(actual, pred)
mae_val = mean_absolute_error(actual, pred)
print(f"LSTM (Spec3: continuous + seasonal) log-scale RMSE={rmse_val:.4f} MAE={mae_val:.4f}")

# --- Save Outputs ---
# Generate "2010Q1" formatting for Quarters
quarters_formatted = test_dates.dt.year.astype(str) + "Q" + test_dates.dt.quarter.astype(str)

out = pd.DataFrame({
    "date": test_dates.dt.strftime('%Y-%m-%d'),
    "Quarter": quarters_formatted,
    "lstm": pred,
    "actual": actual
})

# SCHEME: "fixed" (West, 1996) -- trained once on pre-2010Q1 data, then
# evaluated across the whole test window without re-fitting on new quarters
# as they arrive. This differs from 05_ARRF.py, which uses a "recursive"
# scheme (refit at every origin). Kept fixed here for compute-cost reasons
# (re-running early-stopped training 64 times is expensive and adds its own
# instability); disclosed explicitly in the methods section rather than left
# implicit.
out.to_csv("data/outputs/forecasts/lstm_forecasts.csv", index=False)
print("Saved lstm_forecasts.csv")