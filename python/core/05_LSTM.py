# =============================================================================
# MULTIVARIATE LSTM — UK HOUSING SUPPLY FORECASTING
#   Spec 1: No dummies (baseline)
#   Spec 2: + 3 crisis dummies (covid pulse, crisis pulse, crisis step)
#   Spec 3: + 3 seasonal dummies (Q1, Q2, Q3; Q4 = reference)
#   Spec 4: + crisis dummies AND seasonal dummies (all 6)
# =============================================================================

# =============================================================================
# BLOCK 1: LOAD PACKAGES
# =============================================================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
import os

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")

# =============================================================================
# BLOCK 2: IMPORT AND RENAME COLUMNS
# =============================================================================
OUT_P = "data/outputs/lstm/"
# Update the path to your dataset if needed
file_path = "data/python_master/england_master.csv"

df_raw = pd.read_csv(file_path, index_col=None)
df_raw.head()
df_raw = df_raw.rename(columns={
    'Unnamed: 0': 'date',
    'starts': 'lhstarts',
    'hprice': 'lrprc',
    'cc': 'lrcc',
    'vol': 'lvol',
    'hstock': 'lstock',
    'rate': 'r3'
})

print("--- Raw Data Preview ---")
print(df_raw.head())

# =============================================================================
# BLOCK 3: PARSE DATES — "1975Q1" FORMAT (NO SPACE)
# =============================================================================
df_raw['date'] = pd.PeriodIndex(df_raw['date'], freq='Q').to_timestamp()
df_raw = df_raw.sort_values('date').reset_index(drop=True)

print("\n--- Date Range ---")
print(f"Start: {df_raw['date'].min().date()}")
print(f"End:   {df_raw['date'].max().date()}")
print(f"Total observations: {len(df_raw)}")

# =============================================================================
# BLOCK 3b: DEFLATE NOMINAL VARIABLES USING A SINGLE DEFLATOR (gdp_def)
# =============================================================================
df_raw['lrprc'] = (df_raw['lrprc'] / df_raw['gdp_def']) * 100
df_raw['lrcc'] = (df_raw['lrcc'] / df_raw['gdp_def']) * 100

print("\n--- After Deflation (using gdp_def for both series) ---")
print(df_raw[['date', 'lrprc', 'lrcc', 'gdp_def']].head())
#Dropping 2026Q1
df_raw = df_raw.iloc[:-1].reset_index(drop=True)

# =============================================================================
# BLOCK 4: LOG-TRANSFORM RAW VARIABLES
# =============================================================================
cols_to_log = ['lhstarts', 'lrprc', 'lrcc', 'lvol', 'lstock']
df_raw[cols_to_log] = np.log(df_raw[cols_to_log])

# =============================================================================
# BLOCK 5: CREATE ALL CANDIDATE DUMMY VARIABLES
# =============================================================================
df_raw['d_covid'] = ((df_raw['date'] >= '2020-01-01') & (df_raw['date'] <= '2020-09-30')).astype(int)
df_raw['d_crisis_pulse'] = ((df_raw['date'] >= '2022-10-01') & (df_raw['date'] <= '2023-06-30')).astype(int)
df_raw['d_crisis_step'] = (df_raw['date'] >= '2022-10-01').astype(int)

df_raw['cal_q'] = df_raw['date'].dt.quarter
df_raw['d_Q1'] = (df_raw['cal_q'] == 1).astype(int)
df_raw['d_Q2'] = (df_raw['cal_q'] == 2).astype(int)
df_raw['d_Q3'] = (df_raw['cal_q'] == 3).astype(int)
df_raw = df_raw.drop(columns=['cal_q'])

print("\n--- Dummy Variable Verification ---")
print(f"d_covid active quarters        : {df_raw['d_covid'].sum()}")
print(f"d_crisis_pulse active quarters : {df_raw['d_crisis_pulse'].sum()}")
print(f"d_crisis_step active quarters  : {df_raw['d_crisis_step'].sum()}")
print(f"d_Q1 active quarters           : {df_raw['d_Q1'].sum()}")
print(f"d_Q2 active quarters           : {df_raw['d_Q2'].sum()}")
print(f"d_Q3 active quarters           : {df_raw['d_Q3'].sum()}")

# =============================================================================
# BLOCK 6: HANDLE MISSING VALUES (CONTINUOUS VARS ONLY)
# =============================================================================
target_var = "lhstarts"
continuous_vars = ["lrprc", "lvol", "r3", "lstock", "lrcc"]
all_dummy_vars = ["d_covid", "d_crisis_pulse", "d_crisis_step", "d_Q1", "d_Q2", "d_Q3"]

# Interpolate and fill continuous variables
df_cont = df_raw[continuous_vars + [target_var]].interpolate(method='linear')
df_cont = df_cont.bfill().ffill()

# Combine back
df_full = df_raw[['date'] + all_dummy_vars].copy()
for col in df_cont.columns:
    df_full[col] = df_cont[col]

dates = df_full['date'].values
print("\n--- Missing Values After Imputation ---")
print(df_full.isna().sum())

# =============================================================================
# BLOCK 7: TRAIN / TEST SPLIT (anchored to match the R horse race, not 80/20)
# =============================================================================
TEST_START = pd.Timestamp("2010-01-01")  # matches eval0 in 07_horse_race.R

n_obs = len(df_full)
lookback = 8

hit = np.where(dates == TEST_START.to_numpy())[0]
if len(hit) != 1:
    raise ValueError(f"2010-01-01 not found exactly once in df_full['date'] "
                      f"(found {len(hit)} matches) -- check date construction")
n_train = int(hit[0])
n_test  = n_obs - n_train

print("\n--- Train / Test Split (shared across all 3 specs) ---")
print(f"Total observations : {n_obs}")
print(f"Training           : {n_train} | From {str(pd.to_datetime(dates[0]).date())} to {str(pd.to_datetime(dates[n_train-1]).date())}")
print(f"Testing            : {n_test}  | From {str(pd.to_datetime(dates[n_train]).date())} to {str(pd.to_datetime(dates[-1]).date())}")

test_dates = dates[n_train:]

# =============================================================================
# BLOCK 8: REUSABLE FUNCTION — RUN ONE LSTM SPECIFICATION
# =============================================================================
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_lstm_spec(spec_name, predictor_vars, df_full, target_var, continuous_vars, dates, n_train, n_obs, lookback):
    print(f"\n\n=====================================================")
    print(f"  RUNNING SPEC: {spec_name}")
    print(f"  Predictors ({len(predictor_vars)}): {', '.join(predictor_vars)}")
    print(f"=====================================================")
    
    all_vars = [target_var] + predictor_vars
    
    # Train-set parameters for standardization
    train_raw = df_full.iloc[:n_train]
    scale_params = {}
    vars_to_scale = [target_var] + continuous_vars
    
    for v in vars_to_scale:
        scale_params[v] = {
            'mean': train_raw[v].mean(),
            'sd': train_raw[v].std()
        }
    
    # Apply scaling
    df_scaled = df_full.copy()
    for v in vars_to_scale:
        df_scaled[v] = (df_full[v] - scale_params[v]['mean']) / scale_params[v]['sd']
    
    # Sequences function
    def create_sequences(data_matrix, target_col_idx, lookback):
        n = data_matrix.shape[0]
        n_features = data_matrix.shape[1]
        n_samples = n - lookback
        X = np.zeros((n_samples, lookback, n_features))
        y = np.zeros(n_samples)
        for i in range(n_samples):
            X[i] = data_matrix[i:(i + lookback), :]
            y[i] = data_matrix[i + lookback, target_col_idx]
        return X, y

    data_matrix = df_scaled[all_vars].values
    target_col_idx = all_vars.index(target_var)
    X_all, y_all = create_sequences(data_matrix, target_col_idx, lookback)
    
    n_train_seq = n_train - lookback
    
    X_train = X_all[:n_train_seq]
    y_train = y_all[:n_train_seq]
    
    X_test = X_all[n_train_seq:]
    y_test = y_all[n_train_seq:]
    
    n_features = X_train.shape[2]
    print(f"Input features: {n_features} | Train seqs: {X_train.shape[0]} | Test seqs: {X_test.shape[0]}")
    
    # Reset seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Build model
    model = Sequential()
    model.add(Input(shape=(lookback, X_train.shape[2])))
    model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=False))
    model.add(Dense(1, activation="linear"))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )
    
    early_stop = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=0)
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=300, batch_size=16,
        validation_split=0.15,
        callbacks=[early_stop, reduce_lr],
        shuffle=False, verbose=0
    )
    
    print(f"Epochs run: {len(history.history['loss'])}")
    
    # Predict and invert scaling
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_train_pred_scaled = model.predict(X_train, verbose=0).flatten()
    
    def invert_scale(scaled_vals, var_name, params):
        return scaled_vals * params[var_name]['sd'] + params[var_name]['mean']
    
    y_pred = invert_scale(y_pred_scaled, target_var, scale_params)
    y_true = invert_scale(y_test, target_var, scale_params)
    y_train_pred = invert_scale(y_train_pred_scaled, target_var, scale_params)
    y_train_true = invert_scale(y_train, target_var, scale_params)
    
    # Metrics
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_val = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    
    print(f"RMSE: {rmse_val:.4f} | MAE: {mae_val:.4f} | MAPE: {mape_val:.2f}% | R-squared: {r2_val:.4f}")
    
    return {
        'spec_name': spec_name,
        'y_true': y_true,
        'y_pred': y_pred,
        'rmse': rmse_val,
        'mae': mae_val,
        'mape': mape_val,
        'r2': r2_val,
        'history': history,
        'n_features': n_features,
        'model': model,
        'y_train_true': y_train_true,
        'y_train_pred': y_train_pred
    }

# =============================================================================
# BLOCK 9: DEFINE THE FOUR FEATURE SETS
# =============================================================================
predictors_spec1 = continuous_vars.copy()
predictors_spec2 = continuous_vars + ["d_covid", "d_crisis_pulse", "d_crisis_step"]
predictors_spec3 = continuous_vars + ["d_Q1", "d_Q2", "d_Q3"]
predictors_spec4 = continuous_vars + ["d_covid", "d_crisis_pulse", "d_crisis_step", "d_Q1", "d_Q2", "d_Q3"]

# =============================================================================
# BLOCK 10: RUN ALL FOUR SPECIFICATIONS
# =============================================================================
result1 = run_lstm_spec("Spec1_NoDummy", predictors_spec1, df_full, target_var, continuous_vars, dates, n_train, n_obs, lookback)
result2 = run_lstm_spec("Spec2_CrisisDummy", predictors_spec2, df_full, target_var, continuous_vars, dates, n_train, n_obs, lookback)
result3 = run_lstm_spec("Spec3_SeasonalDummy", predictors_spec3, df_full, target_var, continuous_vars, dates, n_train, n_obs, lookback)
result4 = run_lstm_spec("Spec4_CrisisSeasonal", predictors_spec4, df_full, target_var, continuous_vars, dates, n_train, n_obs, lookback)

# =============================================================================
# BLOCK 11: COMPARISON TABLE
# =============================================================================
comparison_table = pd.DataFrame({
    'Specification': [
        "Spec 1: No dummy",
        "Spec 2: + Crisis dummies (covid, crisis pulse, crisis step)",
        "Spec 3: + Seasonal dummies (Q1-Q3)",
        "Spec 4: + Crisis + Seasonal dummies"
    ],
    'Features': [result1['n_features'], result2['n_features'], result3['n_features'], result4['n_features']],
    'RMSE': [round(r['rmse'], 4) for r in [result1, result2, result3, result4]],
    'MAE': [round(r['mae'], 4) for r in [result1, result2, result3, result4]],
    'MAPE': [round(r['mape'], 2) for r in [result1, result2, result3, result4]],
    'R2': [round(r['r2'], 4) for r in [result1, result2, result3, result4]]
})

print("\n\n=========================================================")
print("  COMPARISON TABLE: 4 LSTM SPECIFICATIONS")
print("=========================================================")
print(comparison_table)
comparison_table.to_csv(f"{OUT_P}/lstm_spec_comparison.csv", index=False)

# =============================================================================
# BLOCKS 12, 13, 14: PLOTTING HELPERS
# =============================================================================
# Note: For brevity in Python translation, plots can be grouped and saved similarly. 
# Re-implementing ggplot visuals using seaborn/matplotlib.
def plot_avp(res1, label1, res2, label2, filename, title):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, res, label in zip(axes, [res1, res2], [label1, label2]):
        ax.plot(test_dates, res['y_true'], label="Actual", color="black", linestyle="-")
        ax.plot(test_dates, res['y_pred'], label="Predicted", color="steelblue", linestyle="--")
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel("Log Housing Starts")
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Date")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle(title)
    plt.tight_layout()
    plt.close()

def plot_resid(res1, label1, res2, label2, filename, title):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, res, label in zip(axes, [res1, res2], [label1, label2]):
        ax.plot(test_dates, res['y_true'] - res['y_pred'], color="darkred")
        ax.axhline(0, linestyle="--", color="grey")
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel("Residual")
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Date")
    fig.suptitle(title)
    plt.tight_layout()
    plt.close()

def plot_loss(res1, label1, res2, label2, filename, title):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    for ax, res, label in zip(axes, [res1, res2], [label1, label2]):
        ax.plot(res['history'].history['loss'], label="Training", color="steelblue")
        ax.plot(res['history'].history['val_loss'], label="Validation", color="tomato")
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel("MSE Loss")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    fig.suptitle(title)
    plt.tight_layout()
    plt.close()

# Plot Group A (1 vs 2)
plot_avp(result1, "Spec 1: No Dummy", result2, "Spec 2: + Crisis Dummies", "lstm_avp_groupA_nodummy_vs_crisis.png", "LSTM Forecast Comparison: No Dummy vs Crisis Dummies")
plot_resid(result1, "Spec 1: No Dummy", result2, "Spec 2: + Crisis Dummies", "lstm_resid_groupA_nodummy_vs_crisis.png", "LSTM Residuals: No Dummy vs Crisis Dummies")
plot_loss(result1, "Spec 1: No Dummy", result2, "Spec 2: + Crisis Dummies", "lstm_loss_groupA_nodummy_vs_crisis.png", "LSTM Training History: No Dummy vs Crisis Dummies")

# Plot Group B (3 vs 4)
plot_avp(result3, "Spec 3: + Seasonal Dummies", result4, "Spec 4: + Crisis + Seasonal Dummies", "lstm_avp_groupB_seasonal_vs_crisisseasonal.png", "LSTM Forecast Comparison: Seasonal vs Crisis + Seasonal Dummies")
plot_resid(result3, "Spec 3: + Seasonal Dummies", result4, "Spec 4: + Crisis + Seasonal Dummies", "lstm_resid_groupB_seasonal_vs_crisisseasonal.png", "LSTM Residuals: Seasonal vs Crisis + Seasonal Dummies")
plot_loss(result3, "Spec 3: + Seasonal Dummies", result4, "Spec 4: + Crisis + Seasonal Dummies", "lstm_loss_groupB_seasonal_vs_crisisseasonal.png", "LSTM Training History: Seasonal vs Crisis + Seasonal Dummies")

# =============================================================================
# BLOCK 15: SAVE INDIVIDUAL FORECASTS
# =============================================================================
def save_forecast(res, filename):
    df_out = pd.DataFrame({
        'date': test_dates,
        'actual': np.round(res['y_true'], 4),
        'lstm': np.round(res['y_pred'], 4),
        'residual': np.round(res['y_true'] - res['y_pred'], 4)
    })
    df_out.to_csv(filename, index=False)

save_forecast(result1, f"{OUT_P}/lstm_spec1_nodummy_forecasts.csv")
save_forecast(result2, f"{OUT_P}/lstm_spec2_crisis_forecasts.csv")
save_forecast(result3, f"{OUT_P}/lstm_spec3_seasonal_forecasts.csv")
save_forecast(result4, f"{OUT_P}/lstm_spec4_crisis_seasonal_forecasts.csv")

# =============================================================================
# BLOCK 16: TIME SERIES CROSS-VALIDATION (ROLLING ORIGIN)
# =============================================================================
n_folds = 5

def run_cv_spec(spec_name, predictor_vars, df_full, target_var, continuous_vars, n_obs, lookback, n_folds):
    print(f"\n\n=========================================================")
    print(f"  TIME SERIES CROSS-VALIDATION (ROLLING ORIGIN) - {spec_name}")
    print(f"=========================================================")
    
    all_vars_cv = [target_var] + predictor_vars
    cv_scale_base = int(np.floor(0.6 * n_obs))
    cv_train_raw = df_full.iloc[:cv_scale_base]
    
    cv_scale_params = {}
    for v in [target_var] + continuous_vars:
        cv_scale_params[v] = {'mean': cv_train_raw[v].mean(), 'sd': cv_train_raw[v].std()}
        
    df_scaled_cv = df_full.copy()
    for v in [target_var] + continuous_vars:
        df_scaled_cv[v] = (df_full[v] - cv_scale_params[v]['mean']) / cv_scale_params[v]['sd']
        
    data_matrix_cv = df_scaled_cv[all_vars_cv].values
    target_col_idx_cv = all_vars_cv.index(target_var)
    
    def create_sequences_cv(data_matrix, target_col_idx, lookback):
        n = data_matrix.shape[0]
        n_samples = n - lookback
        X = np.zeros((n_samples, lookback, data_matrix.shape[1]))
        y = np.zeros(n_samples)
        for i in range(n_samples):
            X[i] = data_matrix[i:i+lookback]
            y[i] = data_matrix[i+lookback, target_col_idx]
        return X, y
        
    X_all_cv, y_all_cv = create_sequences_cv(data_matrix_cv, target_col_idx_cv, lookback)
    n_seq_total = len(y_all_cv)
    
    initial_train = int(np.floor(0.5 * n_seq_total))
    remaining = n_seq_total - initial_train
    fold_size = int(np.floor(remaining / n_folds))
    
    cv_results = []
    
    for fold in range(1, n_folds + 1):
        train_end = initial_train + (fold - 1) * fold_size
        test_start = train_end
        test_end = n_seq_total if fold == n_folds else train_end + fold_size
        
        if test_start >= test_end:
            continue
            
        X_tr = X_all_cv[:train_end]
        y_tr = y_all_cv[:train_end]
        X_te = X_all_cv[test_start:test_end]
        y_te = y_all_cv[test_start:test_end]
        
        np.random.seed(42)
        tf.random.set_seed(42)
        
        model_cv = Sequential()
        model_cv.add(Input(shape=(lookback, X_tr.shape[2])))
        model_cv.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=False))
        model_cv.add(Dense(1, activation='linear'))
        model_cv.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        es_cv = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
        rl_cv = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)
        
        model_cv.fit(X_tr, y_tr, epochs=200, batch_size=16, validation_split=0.15, callbacks=[es_cv, rl_cv], shuffle=False, verbose=0)
        
        pred_te = model_cv.predict(X_te, verbose=0).flatten()
        
        def inv(x):
            return x * cv_scale_params[target_var]['sd'] + cv_scale_params[target_var]['mean']
            
        pred_te_orig = inv(pred_te)
        y_te_orig = inv(y_te)
        
        fold_rmse = np.sqrt(mean_squared_error(y_te_orig, pred_te_orig))
        fold_mae = mean_absolute_error(y_te_orig, pred_te_orig)
        fold_mape = mape(y_te_orig, pred_te_orig)
        fold_r2 = r2_score(y_te_orig, pred_te_orig)
        
        print(f"Fold {fold} | Train n={train_end}, Test n={len(y_te)} | RMSE={fold_rmse:.4f} MAE={fold_mae:.4f} MAPE={fold_mape:.2f}% R2={fold_r2:.4f}")
        
        cv_results.append({
            'Spec': spec_name, 'Fold': fold, 'Train_N': train_end, 'Test_N': len(y_te),
            'RMSE': round(fold_rmse, 4), 'MAE': round(fold_mae, 4), 'MAPE': round(fold_mape, 2), 'R2': round(fold_r2, 4)
        })
        
    cv_df = pd.DataFrame(cv_results)
    cv_summary = pd.DataFrame({
        'Spec': [spec_name]*4,
        'Metric': ['RMSE', 'MAE', 'MAPE', 'R2'],
        'Mean': [cv_df['RMSE'].mean(), cv_df['MAE'].mean(), cv_df['MAPE'].mean(), cv_df['R2'].mean()],
        'SD': [cv_df['RMSE'].std(), cv_df['MAE'].std(), cv_df['MAPE'].std(), cv_df['R2'].std()]
    }).round(4)
    
    print(f"\n--- Cross-Validation Mean +/- SD Across Folds ({spec_name}) ---")
    print(cv_summary)
    
    return {'folds': cv_df, 'summary': cv_summary}

cv_spec1 = run_cv_spec("Spec1_NoDummy", predictors_spec1, df_full, target_var, continuous_vars, n_obs, lookback, n_folds)
cv_spec2 = run_cv_spec("Spec2_CrisisDummy", predictors_spec2, df_full, target_var, continuous_vars, n_obs, lookback, n_folds)
cv_spec3 = run_cv_spec("Spec3_SeasonalDummy", predictors_spec3, df_full, target_var, continuous_vars, n_obs, lookback, n_folds)
cv_spec4 = run_cv_spec("Spec4_CrisisSeasonal", predictors_spec4, df_full, target_var, continuous_vars, n_obs, lookback, n_folds)

cv_folds_all = pd.concat([cv_spec1['folds'], cv_spec2['folds'], cv_spec3['folds'], cv_spec4['folds']])
cv_summary_all = pd.concat([cv_spec1['summary'], cv_spec2['summary'], cv_spec3['summary'], cv_spec4['summary']])

print("\n\n=========================================================")
print("  CROSS-VALIDATION SUMMARY: ALL 4 SPECS (Mean +/- SD)")
print("=========================================================")
print(cv_summary_all)

cv_summary_wide = cv_summary_all.pivot(index='Spec', columns='Metric', values=['Mean', 'SD'])
print("\n--- Cross-Validation Comparison Table (Wide Format) ---")
print(cv_summary_wide)

cv_folds_all.to_csv(f"{OUT_P}/lstm_cv_folds_all_specs.csv", index=False)
cv_summary_all.to_csv(f"{OUT_P}/lstm_cv_summary_all_specs.csv", index=False)
cv_summary_wide.to_csv(f"{OUT_P}/lstm_cv_summary_wide.csv")

# =============================================================================
# BLOCK 17: TRAIN vs TEST PERFORMANCE GAP (the core overfitting check)
# =============================================================================
all_results = {
    'Spec1_NoDummy': result1, 'Spec2_CrisisDummy': result2,
    'Spec3_SeasonalDummy': result3, 'Spec4_CrisisSeasonal': result4
}

overfit_list = []
for nm, res in all_results.items():
    train_rmse = np.sqrt(mean_squared_error(res['y_train_true'], res['y_train_pred']))
    train_r2 = r2_score(res['y_train_true'], res['y_train_pred'])
    overfit_list.append({
        'Spec': nm,
        'Train_RMSE': round(train_rmse, 4),
        'Test_RMSE': round(res['rmse'], 4),
        'RMSE_Ratio': round(res['rmse'] / train_rmse, 2),
        'Train_R2': round(train_r2, 4),
        'Test_R2': round(res['r2'], 4),
        'R2_Gap': round(train_r2 - res['r2'], 4)
    })

overfit_gap_table = pd.DataFrame(overfit_list)
print("\n=========================================================")
print("  BLOCK 17: TRAIN vs TEST PERFORMANCE GAP")
print("=========================================================")
print(overfit_gap_table)
overfit_gap_table.to_csv(f"{OUT_P}/lstm_overfit_gap_table.csv", index=False)

# =============================================================================
# BLOCK 18: LEARNING CURVE GAP (val loss / train loss)
# =============================================================================
gap_list = []
for nm, res in all_results.items():
    h = res['history'].history
    n_epochs = len(h['loss'])
    tail_n = max(1, int(np.floor(0.1 * n_epochs)))
    final_train = np.mean(h['loss'][-tail_n:])
    final_val = np.mean(h['val_loss'][-tail_n:])
    gap_list.append({
        'Spec': nm,
        'Epochs_Run': n_epochs,
        'Final_Train_Loss': round(final_train, 5),
        'Final_Val_Loss': round(final_val, 5),
        'Val_Train_Ratio': round(final_val / final_train, 2)
    })

learning_curve_gap = pd.DataFrame(gap_list)
print("\n=========================================================")
print("  BLOCK 18: LEARNING CURVE GAP (VAL LOSS / TRAIN LOSS)")
print("=========================================================")
print(learning_curve_gap)
learning_curve_gap.to_csv(f"{OUT_P}/lstm_learning_curve_gap.csv", index=False)

# =============================================================================
# BLOCK 19: NAIVE BENCHMARK COMPARISON
# =============================================================================
naive_benchmark = df_full[target_var].values
naive_test_actual = naive_benchmark[n_train:n_obs]
naive_test_pred = naive_benchmark[n_train-1:n_obs-1]

naive_rmse = np.sqrt(mean_squared_error(naive_test_actual, naive_test_pred))
naive_mae = mean_absolute_error(naive_test_actual, naive_test_pred)

print("\n=========================================================")
print("  BLOCK 19: LSTM vs NAIVE (RANDOM-WALK) BENCHMARK")
print("=========================================================")
print(f"Naive persistence RMSE: {naive_rmse:.4f} | MAE: {naive_mae:.4f}")

naive_list = []
for nm, res in all_results.items():
    naive_list.append({
        'Spec': nm,
        'LSTM_RMSE': round(res['rmse'], 4),
        'Naive_RMSE': round(naive_rmse, 4),
        'Skill_Ratio': round(1 - (res['rmse'] / naive_rmse), 4)
    })
naive_comparison = pd.DataFrame(naive_list)
print(naive_comparison)
naive_comparison.to_csv(f"{OUT_P}/lstm_naive_benchmark_comparison.csv", index=False)

# =============================================================================
# BLOCK 20: RESIDUAL AUTOCORRELATION (Ljung-Box test)
# =============================================================================
lb_list = []
for nm, res in all_results.items():
    resid = res['y_true'] - res['y_pred']
    lags = [min(4, len(resid) - 1)]
    lb_res = acorr_ljungbox(resid, lags=lags, return_df=True)
    lb_list.append({
        'Spec': nm,
        'LB_statistic': round(lb_res['lb_stat'].values[0], 3),
        'p_value': round(lb_res['lb_pvalue'].values[0], 4)
    })
ljung_box_table = pd.DataFrame(lb_list)
print("\n=========================================================")
print("  BLOCK 20: LJUNG-BOX TEST ON TEST RESIDUALS")
print("=========================================================")
print(ljung_box_table)
ljung_box_table.to_csv(f"{OUT_P}/lstm_ljung_box_residuals.csv", index=False)

# =============================================================================
# BLOCK 21: PLACEBO / SHUFFLED-TARGET TEST
# =============================================================================
def run_placebo_test(predictor_vars, df_full, target_var, continuous_vars, n_train, n_obs, lookback, n_repeats=3):
    np.random.seed(123)
    placebo_r2s = []
    
    for r in range(n_repeats):
        df_shuffled = df_full.copy()
        train_target = df_shuffled.loc[:n_train-1, target_var].values.copy()
        np.random.shuffle(train_target)
        df_shuffled.loc[:n_train-1, target_var] = train_target
        
        all_vars = [target_var] + predictor_vars
        train_raw = df_shuffled.iloc[:n_train]
        scale_params = {}
        for v in [target_var] + continuous_vars:
            scale_params[v] = {'mean': train_raw[v].mean(), 'sd': train_raw[v].std()}
            
        df_scaled = df_shuffled.copy()
        for v in [target_var] + continuous_vars:
            df_scaled[v] = (df_shuffled[v] - scale_params[v]['mean']) / scale_params[v]['sd']
            
        data_matrix = df_scaled[all_vars].values
        target_col_idx = all_vars.index(target_var)
        
        n_samples = data_matrix.shape[0] - lookback
        X = np.zeros((n_samples, lookback, data_matrix.shape[1]))
        y = np.zeros(n_samples)
        for i in range(n_samples):
            X[i] = data_matrix[i:i+lookback]
            y[i] = data_matrix[i+lookback, target_col_idx]
            
        n_train_seq = n_train - lookback
        X_train, y_train = X[:n_train_seq], y[:n_train_seq]
        X_test, y_test = X[n_train_seq:], y[n_train_seq:]
        
        tf.random.set_seed(42)
        model = Sequential()
        model.add(Input(shape=(lookback, X_train.shape[2])))
        model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=False))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.15,
                  callbacks=[
                      EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
                      ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)
                  ], shuffle=False, verbose=0)
        
        pred = model.predict(X_test, verbose=0).flatten()
        def inv(x): return x * scale_params[target_var]['sd'] + scale_params[target_var]['mean']
        
        pred_o = inv(pred)
        y_o = inv(y_test)
        
        placebo_r2s.append(r2_score(y_o, pred_o))
        print(f"  Placebo repeat {r+1}/{n_repeats}: shuffled-target Test R2 = {placebo_r2s[-1]:.4f}")
        
    return placebo_r2s

print("\n=========================================================")
print("  BLOCK 21: PLACEBO TEST (SHUFFLED TARGET) - Spec 4")
print("=========================================================")
placebo_r2_spec4 = run_placebo_test(predictors_spec4, df_full, target_var, continuous_vars, n_train, n_obs, lookback, n_repeats=3)
print(f"\nMean placebo Test R2 (shuffled target): {np.mean(placebo_r2_spec4):.4f} (SD: {np.std(placebo_r2_spec4):.4f})")
print(f"Actual model Test R2 (Spec 4, real target): {result4['r2']:.4f}")

# =============================================================================
# BLOCK 22: PARAMETER COUNT vs SAMPLE SIZE
# =============================================================================
print("\n=========================================================")
print("  BLOCK 22: PARAMETER COUNT vs TRAINING SAMPLE SIZE")
print("=========================================================")
param_list = []
n_train_seq = n_train - lookback
for nm, res in all_results.items():
    n_params = res['model'].count_params()
    param_list.append({
        'Spec': nm,
        'N_Params': n_params,
        'N_Train_Sequences': n_train_seq,
        'Params_per_Sample': round(n_params / n_train_seq, 2)
    })
param_table = pd.DataFrame(param_list)
print(param_table)
param_table.to_csv(f"{OUT_P}/lstm_param_vs_sample_size.csv", index=False)

# =============================================================================
# BLOCK 23: COMBINED LEARNING CURVE PLOT
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
axes = axes.flatten()
spec_names = ["Spec 1: No Dummy", "Spec 2: + Crisis Dummies", "Spec 3: + Seasonal Dummies", "Spec 4: + Crisis + Seasonal Dummies"]

for idx, (res, label) in enumerate(zip([result1, result2, result3, result4], spec_names)):
    ax = axes[idx]
    ax.plot(res['history'].history['loss'], label='Training', color='steelblue', linewidth=2)
    ax.plot(res['history'].history['val_loss'], label='Validation', color='tomato', linewidth=2)
    ax.set_title(label, fontweight='bold')
    ax.set_ylabel("MSE Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc="upper right")

fig.suptitle("LSTM Learning Curves: All 4 Specifications\nTraining vs Validation Loss by Epoch")
plt.tight_layout()
plt.savefig(f"{OUT_P}/lstm_learning_curves_all_specs.png", dpi=150)
plt.close()

print("\nSaved: lstm_learning_curves_all_specs.png")

# Save best spec: Spec 4 Crisis and Seasonal dummies
save_forecast(result4, f"data/outputs/forecasts/lstm_spec4_forecast.csv")