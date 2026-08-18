# =============================================================================
# MULTIVARIATE LSTM — UK HOUSING SUPPLY FORECASTING
#
#   Spec 1: No dummies (baseline)
#   Spec 2: + 3 crisis dummies (covid pulse, crisis pulse, crisis step)
#   Spec 3: + 3 seasonal dummies (Q1, Q2, Q3; Q4 = reference)
#   Spec 4: + crisis dummies AND seasonal dummies (all 6)
#
# -----------------------------------------------------------------------------
# SPECIFICATION NOTES (why this differs from the original levels version)
# -----------------------------------------------------------------------------
# 1. STATIONARY TARGET AND PREDICTORS.
#    The original modelled log LEVELS, standardised on the 1975-2009 training
#    window. Every series is I(1) and drifts out of that window: over the
#    2010Q1-2025Q4 test period 100% of lstock, 62% of lrcc, 44% of lrprc and
#    22% of r3 observations fall outside the training range entirely, and the
#    test mean of the target sits 0.63 training-sd below the training mean.
#    An LSTM with a linear head cannot extrapolate beyond its fitted range, so
#    it collapsed onto the training mean: forecast sd was 0.05-0.06 against an
#    actual sd of 0.285, with a systematic -0.11 to -0.18 log-point bias in
#    every quarter. Almost all of the reported RMSE was that bias.
#    We therefore model FIRST DIFFERENCES and reconstruct the level as
#    (last actual level + predicted change), i.e. a genuine 1-step-ahead
#    forecast on the same information set as the random-walk benchmark.
#
# 2. DUMMY IDENTIFIABILITY.
#    With the test window starting 2010Q1, the three crisis dummies are
#    identically zero across ALL 140 training observations. Their input
#    weights receive no gradient, remain at their random initialisation, and
#    then switch on during the test period — injecting noise at exactly the
#    hardest point of the sample. Specs containing them are not identified
#    under this split. Block 6b detects this automatically and drops such
#    columns rather than silently fitting them.
#
# 3. SEED DISPERSION.
#    A single fit is not informative here. Across 5 seeds the within-spec test
#    RMSE range (median 0.070) exceeded the entire between-spec spread of the
#    original comparison table (0.044), and the spec ranking inverted. Every
#    specification is therefore run over multiple seeds; we report mean +/- sd
#    and use the seed-ensemble mean as the point forecast.
#
# 4. CAPACITY.
#    64 units on 132 training sequences is ~19,000 parameters, ~150 per
#    sample. Reduced to LSTM_UNITS below.
#
# Set DIFFERENCE_TARGET = DIFFERENCE_PREDICTORS = False to reproduce the
# original levels specification for comparison.
# =============================================================================

# =============================================================================
# BLOCK 1: LOAD PACKAGES AND FIX DETERMINISM
# =============================================================================
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

# tf.random.set_seed alone does NOT make this reproducible: repeated runs of
# the original script with seed 42 returned test RMSE of 0.277 / 0.280 / 0.311
# / 0.423. Op-level determinism and single-threaded execution are both needed.
tf.config.experimental.enable_op_determinism()
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

print(f"TensorFlow version: {tf.__version__}")

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DIFFERENCE_TARGET = True      # model d(log starts), reconstruct to levels
DIFFERENCE_PREDICTORS = True  # difference the I(1) continuous predictors
LSTM_UNITS = 16               # was 64; 132 training sequences
SEEDS = [42, 1, 7, 123, 2024, 31, 99, 256, 777, 8]
CV_SEEDS = [42, 1, 7]         # rolling-origin CV is 5 folds x 4 specs per seed
PLACEBO_REPEATS = 3

OUT_P = "data/outputs/lstm"
FORECAST_P = "data/outputs/forecasts"
os.makedirs(OUT_P, exist_ok=True)
os.makedirs(FORECAST_P, exist_ok=True)

# =============================================================================
# BLOCK 2: IMPORT AND RENAME COLUMNS
# =============================================================================
file_path = "data/python_master/england_master.csv"

df_raw = pd.read_csv(file_path, index_col=None)
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

# Drop trailing rows with no target: the final quarter of the master carries
# published deflators but no starts/vol/stock/rate yet. Imputing it would
# fabricate a test observation.
n_before = len(df_raw)
df_raw = df_raw[df_raw['lhstarts'].notna()].reset_index(drop=True)
if len(df_raw) < n_before:
    print(f"Dropped {n_before - len(df_raw)} trailing row(s) with no observed starts")

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
for c in ['d_covid', 'd_crisis_pulse', 'd_crisis_step', 'd_Q1', 'd_Q2', 'd_Q3']:
    print(f"{c:16s} active quarters : {df_raw[c].sum()}")

# =============================================================================
# BLOCK 6: HANDLE MISSING VALUES (CONTINUOUS VARS ONLY)
# =============================================================================
target_var = "lhstarts"
continuous_vars = ["lrprc", "lvol", "r3", "lstock", "lrcc"]
all_dummy_vars = ["d_covid", "d_crisis_pulse", "d_crisis_step", "d_Q1", "d_Q2", "d_Q3"]

df_cont = df_raw[continuous_vars + [target_var]].interpolate(method='linear')
df_cont = df_cont.bfill().ffill()

df_full = df_raw[['date'] + all_dummy_vars].copy()
for col in df_cont.columns:
    df_full[col] = df_cont[col]

dates = df_full['date'].values
print("\n--- Missing Values After Imputation ---")
print(df_full.isna().sum().to_string())

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
n_test = n_obs - n_train

print("\n--- Train / Test Split (shared across all specs) ---")
print(f"Total observations : {n_obs}")
print(f"Training           : {n_train} | From {pd.to_datetime(dates[0]).date()} to {pd.to_datetime(dates[n_train-1]).date()}")
print(f"Testing            : {n_test}  | From {pd.to_datetime(dates[n_train]).date()} to {pd.to_datetime(dates[-1]).date()}")

test_dates = dates[n_train:]

# Log level of the target, kept undifferenced for reconstruction and scoring.
LEVEL = df_full[target_var].values.copy()

# =============================================================================
# BLOCK 6b: DUMMY IDENTIFIABILITY GUARD
# =============================================================================
# A dummy that never varies inside the training window cannot be estimated.
# Its input weights stay at their random initialisation and then activate in
# the test period, injecting noise. Detect and drop such columns explicitly.
print("\n--- Dummy Identifiability Check (variation within training window) ---")
identified_dummies, unidentified_dummies = [], []
for d in all_dummy_vars:
    train_active = int(df_full[d].iloc[:n_train].sum())
    test_active = int(df_full[d].iloc[n_train:].sum())
    ok = df_full[d].iloc[:n_train].nunique() > 1
    (identified_dummies if ok else unidentified_dummies).append(d)
    flag = "OK" if ok else "NOT IDENTIFIED (constant in training) -> dropped"
    print(f"  {d:16s} train_active={train_active:3d} test_active={test_active:3d}  {flag}")

if unidentified_dummies:
    print(f"\n  Dropping {len(unidentified_dummies)} unidentified dummy/dummies: "
          f"{', '.join(unidentified_dummies)}")
    print("  (constant across the whole training window under this split)")


def filter_predictors(predictor_vars):
    """Remove dummies that carry no training-window variation."""
    return [v for v in predictor_vars if v not in unidentified_dummies]


# =============================================================================
# BLOCK 8: DATA PREPARATION AND MODEL FITTING
# =============================================================================
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def create_sequences(data_matrix, target_col_idx, lookback):
    n = data_matrix.shape[0]
    n_samples = n - lookback
    X = np.zeros((n_samples, lookback, data_matrix.shape[1]))
    y = np.zeros(n_samples)
    for i in range(n_samples):
        X[i] = data_matrix[i:(i + lookback), :]
        y[i] = data_matrix[i + lookback, target_col_idx]
    return X, y


def prepare_arrays(predictor_vars, split_row, scale_rows=None):
    """Build supervised arrays in the (optionally differenced) modelling frame.

    Returns X, y, orig_rows, scale_params, n_train_seq where orig_rows maps each
    sequence back to its row in the ORIGINAL undifferenced frame, so that
    predictions can always be scored against LEVEL[orig_rows].
    """
    d = df_full.copy()
    offset = 0
    to_diff = []
    if DIFFERENCE_TARGET:
        to_diff.append(target_var)
    if DIFFERENCE_PREDICTORS:
        to_diff += continuous_vars
    if to_diff:
        d[to_diff] = d[to_diff].diff()
        d = d.iloc[1:].reset_index(drop=True)
        offset = 1

    nt = split_row - offset
    all_vars = [target_var] + predictor_vars
    vars_to_scale = [target_var] + continuous_vars

    # Scaling statistics come from training rows only (no test-period leakage).
    scale_end = nt if scale_rows is None else scale_rows - offset
    train_raw = d.iloc[:scale_end]
    scale_params = {v: {'mean': train_raw[v].mean(), 'sd': train_raw[v].std()}
                    for v in vars_to_scale}

    df_scaled = d.copy()
    for v in vars_to_scale:
        df_scaled[v] = (d[v] - scale_params[v]['mean']) / scale_params[v]['sd']

    data_matrix = df_scaled[all_vars].values
    X, y = create_sequences(data_matrix, all_vars.index(target_var), lookback)

    # sequence j targets modelling-frame row j+lookback = original row j+lookback+offset
    orig_rows = np.arange(len(y)) + lookback + offset
    return X, y, orig_rows, scale_params, nt - lookback


def to_level(pred_scaled, scale_params, orig_rows):
    """Undo standardisation and, if differencing, rebuild the log level.

    Reconstruction uses the last ACTUAL level plus the predicted change, i.e. a
    1-step-ahead forecast on the same information set as the random walk.
    """
    p = pred_scaled * scale_params[target_var]['sd'] + scale_params[target_var]['mean']
    if DIFFERENCE_TARGET:
        return LEVEL[orig_rows - 1] + p
    return p


def build_model(n_features, units=LSTM_UNITS):
    model = Sequential([
        Input(shape=(lookback, n_features)),
        LSTM(units, dropout=0.1, recurrent_dropout=0.1, return_sequences=False),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="mean_squared_error",
                  metrics=["mean_absolute_error"])
    return model


def fit_one(X_train, y_train, seed, epochs=300, patience=30, lr_patience=15, units=LSTM_UNITS):
    """Fit a single model under a fixed seed. Validation is the final 15% of the
    training sequences (chronological, no shuffling): correct for time series,
    but note it is a single contiguous block, so early stopping is selecting on
    whatever regime happens to sit at the end of the training window."""
    tf.keras.utils.set_random_seed(seed)
    model = build_model(X_train.shape[2], units=units)
    history = model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=16,
        validation_split=0.15,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=lr_patience, min_lr=1e-6, verbose=0),
        ],
        shuffle=False, verbose=0,
    )
    return model, history


def score(y_true, y_pred):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mape': float(mape(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def run_lstm_spec(spec_name, predictor_vars, seeds=SEEDS):
    """Fit one specification across several seeds and return the seed ensemble.

    All metrics are reported on the LOG LEVEL of housing starts regardless of
    whether the model was estimated in differences, so they stay comparable to
    the random-walk benchmark and to the rest of the horse race.
    """
    predictor_vars = filter_predictors(predictor_vars)

    print(f"\n\n=====================================================")
    print(f"  RUNNING SPEC: {spec_name}")
    print(f"  Predictors ({len(predictor_vars)}): {', '.join(predictor_vars)}")
    print(f"  Seeds: {len(seeds)} | units: {LSTM_UNITS} | "
          f"target: {'d(log)' if DIFFERENCE_TARGET else 'log level'}")
    print(f"=====================================================")

    X, y, orig_rows, scale_params, ns = prepare_arrays(predictor_vars, n_train)

    X_train, y_train = X[:ns], y[:ns]
    X_test = X[ns:]
    rows_train, rows_test = orig_rows[:ns], orig_rows[ns:]

    y_true = LEVEL[rows_test]
    y_train_true = LEVEL[rows_train]

    n_features = X_train.shape[2]
    print(f"Input features: {n_features} | Train seqs: {len(X_train)} | Test seqs: {len(X_test)}")

    test_preds, train_preds, histories, per_seed, best_epochs = [], [], [], [], []
    for seed in seeds:
        model, history = fit_one(X_train, y_train, seed)
        p_te = to_level(model.predict(X_test, verbose=0).flatten(), scale_params, rows_test)
        p_tr = to_level(model.predict(X_train, verbose=0).flatten(), scale_params, rows_train)
        test_preds.append(p_te)
        train_preds.append(p_tr)
        histories.append(history)
        best_epochs.append(int(np.argmin(history.history['val_loss'])) + 1)
        m = score(y_true, p_te)
        m['seed'] = seed
        m['epochs'] = len(history.history['loss'])
        m['best_epoch'] = best_epochs[-1]
        per_seed.append(m)
        print(f"  seed {seed:5d}: RMSE={m['rmse']:.4f} R2={m['r2']:+.4f} "
              f"(epochs {m['epochs']}, best {m['best_epoch']})")
        # Keep the first seed's history (a plain dict of floats) for the
        # learning-curve diagnostics and its parameter count for Block 22.
        # The model objects themselves are released every iteration: retaining
        # one across clear_session() would leave an invalidated handle.
        if seed == seeds[0]:
            first_history = history
            first_n_params = int(model.count_params())
        del model
        tf.keras.backend.clear_session()

    y_pred = np.mean(test_preds, axis=0)          # seed-ensemble point forecast
    y_train_pred = np.mean(train_preds, axis=0)
    ens = score(y_true, y_pred)

    per_seed_df = pd.DataFrame(per_seed)
    print(f"  --> per-seed RMSE  mean {per_seed_df['rmse'].mean():.4f} "
          f"sd {per_seed_df['rmse'].std():.4f} "
          f"range {per_seed_df['rmse'].max() - per_seed_df['rmse'].min():.4f}")
    print(f"  --> ENSEMBLE       RMSE {ens['rmse']:.4f} | MAE {ens['mae']:.4f} | "
          f"MAPE {ens['mape']:.2f}% | R2 {ens['r2']:+.4f}")
    print(f"  --> forecast sd {y_pred.std():.4f} vs actual sd {y_true.std():.4f} | "
          f"mean bias {(y_true - y_pred).mean():+.4f}")

    return {
        'spec_name': spec_name,
        'predictor_vars': predictor_vars,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_train_true': y_train_true,
        'y_train_pred': y_train_pred,
        'test_dates': dates[rows_test],
        'per_seed': per_seed_df,
        'history': first_history,
        'histories': histories,
        'n_params': first_n_params,
        'n_features': n_features,
        'n_train_seq': ns,
        **ens,
    }


# =============================================================================
# BLOCK 9: DEFINE THE FOUR FEATURE SETS
# =============================================================================
SPEC_DEFS = [
    ("Spec1_NoDummy", "Spec 1: No dummy", continuous_vars),
    ("Spec2_CrisisDummy", "Spec 2: + Crisis dummies (covid, crisis pulse, crisis step)",
     continuous_vars + ["d_covid", "d_crisis_pulse", "d_crisis_step"]),
    ("Spec3_SeasonalDummy", "Spec 3: + Seasonal dummies (Q1-Q3)",
     continuous_vars + ["d_Q1", "d_Q2", "d_Q3"]),
    ("Spec4_CrisisSeasonal", "Spec 4: + Crisis + Seasonal dummies",
     continuous_vars + ["d_covid", "d_crisis_pulse", "d_crisis_step", "d_Q1", "d_Q2", "d_Q3"]),
]

# Drop specs that collapse onto an already-estimated one once unidentified
# dummies are removed, rather than reporting the same fit under two names.
runnable, skipped = [], []
seen = {}
for key, label, preds in SPEC_DEFS:
    filt = tuple(filter_predictors(preds))
    if filt in seen:
        skipped.append((key, label, seen[filt]))
    else:
        seen[filt] = key
        runnable.append((key, label, preds))

if skipped:
    print("\n--- Specifications not identified under this split ---")
    for key, label, dup_of in skipped:
        print(f"  {key:22s} collapses onto {dup_of} once unidentified dummies are dropped -> skipped")

# =============================================================================
# BLOCK 10: RUN ALL IDENTIFIED SPECIFICATIONS
# =============================================================================
results = {}
for key, label, preds in runnable:
    results[key] = run_lstm_spec(key, preds)
    results[key]['label'] = label

# =============================================================================
# BLOCK 11: COMPARISON TABLE (with seed dispersion and the RW benchmark)
# =============================================================================
naive_actual = LEVEL[n_train:n_obs]
naive_pred = LEVEL[n_train - 1:n_obs - 1]
naive = score(naive_actual, naive_pred)

rows = []
for key, res in results.items():
    ps = res['per_seed']
    rows.append({
        'Specification': res['label'],
        'Features': res['n_features'],
        'RMSE': round(res['rmse'], 4),
        'MAE': round(res['mae'], 4),
        'MAPE': round(res['mape'], 2),
        'R2': round(res['r2'], 4),
        'RMSE_seed_mean': round(ps['rmse'].mean(), 4),
        'RMSE_seed_sd': round(ps['rmse'].std(), 4),
        'RMSE_seed_range': round(ps['rmse'].max() - ps['rmse'].min(), 4),
        'Skill_vs_RW': round(1 - res['rmse'] / naive['rmse'], 4),
    })
rows.append({
    'Specification': 'Benchmark: random walk (no change)',
    'Features': 0, 'RMSE': round(naive['rmse'], 4), 'MAE': round(naive['mae'], 4),
    'MAPE': round(naive['mape'], 2), 'R2': round(naive['r2'], 4),
    'RMSE_seed_mean': np.nan, 'RMSE_seed_sd': np.nan, 'RMSE_seed_range': np.nan,
    'Skill_vs_RW': 0.0,
})
comparison_table = pd.DataFrame(rows)

print("\n\n=========================================================")
print("  COMPARISON TABLE: LSTM SPECIFICATIONS (seed ensembles)")
print("=========================================================")
print(comparison_table.to_string(index=False))
print("\nRMSE/MAE/MAPE/R2 are for the seed-ensemble forecast, scored on the LOG LEVEL.")
print("RMSE_seed_* describe dispersion of individual fits: if RMSE_seed_range is")
print("comparable to the spread ACROSS specifications, the ranking is not identified.")
comparison_table.to_csv(f"{OUT_P}/lstm_spec_comparison.csv", index=False)

per_seed_all = pd.concat(
    [res['per_seed'].assign(Spec=key) for key, res in results.items()], ignore_index=True
)
per_seed_all.to_csv(f"{OUT_P}/lstm_per_seed_metrics.csv", index=False)

# =============================================================================
# BLOCKS 12, 13, 14: ACTUAL-VS-PREDICTED, RESIDUAL AND LOSS PLOTS
# =============================================================================
# NOTE: the previous version built these figures and then called plt.close()
# without plt.savefig(), so all six group plots were silently discarded.
def _pairs(keys):
    return [(keys[i], keys[i + 1] if i + 1 < len(keys) else None)
            for i in range(0, len(keys), 2)]


def plot_avp(specs, filename, title):
    fig, axes = plt.subplots(len(specs), 1, figsize=(10, 3.6 * len(specs)), sharex=True, squeeze=False)
    for ax, key in zip(axes[:, 0], specs):
        res = results[key]
        ax.plot(res['test_dates'], res['y_true'], label="Actual", color="black")
        ax.plot(res['test_dates'], res['y_pred'], label="Predicted", color="steelblue", linestyle="--")
        ax.set_title(res['label'], fontweight='bold')
        ax.set_ylabel("Log Housing Starts")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Date")
    axes[0, 0].legend(loc="best", ncol=2)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{OUT_P}/{filename}", dpi=150)
    plt.close(fig)


def plot_resid(specs, filename, title):
    fig, axes = plt.subplots(len(specs), 1, figsize=(10, 3.6 * len(specs)), sharex=True, squeeze=False)
    for ax, key in zip(axes[:, 0], specs):
        res = results[key]
        ax.plot(res['test_dates'], res['y_true'] - res['y_pred'], color="darkred")
        ax.axhline(0, linestyle="--", color="grey")
        ax.set_title(res['label'], fontweight='bold')
        ax.set_ylabel("Residual")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Date")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{OUT_P}/{filename}", dpi=150)
    plt.close(fig)


def plot_loss(specs, filename, title):
    fig, axes = plt.subplots(len(specs), 1, figsize=(9, 3.6 * len(specs)), squeeze=False)
    for ax, key in zip(axes[:, 0], specs):
        res = results[key]
        ax.plot(res['history'].history['loss'], label="Training", color="steelblue")
        ax.plot(res['history'].history['val_loss'], label="Validation", color="tomato")
        ax.set_title(f"{res['label']} (seed {SEEDS[0]})", fontweight='bold')
        ax.set_ylabel("MSE Loss")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{OUT_P}/{filename}", dpi=150)
    plt.close(fig)


spec_keys = list(results.keys())
plot_avp(spec_keys, "lstm_avp_all_specs.png", "LSTM Forecast Comparison (seed ensemble)")
plot_resid(spec_keys, "lstm_resid_all_specs.png", "LSTM Residuals (seed ensemble)")
plot_loss(spec_keys, "lstm_loss_all_specs.png", "LSTM Training History")
print(f"\nSaved: lstm_avp_all_specs.png, lstm_resid_all_specs.png, lstm_loss_all_specs.png")

# Forecast dispersion plot: the failure mode of the levels specification was a
# near-flat forecast, so make that directly visible.
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = [results[k]['spec_name'] for k in spec_keys]
ax.bar(np.arange(len(spec_keys)) - 0.2, [results[k]['y_pred'].std() for k in spec_keys],
       width=0.4, label="Forecast sd", color="steelblue")
ax.bar(np.arange(len(spec_keys)) + 0.2, [results[k]['y_true'].std() for k in spec_keys],
       width=0.4, label="Actual sd", color="black")
ax.set_xticks(np.arange(len(spec_keys)))
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel("sd of log housing starts")
ax.set_title("Forecast dispersion vs actual dispersion")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_P}/lstm_forecast_dispersion.png", dpi=150)
plt.close(fig)

# =============================================================================
# BLOCK 15: SAVE INDIVIDUAL FORECASTS
# =============================================================================
def save_forecast(res, filename):
    pd.DataFrame({
        'date': res['test_dates'],
        'actual': np.round(res['y_true'], 4),
        'lstm': np.round(res['y_pred'], 4),
        'residual': np.round(res['y_true'] - res['y_pred'], 4),
    }).to_csv(filename, index=False)


FILE_STEMS = {
    'Spec1_NoDummy': 'lstm_spec1_nodummy_forecasts.csv',
    'Spec2_CrisisDummy': 'lstm_spec2_crisis_forecasts.csv',
    'Spec3_SeasonalDummy': 'lstm_spec3_seasonal_forecasts.csv',
    'Spec4_CrisisSeasonal': 'lstm_spec4_crisis_seasonal_forecasts.csv',
}
for key, res in results.items():
    save_forecast(res, f"{OUT_P}/{FILE_STEMS[key]}")

# Headline forecast = best identified specification by ensemble RMSE.
# The previous version hard-coded Spec 4, which is not identified under this
# split (its crisis dummies never vary in training).
best_key = min(results, key=lambda k: results[k]['rmse'])
save_forecast(results[best_key], f"{FORECAST_P}/lstm_headline_forecast.csv")
print(f"\nHeadline specification: {best_key} ({results[best_key]['label']})")
print(f"Saved: {FORECAST_P}/lstm_headline_forecast.csv")

pd.DataFrame([{
    'headline_spec': best_key,
    'label': results[best_key]['label'],
    'predictors': ", ".join(results[best_key]['predictor_vars']),
    'rmse': round(results[best_key]['rmse'], 4),
    'r2': round(results[best_key]['r2'], 4),
    'skill_vs_rw': round(1 - results[best_key]['rmse'] / naive['rmse'], 4),
    'n_seeds': len(SEEDS),
    'differenced_target': DIFFERENCE_TARGET,
    'differenced_predictors': DIFFERENCE_PREDICTORS,
    'lstm_units': LSTM_UNITS,
    'unidentified_dummies': ", ".join(unidentified_dummies) or "none",
}]).to_csv(f"{OUT_P}/lstm_headline_spec.csv", index=False)

# =============================================================================
# BLOCK 16: TIME SERIES CROSS-VALIDATION (ROLLING ORIGIN)
# =============================================================================
n_folds = 5


def run_cv_spec(spec_name, predictor_vars, seeds=CV_SEEDS):
    predictor_vars = filter_predictors(predictor_vars)
    print(f"\n\n=========================================================")
    print(f"  TIME SERIES CROSS-VALIDATION (ROLLING ORIGIN) - {spec_name}")
    print(f"=========================================================")

    # Scale on the first 60% of observations (a fixed reference window, so the
    # folds stay comparable) and never on anything a fold has not yet seen.
    cv_scale_base = int(np.floor(0.6 * n_obs))
    X, y, orig_rows, scale_params, _ = prepare_arrays(
        predictor_vars, n_train, scale_rows=cv_scale_base)

    n_seq_total = len(y)
    initial_train = int(np.floor(0.5 * n_seq_total))
    fold_size = int(np.floor((n_seq_total - initial_train) / n_folds))

    cv_results = []
    for fold in range(1, n_folds + 1):
        train_end = initial_train + (fold - 1) * fold_size
        test_start = train_end
        test_end = n_seq_total if fold == n_folds else train_end + fold_size
        if test_start >= test_end:
            continue

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te = X[test_start:test_end]
        rows_te = orig_rows[test_start:test_end]
        y_te_lvl = LEVEL[rows_te]

        fold_preds = []
        for seed in seeds:
            model_cv, _ = fit_one(X_tr, y_tr, seed, epochs=200, patience=20, lr_patience=10)
            fold_preds.append(to_level(model_cv.predict(X_te, verbose=0).flatten(),
                                       scale_params, rows_te))
            tf.keras.backend.clear_session()
        pred_lvl = np.mean(fold_preds, axis=0)

        m = score(y_te_lvl, pred_lvl)
        rw = score(y_te_lvl, LEVEL[rows_te - 1])
        print(f"Fold {fold} | Train n={train_end}, Test n={len(y_te_lvl)} | "
              f"RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} MAPE={m['mape']:.2f}% "
              f"R2={m['r2']:+.4f} | RW RMSE={rw['rmse']:.4f}")
        cv_results.append({
            'Spec': spec_name, 'Fold': fold, 'Train_N': train_end, 'Test_N': len(y_te_lvl),
            'RMSE': round(m['rmse'], 4), 'MAE': round(m['mae'], 4),
            'MAPE': round(m['mape'], 2), 'R2': round(m['r2'], 4),
            'RW_RMSE': round(rw['rmse'], 4),
            'Skill_vs_RW': round(1 - m['rmse'] / rw['rmse'], 4),
        })

    cv_df = pd.DataFrame(cv_results)
    cv_summary = pd.DataFrame({
        'Spec': [spec_name] * 5,
        'Metric': ['RMSE', 'MAE', 'MAPE', 'R2', 'Skill_vs_RW'],
        'Mean': [cv_df[c].mean() for c in ['RMSE', 'MAE', 'MAPE', 'R2', 'Skill_vs_RW']],
        'SD': [cv_df[c].std() for c in ['RMSE', 'MAE', 'MAPE', 'R2', 'Skill_vs_RW']],
    }).round(4)

    print(f"\n--- Cross-Validation Mean +/- SD Across Folds ({spec_name}) ---")
    print(cv_summary.to_string(index=False))
    return {'folds': cv_df, 'summary': cv_summary}


cv_out = {key: run_cv_spec(key, preds) for key, _, preds in runnable}

cv_folds_all = pd.concat([v['folds'] for v in cv_out.values()], ignore_index=True)
cv_summary_all = pd.concat([v['summary'] for v in cv_out.values()], ignore_index=True)

print("\n\n=========================================================")
print("  CROSS-VALIDATION SUMMARY: ALL SPECS (Mean +/- SD)")
print("=========================================================")
print(cv_summary_all.to_string(index=False))

cv_summary_wide = cv_summary_all.pivot(index='Spec', columns='Metric', values=['Mean', 'SD'])
cv_summary_wide.columns = [f"{a}_{b}" for a, b in cv_summary_wide.columns]
cv_summary_wide = cv_summary_wide.reset_index()
print("\n--- Cross-Validation Comparison Table (Wide Format) ---")
print(cv_summary_wide.to_string(index=False))

cv_folds_all.to_csv(f"{OUT_P}/lstm_cv_folds_all_specs.csv", index=False)
cv_summary_all.to_csv(f"{OUT_P}/lstm_cv_summary_all_specs.csv", index=False)
cv_summary_wide.to_csv(f"{OUT_P}/lstm_cv_summary_wide.csv", index=False)

# =============================================================================
# BLOCK 17: TRAIN vs TEST PERFORMANCE GAP
# =============================================================================
overfit_list = []
for nm, res in results.items():
    train_rmse = float(np.sqrt(mean_squared_error(res['y_train_true'], res['y_train_pred'])))
    train_r2 = float(r2_score(res['y_train_true'], res['y_train_pred']))
    overfit_list.append({
        'Spec': nm,
        'Train_RMSE': round(train_rmse, 4),
        'Test_RMSE': round(res['rmse'], 4),
        'RMSE_Ratio': round(res['rmse'] / train_rmse, 2),
        'Train_R2': round(train_r2, 4),
        'Test_R2': round(res['r2'], 4),
        'R2_Gap': round(train_r2 - res['r2'], 4),
    })
overfit_gap_table = pd.DataFrame(overfit_list)
print("\n=========================================================")
print("  BLOCK 17: TRAIN vs TEST PERFORMANCE GAP")
print("=========================================================")
print(overfit_gap_table.to_string(index=False))
print("\nRead Train_R2 before R2_Gap: a large gap alongside a LOW Train_R2 is not")
print("memorisation, it is a biased/underfit model whose test R2 has gone negative.")
overfit_gap_table.to_csv(f"{OUT_P}/lstm_overfit_gap_table.csv", index=False)

# =============================================================================
# BLOCK 18: LEARNING CURVE GAP (val loss / train loss)
# =============================================================================
gap_list = []
for nm, res in results.items():
    h = res['history'].history
    n_epochs = len(h['loss'])
    tail_n = max(1, int(np.floor(0.1 * n_epochs)))
    final_train = float(np.mean(h['loss'][-tail_n:]))
    final_val = float(np.mean(h['val_loss'][-tail_n:]))
    gap_list.append({
        'Spec': nm,
        'Epochs_Run': n_epochs,
        'Best_Epoch_mean': round(res['per_seed']['best_epoch'].mean(), 1),
        'Final_Train_Loss': round(final_train, 5),
        'Final_Val_Loss': round(final_val, 5),
        'Val_Train_Ratio': round(final_val / final_train, 2),
    })
learning_curve_gap = pd.DataFrame(gap_list)
print("\n=========================================================")
print("  BLOCK 18: LEARNING CURVE GAP (VAL LOSS / TRAIN LOSS)")
print("=========================================================")
print(learning_curve_gap.to_string(index=False))
print("\nThe validation block is the final 15% of the training window. Under the")
print("2010 split that is roughly 2005Q2-2009Q4, i.e. the financial crisis, so a")
print("high ratio partly reflects that block being unrepresentative rather than")
print("overfitting as such. Treat Best_Epoch alongside it.")
learning_curve_gap.to_csv(f"{OUT_P}/lstm_learning_curve_gap.csv", index=False)

# =============================================================================
# BLOCK 19: NAIVE BENCHMARK COMPARISON
# =============================================================================
print("\n=========================================================")
print("  BLOCK 19: LSTM vs NAIVE (RANDOM-WALK) BENCHMARK")
print("=========================================================")
print(f"Naive persistence RMSE: {naive['rmse']:.4f} | MAE: {naive['mae']:.4f}")

naive_list = []
for nm, res in results.items():
    ps = res['per_seed']
    naive_list.append({
        'Spec': nm,
        'LSTM_RMSE': round(res['rmse'], 4),
        'Naive_RMSE': round(naive['rmse'], 4),
        'Skill_Ratio': round(1 - (res['rmse'] / naive['rmse']), 4),
        'Seeds_beating_RW': int((ps['rmse'] < naive['rmse']).sum()),
        'N_seeds': len(ps),
    })
naive_comparison = pd.DataFrame(naive_list)
print(naive_comparison.to_string(index=False))
print("\nSeeds_beating_RW counts individual fits, not the ensemble: if this is near")
print("half the seeds, any skill claim is a coin flip rather than a result.")
naive_comparison.to_csv(f"{OUT_P}/lstm_naive_benchmark_comparison.csv", index=False)

# =============================================================================
# BLOCK 20: RESIDUAL AUTOCORRELATION (Ljung-Box test)
# =============================================================================
lb_list = []
for nm, res in results.items():
    resid = res['y_true'] - res['y_pred']
    lb_res = acorr_ljungbox(resid, lags=[min(4, len(resid) - 1)], return_df=True)
    lb_list.append({
        'Spec': nm,
        'Mean_Residual': round(float(resid.mean()), 4),
        'LB_statistic': round(float(lb_res['lb_stat'].values[0]), 3),
        'p_value': round(float(lb_res['lb_pvalue'].values[0]), 4),
    })
ljung_box_table = pd.DataFrame(lb_list)
print("\n=========================================================")
print("  BLOCK 20: LJUNG-BOX TEST ON TEST RESIDUALS")
print("=========================================================")
print(ljung_box_table.to_string(index=False))
ljung_box_table.to_csv(f"{OUT_P}/lstm_ljung_box_residuals.csv", index=False)

# =============================================================================
# BLOCK 21: PLACEBO / SHUFFLED-TARGET TEST
# =============================================================================
# Shuffle the target WITHIN THE TRAINING WINDOW ONLY, so the test target stays
# real and the question is whether a model trained on scrambled targets can
# still appear to predict. (The R original shuffled the entire column, which
# also scrambles the thing being predicted and makes the test uninformative.)
def run_placebo_test(predictor_vars, n_repeats=PLACEBO_REPEATS):
    predictor_vars = filter_predictors(predictor_vars)
    rng = np.random.default_rng(123)
    global df_full
    df_original = df_full.copy()
    placebo = []
    try:
        for r in range(n_repeats):
            shuffled = df_original.copy()
            block = shuffled.loc[:n_train - 1, target_var].values.copy()
            rng.shuffle(block)
            shuffled.loc[:n_train - 1, target_var] = block
            df_full = shuffled

            X, y, orig_rows, sp, ns = prepare_arrays(predictor_vars, n_train)
            model, _ = fit_one(X[:ns], y[:ns], 42, epochs=150, patience=20, lr_patience=10)
            rows_te = orig_rows[ns:]
            # Score against the REAL level, which the shuffle never touched.
            pred = to_level(model.predict(X[ns:], verbose=0).flatten(), sp, rows_te)
            placebo.append(float(r2_score(LEVEL[rows_te], pred)))
            tf.keras.backend.clear_session()
            print(f"  Placebo repeat {r+1}/{n_repeats}: shuffled-target Test R2 = {placebo[-1]:.4f}")
    finally:
        df_full = df_original
    return placebo


print("\n=========================================================")
print(f"  BLOCK 21: PLACEBO TEST (SHUFFLED TRAINING TARGET) - {best_key}")
print("=========================================================")
placebo_r2 = run_placebo_test(dict((k, p) for k, _, p in runnable)[best_key])
print(f"\nMean placebo Test R2 (shuffled training target): {np.mean(placebo_r2):.4f} "
      f"(SD: {np.std(placebo_r2, ddof=1):.4f})")
print(f"Random-walk benchmark Test R2 (the placebo floor) : {naive['r2']:.4f}")
print(f"Actual model Test R2 ({best_key}, real target)    : {results[best_key]['r2']:.4f}")
if DIFFERENCE_TARGET:
    print("\nInterpretation: the level is reconstructed as (last actual level + predicted")
    print("change), so a random walk is embedded in every forecast and the placebo floor is")
    print("the RW R2 above, NOT zero. A placebo landing near the RW R2 is behaving as it")
    print("should. The damning outcomes are a placebo materially ABOVE the RW floor, or a")
    print("real-target R2 that fails to clear it.")
else:
    print("\nInterpretation: with an undifferenced target the placebo R2 should be near 0 or")
    print("negative. A meaningfully positive value means the setup can fit pure noise.")
pd.DataFrame({
    'repeat': list(range(1, len(placebo_r2) + 1)) + ['mean', 'rw_floor', 'real_target'],
    'placebo_r2': list(np.round(placebo_r2, 4)) + [round(float(np.mean(placebo_r2)), 4),
                                                   round(naive['r2'], 4),
                                                   round(results[best_key]['r2'], 4)],
}).to_csv(f"{OUT_P}/lstm_placebo_test.csv", index=False)

# =============================================================================
# BLOCK 22: PARAMETER COUNT vs SAMPLE SIZE
# =============================================================================
print("\n=========================================================")
print("  BLOCK 22: PARAMETER COUNT vs TRAINING SAMPLE SIZE")
print("=========================================================")
param_list = []
for nm, res in results.items():
    n_params = res['n_params']
    param_list.append({
        'Spec': nm,
        'N_Params': n_params,
        'N_Train_Sequences': res['n_train_seq'],
        'Params_per_Sample': round(n_params / res['n_train_seq'], 2),
    })
param_table = pd.DataFrame(param_list)
print(param_table.to_string(index=False))
param_table.to_csv(f"{OUT_P}/lstm_param_vs_sample_size.csv", index=False)

# =============================================================================
# BLOCK 23: COMBINED LEARNING CURVE PLOT
# =============================================================================
ncol = 2
nrow = int(np.ceil(len(results) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4 * nrow), squeeze=False)
axes = axes.flatten()
for idx, (nm, res) in enumerate(results.items()):
    ax = axes[idx]
    ax.plot(res['history'].history['loss'], label='Training', color='steelblue', linewidth=2)
    ax.plot(res['history'].history['val_loss'], label='Validation', color='tomato', linewidth=2)
    ax.set_title(res['label'], fontweight='bold')
    ax.set_ylabel("MSE Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc="upper right")
for j in range(len(results), len(axes)):
    axes[j].axis('off')
fig.suptitle(f"LSTM Learning Curves (seed {SEEDS[0]})\nTraining vs Validation Loss by Epoch")
plt.tight_layout()
plt.savefig(f"{OUT_P}/lstm_learning_curves_all_specs.png", dpi=150)
plt.close(fig)
print("\nSaved: lstm_learning_curves_all_specs.png")

print("\n\n=== LSTM PIPELINE COMPLETE ===")
print(f"Headline forecast written to {FORECAST_P}/lstm_headline_forecast.csv ({best_key})")
