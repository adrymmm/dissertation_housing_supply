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
reticulate::py_require("tensorflow")

library(keras3)
library(tensorflow)
library(lubridate)
library(tidyverse)
library(zoo)
library(Metrics)
library(ggplot2)
library(scales)

set.seed(42)
tensorflow::set_random_seed(42)

cat("TensorFlow version:", tf$`__version__`, "\n")

# =============================================================================
# BLOCK 2: IMPORT AND RENAME COLUMNS
# =============================================================================
df_raw <- read.csv(
  "/Users/madeleineng/Downloads/Dissertation/MSc Dissertation/MAY/england_master.csv",
  stringsAsFactors = FALSE
)

df_raw <- df_raw %>%
  rename(
    date     = X,
    lhstarts = starts,
    lrprc    = hprice,
    lrcc     = cc,
    lvol     = vol,
    lstock   = hstock,
    r3       = rate
  )

cat("--- Raw Data Preview ---\n")
print(head(df_raw[, c("date","lhstarts","lrprc","lrcc","lvol","lstock","r3")]))

# =============================================================================
# BLOCK 3: PARSE DATES — "1975Q1" FORMAT (NO SPACE)
# =============================================================================
df_raw$date <- as.Date(as.yearqtr(df_raw$date, format = "%YQ%q"))
df_raw      <- df_raw %>% arrange(date)

cat("\n--- Date Range ---\n")
cat("Start:", as.character(min(df_raw$date)), "\n")
cat("End:  ", as.character(max(df_raw$date)), "\n")
cat("Total observations:", nrow(df_raw), "\n")

# =============================================================================
# BLOCK 3b: DEFLATE NOMINAL VARIABLES USING A SINGLE DEFLATOR (cc_def)
# =============================================================================
# The dataset contains two deflator series, p_def and cc_def, originally
# intended for separate use (house prices vs construction costs). However,
# using two different deflators introduces inconsistency across variables
# and can distort cointegration and elasticity estimates.
#
# Following the data preparation team's instruction, a SINGLE deflator
# (cc_def, the GDP-based deflator) is applied consistently to BOTH the
# house price series (lrprc, still nominal at this point) and the
# construction cost series (lrcc, still nominal at this point).
#
# Standard deflation formula for a base-100 style price index:
#   real_value = (nominal_value / deflator) * 100
#
# This converts both series into real (inflation-adjusted) terms on a
# common deflator basis, consistent with the "lr" (log real) prefix used
# in the variable names lrprc and lrcc. Note: at this stage lrprc and lrcc
# are still NOMINAL LEVELS (renamed from hprice and cc in Block 2) - the
# "lr" naming reflects their intended final form after this deflation step
# and the log transform in Block 4.

df_raw <- df_raw %>%
  mutate(
    lrprc = (lrprc / cc_def) * 100,   # real house prices (GDP-deflated)
    lrcc  = (lrcc  / cc_def) * 100    # real construction costs (GDP-deflated)
  )

cat("\n--- After Deflation (using cc_def for both series) ---\n")
print(head(df_raw[, c("date", "lrprc", "lrcc", "cc_def")]))

# =============================================================================
# BLOCK 4: LOG-TRANSFORM RAW VARIABLES
# =============================================================================
df_raw <- df_raw %>%
  mutate(
    lhstarts = log(lhstarts),
    lrprc    = log(lrprc),
    lrcc     = log(lrcc),
    lvol     = log(lvol),
    lstock   = log(lstock)
  )

# =============================================================================
# BLOCK 5: CREATE ALL CANDIDATE DUMMY VARIABLES
# =============================================================================
# Structural dummies:
#   d_covid        (pulse: 2020Q1-2020Q3) - transitory COVID shutdown
#   d_crisis_pulse (pulse: 2022Q4-2023Q2) - acute mini-budget mortgage shock
#   d_crisis_step  (step:  2022Q4 onwards) - permanent post-crisis level shift
#
# Seasonal dummies (Q4 = reference category):
#   d_Q1, d_Q2, d_Q3

df_raw <- df_raw %>%
  mutate(
    d_covid = ifelse(
      date >= as.Date("2020-01-01") &
        date <= as.Date("2020-09-30"), 1, 0),
    
    d_crisis_pulse = ifelse(
      date >= as.Date("2022-10-01") &
        date <= as.Date("2023-06-30"), 1, 0),
    
    d_crisis_step = ifelse(
      date >= as.Date("2022-10-01"), 1, 0),
    
    cal_q = quarter(date),
    d_Q1  = ifelse(cal_q == 1, 1, 0),
    d_Q2  = ifelse(cal_q == 2, 1, 0),
    d_Q3  = ifelse(cal_q == 3, 1, 0)
  ) %>%
  select(-cal_q)

cat("\n--- Dummy Variable Verification ---\n")
cat("d_covid active quarters        :", sum(df_raw$d_covid), "\n")
cat("d_crisis_pulse active quarters :", sum(df_raw$d_crisis_pulse), "\n")
cat("d_crisis_step active quarters  :", sum(df_raw$d_crisis_step), "\n")
cat("d_Q1 active quarters           :", sum(df_raw$d_Q1), "\n")
cat("d_Q2 active quarters           :", sum(df_raw$d_Q2), "\n")
cat("d_Q3 active quarters           :", sum(df_raw$d_Q3), "\n")

# =============================================================================
# BLOCK 6: HANDLE MISSING VALUES (CONTINUOUS VARS ONLY)
# =============================================================================
target_var      <- "lhstarts"
continuous_vars <- c("lrprc", "lvol", "r3", "lstock", "lrcc")
all_dummy_vars  <- c("d_covid", "d_crisis_pulse", "d_crisis_step",
                     "d_Q1", "d_Q2", "d_Q3")

ts_cont  <- zoo(df_raw[, c(target_var, continuous_vars)], order.by = df_raw$date)
ts_cont  <- na.approx(ts_cont, na.rm = FALSE)
ts_cont  <- na.locf(ts_cont,  na.rm = FALSE)
ts_cont  <- na.locf(ts_cont,  fromLast = TRUE, na.rm = FALSE)

ts_dummy <- zoo(df_raw[, all_dummy_vars], order.by = df_raw$date)

ts_data  <- merge(ts_cont, ts_dummy)
dates    <- index(ts_data)
df_full  <- as.data.frame(ts_data)

cat("\n--- Missing Values After Imputation ---\n")
print(colSums(is.na(df_full)))

# =============================================================================
# BLOCK 7: TRAIN / TEST SPLIT (80/20 CHRONOLOGICAL) - SHARED ACROSS ALL SPECS
# =============================================================================
n_obs   <- nrow(df_full)
n_train <- floor(0.8 * n_obs)
n_test  <- n_obs - n_train
lookback <- 8

cat("\n--- Train / Test Split (shared across all 3 specs) ---\n")
cat("Total observations :", n_obs, "\n")
cat("Training (80%)     :", n_train, "| From", as.character(dates[1]),
    "to", as.character(dates[n_train]), "\n")
cat("Testing  (20%)     :", n_test,  "| From", as.character(dates[n_train + 1]),
    "to", as.character(dates[n_obs]), "\n")

test_dates <- dates[(n_train + 1):n_obs]

# =============================================================================
# BLOCK 8: REUSABLE FUNCTION — RUN ONE LSTM SPECIFICATION
# =============================================================================
# This function takes a vector of predictor variable names, builds the
# sequences, trains the LSTM, and returns predictions, metrics, and
# training history — so we can run it 3 times with different feature sets
# under IDENTICAL settings (lookback, architecture, callbacks, seed).

run_lstm_spec <- function(spec_name, predictor_vars, df_full, target_var,
                          continuous_vars, dates, n_train, n_obs, lookback) {
  
  cat("\n\n=====================================================\n")
  cat("  RUNNING SPEC:", spec_name, "\n")
  cat("  Predictors (", length(predictor_vars), "):",
      paste(predictor_vars, collapse = ", "), "\n")
  cat("=====================================================\n")
  
  all_vars <- c(target_var, predictor_vars)
  
  # --- Standardise continuous variables only (train-set parameters) ---
  train_raw    <- df_full[1:n_train, ]
  scale_params <- list()
  vars_to_scale <- c(target_var, continuous_vars)
  
  for (v in vars_to_scale) {
    scale_params[[v]] <- list(
      mean = mean(train_raw[[v]], na.rm = TRUE),
      sd   = sd(train_raw[[v]],   na.rm = TRUE)
    )
  }
  
  df_scaled <- df_full
  for (v in vars_to_scale) {
    df_scaled[[v]] <- (df_full[[v]] - scale_params[[v]]$mean) / scale_params[[v]]$sd
  }
  # Dummy columns (if any in predictor_vars) remain raw 0/1
  
  # --- Create sequences ---
  create_sequences <- function(data_matrix, target_col_idx, lookback) {
    n          <- nrow(data_matrix)
    n_features <- ncol(data_matrix)
    n_samples  <- n - lookback
    
    X <- array(0, dim = c(n_samples, lookback, n_features))
    y <- numeric(n_samples)
    
    for (i in seq_len(n_samples)) {
      X[i, , ] <- data_matrix[i:(i + lookback - 1), ]
      y[i]     <- data_matrix[i + lookback, target_col_idx]
    }
    list(X = X, y = y)
  }
  
  data_matrix    <- as.matrix(df_scaled[, all_vars])
  target_col_idx <- which(colnames(data_matrix) == target_var)
  seqs           <- create_sequences(data_matrix, target_col_idx, lookback)
  
  X_all <- seqs$X
  y_all <- seqs$y
  
  n_train_seq <- n_train - lookback
  
  X_train <- X_all[1:n_train_seq, , , drop = FALSE]
  y_train <- y_all[1:n_train_seq]
  
  X_test  <- X_all[(n_train_seq + 1):dim(X_all)[1], , , drop = FALSE]
  y_test  <- y_all[(n_train_seq + 1):length(y_all)]
  
  n_features <- dim(X_train)[3]
  cat("Input features:", n_features, "| Train seqs:", nrow(X_train),
      "| Test seqs:", nrow(X_test), "\n")
  
  # --- Reset seed before each run for fair comparison ---
  set.seed(42)
  tensorflow::set_random_seed(42)
  
  # --- Build model (IDENTICAL architecture across all specs) ---
  model <- keras_model_sequential(name = paste0("LSTM_", spec_name)) %>%
    layer_lstm(
      units             = 64,
      input_shape       = c(lookback, n_features),
      dropout           = 0.1,
      recurrent_dropout = 0.1,
      return_sequences  = FALSE
    ) %>%
    layer_dense(units = 1, activation = "linear")
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss      = "mean_squared_error",
    metrics   = list("mean_absolute_error")
  )
  
  early_stop <- callback_early_stopping(
    monitor = "val_loss", patience = 30,
    restore_best_weights = TRUE, verbose = 0
  )
  reduce_lr <- callback_reduce_lr_on_plateau(
    monitor = "val_loss", factor = 0.5,
    patience = 15, min_lr = 1e-6, verbose = 0
  )
  
  # --- Train ---
  history <- model %>% fit(
    x = X_train, y = y_train,
    epochs = 300, batch_size = 16,
    validation_split = 0.15,
    callbacks = list(early_stop, reduce_lr),
    shuffle = FALSE, verbose = 0
  )
  
  cat("Epochs run:", length(history$metrics$loss), "\n")
  
  # --- Predict and invert scaling ---
  y_pred_scaled <- as.vector(predict(model, X_test, verbose = 0))
  
  invert_scale <- function(scaled_vals, var_name, params) {
    scaled_vals * params[[var_name]]$sd + params[[var_name]]$mean
  }
  
  y_pred <- invert_scale(y_pred_scaled, target_var, scale_params)
  y_true <- invert_scale(y_test,        target_var, scale_params)
  
  # --- Metrics ---
  rmse_val <- rmse(y_true, y_pred)
  mae_val  <- mae(y_true,  y_pred)
  mape_val <- mape(y_true, y_pred) * 100
  
  # R-squared: proportion of variance in actual values explained by predictions
  # R2 = 1 - (SS_residual / SS_total)
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  r2_val <- 1 - (ss_res / ss_tot)
  
  cat(sprintf("RMSE: %.4f | MAE: %.4f | MAPE: %.2f%% | R-squared: %.4f\n",
              rmse_val, mae_val, mape_val, r2_val))
  
  list(
    spec_name = spec_name,
    y_true    = y_true,
    y_pred    = y_pred,
    rmse      = rmse_val,
    mae       = mae_val,
    mape      = mape_val,
    r2        = r2_val,
    history   = history,
    n_features = n_features
  )
}

# =============================================================================
# BLOCK 9: DEFINE THE FOUR FEATURE SETS
# =============================================================================
predictors_spec1 <- continuous_vars                                    # no dummy
predictors_spec2 <- c(continuous_vars, "d_covid", "d_crisis_pulse",
                      "d_crisis_step")                                 # + crisis dummies
predictors_spec3 <- c(continuous_vars, "d_Q1", "d_Q2", "d_Q3")          # + seasonal dummies only
predictors_spec4 <- c(continuous_vars, "d_covid", "d_crisis_pulse",
                      "d_crisis_step", "d_Q1", "d_Q2", "d_Q3")         # + crisis + seasonal

# =============================================================================
# BLOCK 10: RUN ALL FOUR SPECIFICATIONS
# =============================================================================
result1 <- run_lstm_spec("Spec1_NoDummy",         predictors_spec1, df_full,
                         target_var, continuous_vars, dates, n_train, n_obs, lookback)

result2 <- run_lstm_spec("Spec2_CrisisDummy",     predictors_spec2, df_full,
                         target_var, continuous_vars, dates, n_train, n_obs, lookback)

result3 <- run_lstm_spec("Spec3_SeasonalDummy",   predictors_spec3, df_full,
                         target_var, continuous_vars, dates, n_train, n_obs, lookback)

result4 <- run_lstm_spec("Spec4_CrisisSeasonal",  predictors_spec4, df_full,
                         target_var, continuous_vars, dates, n_train, n_obs, lookback)

# =============================================================================
# BLOCK 11: COMPARISON TABLE
# =============================================================================
comparison_table <- data.frame(
  Specification = c("Spec 1: No dummy",
                    "Spec 2: + Crisis dummies (covid, crisis pulse, crisis step)",
                    "Spec 3: + Seasonal dummies (Q1-Q3)",
                    "Spec 4: + Crisis + Seasonal dummies"),
  Features = c(result1$n_features, result2$n_features,
               result3$n_features, result4$n_features),
  RMSE  = round(c(result1$rmse, result2$rmse, result3$rmse, result4$rmse), 4),
  MAE   = round(c(result1$mae,  result2$mae,  result3$mae,  result4$mae),  4),
  MAPE  = round(c(result1$mape, result2$mape, result3$mape, result4$mape), 2),
  R2    = round(c(result1$r2,   result2$r2,   result3$r2,   result4$r2),   4)
)

cat("\n\n=========================================================\n")
cat("  COMPARISON TABLE: 4 LSTM SPECIFICATIONS\n")
cat("=========================================================\n")
print(comparison_table)

write.csv(comparison_table, "lstm_spec_comparison.csv", row.names = FALSE)

# =============================================================================
# BLOCK 12: COMBINED ACTUAL VS PREDICTED PLOTS
#   Plot Group A: Spec 1 (No Dummy) vs Spec 2 (+ Crisis Dummies)
#   Plot Group B: Spec 3 (+ Seasonal) vs Spec 4 (+ Crisis + Seasonal)
# =============================================================================

build_avp_data <- function(result, spec_label) {
  bind_rows(
    data.frame(Date = test_dates, Value = result$y_true,
               Series = "Actual", Spec = spec_label),
    data.frame(Date = test_dates, Value = result$y_pred,
               Series = "Predicted", Spec = spec_label)
  )
}

make_avp_plot <- function(plot_data, title_text) {
  ggplot(plot_data, aes(x = Date, y = Value,
                        colour = Series, linetype = Series)) +
    geom_line(linewidth = 0.8) +
    scale_colour_manual(values = c("Actual" = "black", "Predicted" = "steelblue")) +
    scale_linetype_manual(values = c("Actual" = "solid", "Predicted" = "dashed")) +
    facet_wrap(~ Spec, ncol = 1) +
    labs(
      title = title_text,
      subtitle = "Actual vs Predicted Log Housing Starts (Out-of-Sample)",
      x = "Date", y = "Log Housing Starts (lhstarts)",
      colour = NULL, linetype = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom",
          strip.text = element_text(face = "bold"))
}

# --- Group A: No Dummy vs Crisis Dummies ---
plot_data_A <- bind_rows(
  build_avp_data(result1, "Spec 1: No Dummy"),
  build_avp_data(result2, "Spec 2: + Crisis Dummies")
)
plot_data_A$Spec <- factor(plot_data_A$Spec,
                           levels = c("Spec 1: No Dummy", "Spec 2: + Crisis Dummies"))

p_avp_A <- make_avp_plot(plot_data_A, "LSTM Forecast Comparison: No Dummy vs Crisis Dummies")
print(p_avp_A)
ggsave("lstm_avp_groupA_nodummy_vs_crisis.png", p_avp_A,
       width = 10, height = 7, dpi = 150)

# --- Group B: Seasonal vs Crisis + Seasonal ---
plot_data_B <- bind_rows(
  build_avp_data(result3, "Spec 3: + Seasonal Dummies"),
  build_avp_data(result4, "Spec 4: + Crisis + Seasonal Dummies")
)
plot_data_B$Spec <- factor(plot_data_B$Spec,
                           levels = c("Spec 3: + Seasonal Dummies",
                                      "Spec 4: + Crisis + Seasonal Dummies"))

p_avp_B <- make_avp_plot(plot_data_B, "LSTM Forecast Comparison: Seasonal vs Crisis + Seasonal Dummies")
print(p_avp_B)
ggsave("lstm_avp_groupB_seasonal_vs_crisisseasonal.png", p_avp_B,
       width = 10, height = 7, dpi = 150)

# =============================================================================
# BLOCK 13: COMBINED RESIDUAL PLOTS (SAME GROUPING)
# =============================================================================

build_resid_data <- function(result, spec_label) {
  data.frame(Date = test_dates, Residual = result$y_true - result$y_pred,
             Spec = spec_label)
}

make_resid_plot <- function(plot_data, title_text) {
  ggplot(plot_data, aes(x = Date, y = Residual)) +
    geom_line(colour = "darkred", linewidth = 0.7) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
    facet_wrap(~ Spec, ncol = 1) +
    labs(
      title = title_text,
      subtitle = "Residual = Actual - Predicted",
      x = "Date", y = "Residual"
    ) +
    theme_minimal(base_size = 12) +
    theme(strip.text = element_text(face = "bold"))
}

# --- Group A residuals ---
resid_data_A <- bind_rows(
  build_resid_data(result1, "Spec 1: No Dummy"),
  build_resid_data(result2, "Spec 2: + Crisis Dummies")
)
resid_data_A$Spec <- factor(resid_data_A$Spec,
                            levels = c("Spec 1: No Dummy", "Spec 2: + Crisis Dummies"))

p_resid_A <- make_resid_plot(resid_data_A, "LSTM Residuals: No Dummy vs Crisis Dummies")
print(p_resid_A)
ggsave("lstm_resid_groupA_nodummy_vs_crisis.png", p_resid_A,
       width = 10, height = 7, dpi = 150)

# --- Group B residuals ---
resid_data_B <- bind_rows(
  build_resid_data(result3, "Spec 3: + Seasonal Dummies"),
  build_resid_data(result4, "Spec 4: + Crisis + Seasonal Dummies")
)
resid_data_B$Spec <- factor(resid_data_B$Spec,
                            levels = c("Spec 3: + Seasonal Dummies",
                                       "Spec 4: + Crisis + Seasonal Dummies"))

p_resid_B <- make_resid_plot(resid_data_B, "LSTM Residuals: Seasonal vs Crisis + Seasonal Dummies")
print(p_resid_B)
ggsave("lstm_resid_groupB_seasonal_vs_crisisseasonal.png", p_resid_B,
       width = 10, height = 7, dpi = 150)

# =============================================================================
# BLOCK 14: COMBINED TRAINING LOSS PLOTS (SAME GROUPING)
# =============================================================================

build_loss_data <- function(result, spec_label) {
  bind_rows(
    data.frame(Epoch = seq_along(result$history$metrics$loss),
               Loss = result$history$metrics$loss,
               Type = "Training", Spec = spec_label),
    data.frame(Epoch = seq_along(result$history$metrics$val_loss),
               Loss = result$history$metrics$val_loss,
               Type = "Validation", Spec = spec_label)
  )
}

make_loss_plot <- function(plot_data, title_text) {
  ggplot(plot_data, aes(x = Epoch, y = Loss, colour = Type)) +
    geom_line(linewidth = 0.8) +
    scale_colour_manual(values = c("Training" = "steelblue", "Validation" = "tomato")) +
    facet_wrap(~ Spec, ncol = 1, scales = "free_x") +
    labs(
      title = title_text,
      x = "Epoch", y = "MSE Loss", colour = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom",
          strip.text = element_text(face = "bold"))
}

# --- Group A training loss ---
loss_data_A <- bind_rows(
  build_loss_data(result1, "Spec 1: No Dummy"),
  build_loss_data(result2, "Spec 2: + Crisis Dummies")
)
loss_data_A$Spec <- factor(loss_data_A$Spec,
                           levels = c("Spec 1: No Dummy", "Spec 2: + Crisis Dummies"))

p_loss_A <- make_loss_plot(loss_data_A, "LSTM Training History: No Dummy vs Crisis Dummies")
print(p_loss_A)
ggsave("lstm_loss_groupA_nodummy_vs_crisis.png", p_loss_A,
       width = 9, height = 7, dpi = 150)

# --- Group B training loss ---
loss_data_B <- bind_rows(
  build_loss_data(result3, "Spec 3: + Seasonal Dummies"),
  build_loss_data(result4, "Spec 4: + Crisis + Seasonal Dummies")
)
loss_data_B$Spec <- factor(loss_data_B$Spec,
                           levels = c("Spec 3: + Seasonal Dummies",
                                      "Spec 4: + Crisis + Seasonal Dummies"))

p_loss_B <- make_loss_plot(loss_data_B, "LSTM Training History: Seasonal vs Crisis + Seasonal Dummies")
print(p_loss_B)
ggsave("lstm_loss_groupB_seasonal_vs_crisisseasonal.png", p_loss_B,
       width = 9, height = 7, dpi = 150)

# =============================================================================
# BLOCK 15: SAVE INDIVIDUAL FORECASTS
# =============================================================================
save_forecast <- function(result, filename) {
  df_out <- data.frame(
    Date      = test_dates,
    Actual    = round(result$y_true, 4),
    Predicted = round(result$y_pred, 4),
    Residual  = round(result$y_true - result$y_pred, 4)
  )
  write.csv(df_out, filename, row.names = FALSE)
}

save_forecast(result1, "lstm_spec1_nodummy_forecasts.csv")
save_forecast(result2, "lstm_spec2_crisis_forecasts.csv")
save_forecast(result3, "lstm_spec3_seasonal_forecasts.csv")
save_forecast(result4, "lstm_spec4_crisis_seasonal_forecasts.csv")

# =============================================================================
# BLOCK 16: TIME SERIES CROSS-VALIDATION (ROLLING ORIGIN) - ALL 4 SPECS
# =============================================================================
# Standard k-fold CV is invalid for time series (it shuffles and causes
# data leakage from future to past). Instead we use "rolling origin" /
# expanding window cross-validation (Cerqueira et al., 2020; Bergmeir et al.,
# 2018): the training window expands forward in time across several folds,
# each time testing on the next unseen block of quarters.
#
# This assesses how STABLE each specification's performance is across
# different time periods, rather than relying on a single 80/20 split.
# Run for ALL 4 specifications to compare both average performance and
# stability (mean +/- SD across folds) across the full model set.

n_folds <- 5

run_cv_spec <- function(spec_name, predictor_vars, df_full, target_var,
                        continuous_vars, n_obs, lookback, n_folds) {
  
  cat("\n\n=========================================================\n")
  cat("  TIME SERIES CROSS-VALIDATION (ROLLING ORIGIN) -", spec_name, "\n")
  cat("=========================================================\n")
  
  all_vars_cv <- c(target_var, predictor_vars)
  
  # Standardise using first 60% as a stable reference (computational
  # compromise for small samples - avoids re-standardising per fold)
  cv_scale_base <- floor(0.6 * n_obs)
  cv_train_raw  <- df_full[1:cv_scale_base, ]
  
  cv_scale_params <- list()
  for (v in c(target_var, continuous_vars)) {
    cv_scale_params[[v]] <- list(
      mean = mean(cv_train_raw[[v]], na.rm = TRUE),
      sd   = sd(cv_train_raw[[v]],   na.rm = TRUE)
    )
  }
  
  df_scaled_cv <- df_full
  for (v in c(target_var, continuous_vars)) {
    df_scaled_cv[[v]] <- (df_full[[v]] - cv_scale_params[[v]]$mean) / cv_scale_params[[v]]$sd
  }
  
  data_matrix_cv    <- as.matrix(df_scaled_cv[, all_vars_cv])
  target_col_idx_cv <- which(colnames(data_matrix_cv) == target_var)
  
  create_sequences_cv <- function(data_matrix, target_col_idx, lookback) {
    n          <- nrow(data_matrix)
    n_features <- ncol(data_matrix)
    n_samples  <- n - lookback
    X <- array(0, dim = c(n_samples, lookback, n_features))
    y <- numeric(n_samples)
    for (i in seq_len(n_samples)) {
      X[i, , ] <- data_matrix[i:(i + lookback - 1), ]
      y[i]     <- data_matrix[i + lookback, target_col_idx]
    }
    list(X = X, y = y)
  }
  
  seqs_cv  <- create_sequences_cv(data_matrix_cv, target_col_idx_cv, lookback)
  X_all_cv <- seqs_cv$X
  y_all_cv <- seqs_cv$y
  
  n_seq_total <- length(y_all_cv)
  
  # Expanding window: start with first 50% as initial training, then
  # add 1/n_folds of the remaining data per fold as new test block
  initial_train <- floor(0.5 * n_seq_total)
  remaining     <- n_seq_total - initial_train
  fold_size     <- floor(remaining / n_folds)
  
  cv_results <- data.frame()
  
  for (fold in 1:n_folds) {
    
    train_end  <- initial_train + (fold - 1) * fold_size
    test_start <- train_end + 1
    test_end   <- if (fold == n_folds) n_seq_total else train_end + fold_size
    
    if (test_start > test_end) next
    
    X_tr <- X_all_cv[1:train_end, , , drop = FALSE]
    y_tr <- y_all_cv[1:train_end]
    X_te <- X_all_cv[test_start:test_end, , , drop = FALSE]
    y_te <- y_all_cv[test_start:test_end]
    
    set.seed(42)
    tensorflow::set_random_seed(42)
    
    model_cv <- keras_model_sequential() %>%
      layer_lstm(
        units = 64,
        input_shape = c(lookback, dim(X_tr)[3]),
        dropout = 0.1, recurrent_dropout = 0.1,
        return_sequences = FALSE
      ) %>%
      layer_dense(units = 1, activation = "linear")
    
    model_cv %>% compile(
      optimizer = optimizer_adam(learning_rate = 0.001),
      loss = "mean_squared_error"
    )
    
    es_cv <- callback_early_stopping(monitor = "val_loss", patience = 20,
                                     restore_best_weights = TRUE, verbose = 0)
    rl_cv <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5,
                                           patience = 10, min_lr = 1e-6, verbose = 0)
    
    invisible(capture.output(
      model_cv %>% fit(
        x = X_tr, y = y_tr, epochs = 200, batch_size = 16,
        validation_split = 0.15,
        callbacks = list(es_cv, rl_cv),
        shuffle = FALSE, verbose = 0
      )
    ))
    
    pred_te <- as.vector(predict(model_cv, X_te, verbose = 0))
    
    inv <- function(x) x * cv_scale_params[[target_var]]$sd + cv_scale_params[[target_var]]$mean
    pred_te_orig <- inv(pred_te)
    y_te_orig    <- inv(y_te)
    
    fold_rmse   <- rmse(y_te_orig, pred_te_orig)
    fold_mae    <- mae(y_te_orig, pred_te_orig)
    fold_mape   <- mape(y_te_orig, pred_te_orig) * 100
    fold_ss_res <- sum((y_te_orig - pred_te_orig)^2)
    fold_ss_tot <- sum((y_te_orig - mean(y_te_orig))^2)
    fold_r2     <- 1 - (fold_ss_res / fold_ss_tot)
    
    cat(sprintf("Fold %d | Train n=%d, Test n=%d | RMSE=%.4f MAE=%.4f MAPE=%.2f%% R2=%.4f\n",
                fold, train_end, length(y_te), fold_rmse, fold_mae, fold_mape, fold_r2))
    
    cv_results <- rbind(cv_results, data.frame(
      Spec = spec_name, Fold = fold, Train_N = train_end, Test_N = length(y_te),
      RMSE = round(fold_rmse, 4), MAE = round(fold_mae, 4),
      MAPE = round(fold_mape, 2), R2 = round(fold_r2, 4)
    ))
  }
  
  cv_summary <- data.frame(
    Spec   = spec_name,
    Metric = c("RMSE", "MAE", "MAPE", "R2"),
    Mean   = round(c(mean(cv_results$RMSE), mean(cv_results$MAE),
                     mean(cv_results$MAPE), mean(cv_results$R2)), 4),
    SD     = round(c(sd(cv_results$RMSE), sd(cv_results$MAE),
                     sd(cv_results$MAPE), sd(cv_results$R2)), 4)
  )
  
  cat("\n--- Cross-Validation Mean +/- SD Across Folds (", spec_name, ") ---\n")
  print(cv_summary)
  
  list(folds = cv_results, summary = cv_summary)
}

# --- Run CV for all 4 specifications ---
cv_spec1 <- run_cv_spec("Spec1_NoDummy",        predictors_spec1, df_full,
                        target_var, continuous_vars, n_obs, lookback, n_folds)
cv_spec2 <- run_cv_spec("Spec2_CrisisDummy",    predictors_spec2, df_full,
                        target_var, continuous_vars, n_obs, lookback, n_folds)
cv_spec3 <- run_cv_spec("Spec3_SeasonalDummy",  predictors_spec3, df_full,
                        target_var, continuous_vars, n_obs, lookback, n_folds)
cv_spec4 <- run_cv_spec("Spec4_CrisisSeasonal", predictors_spec4, df_full,
                        target_var, continuous_vars, n_obs, lookback, n_folds)

# --- Combine all fold-level results ---
cv_folds_all <- bind_rows(cv_spec1$folds, cv_spec2$folds,
                          cv_spec3$folds, cv_spec4$folds)

# --- Combine all summary results into one comparison table ---
cv_summary_all <- bind_rows(cv_spec1$summary, cv_spec2$summary,
                            cv_spec3$summary, cv_spec4$summary)

cat("\n\n=========================================================\n")
cat("  CROSS-VALIDATION SUMMARY: ALL 4 SPECS (Mean +/- SD)\n")
cat("=========================================================\n")
print(cv_summary_all)

# Reshape to wide format for easy comparison: one row per spec,
# columns for Mean RMSE, SD RMSE, Mean R2, SD R2 etc.
cv_summary_wide <- cv_summary_all %>%
  pivot_wider(
    id_cols = Spec,
    names_from = Metric,
    values_from = c(Mean, SD)
  )

cat("\n--- Cross-Validation Comparison Table (Wide Format) ---\n")
print(cv_summary_wide)

write.csv(cv_folds_all,    "lstm_cv_folds_all_specs.csv",   row.names = FALSE)
write.csv(cv_summary_all,  "lstm_cv_summary_all_specs.csv", row.names = FALSE)
write.csv(cv_summary_wide, "lstm_cv_summary_wide.csv",      row.names = FALSE)

cat("\nSaved: lstm_cv_folds_all_specs.csv\n")
cat("Saved: lstm_cv_summary_all_specs.csv\n")
cat("Saved: lstm_cv_summary_wide.csv\n")

cat("\n\n=== ALL FOUR SPECIFICATIONS + CROSS-VALIDATION COMPLETE ===\n")
cat("Saved: lstm_spec_comparison.csv\n")
cat("Saved: lstm_spec1_nodummy_forecasts.csv\n")
cat("Saved: lstm_spec2_crisis_forecasts.csv\n")
cat("Saved: lstm_spec3_seasonal_forecasts.csv\n")
cat("Saved: lstm_spec4_crisis_seasonal_forecasts.csv\n")
cat("Saved: lstm_avp_groupA_nodummy_vs_crisis.png\n")
cat("Saved: lstm_avp_groupB_seasonal_vs_crisisseasonal.png\n")
cat("Saved: lstm_resid_groupA_nodummy_vs_crisis.png\n")
cat("Saved: lstm_resid_groupB_seasonal_vs_crisisseasonal.png\n")
cat("Saved: lstm_loss_groupA_nodummy_vs_crisis.png\n")
cat("Saved: lstm_loss_groupB_seasonal_vs_crisisseasonal.png\n")



# =============================================================================
# OVERFITTING / NOISE-FITTING DIAGNOSTICS FOR YOUR LSTM PIPELINE
#
# HOW TO USE:
#   1. Apply the two small edits to run_lstm_spec() marked "*** EDIT ***"
#      below (these just add train-set predictions + the model object to
#      the returned list - nothing about your training changes).
#   2. Re-run Blocks 1-11 from your original script as normal.
#   3. Paste/run this file afterwards (Block 17 onward). It reuses
#      result1..result4, df_full, dates, n_train, target_var, etc.
# =============================================================================

# -----------------------------------------------------------------------
# *** EDIT 1 *** - inside run_lstm_spec(), right after:
#   y_pred_scaled <- as.vector(predict(model, X_test, verbose = 0))
# ADD:
#   y_train_pred_scaled <- as.vector(predict(model, X_train, verbose = 0))
#
# *** EDIT 2 *** - inside run_lstm_spec(), in the final `list(...)` that is
# returned, ADD these two entries:
#   model         = model,
#   y_train_true  = invert_scale(y_train, target_var, scale_params),
#   y_train_pred  = invert_scale(y_train_pred_scaled, target_var, scale_params)
#
# (y_train is already computed earlier in the function as the training
# target vector, so no extra work needed there.)
# -----------------------------------------------------------------------

library(dplyr)
library(Metrics)

all_results <- list(Spec1_NoDummy = result1, Spec2_CrisisDummy = result2,
                    Spec3_SeasonalDummy = result3, Spec4_CrisisSeasonal = result4)

# =============================================================================
# BLOCK 17: TRAIN vs TEST PERFORMANCE GAP (the core overfitting check)
# =============================================================================
# Rule of thumb: if Test R2 is much lower than Train R2 (e.g. Train R2 = 0.95,
# Test R2 = 0.30), the model has memorised the training sequences rather than
# learning generalisable structure.

overfit_gap_table <- lapply(names(all_results), function(nm) {
  res <- all_results[[nm]]
  
  train_rmse <- rmse(res$y_train_true, res$y_train_pred)
  train_r2   <- 1 - sum((res$y_train_true - res$y_train_pred)^2) /
    sum((res$y_train_true - mean(res$y_train_true))^2)
  
  data.frame(
    Spec        = nm,
    Train_RMSE  = round(train_rmse, 4),
    Test_RMSE   = round(res$rmse, 4),
    RMSE_Ratio  = round(res$rmse / train_rmse, 2),   # >1.5-2x is a warning sign
    Train_R2    = round(train_r2, 4),
    Test_R2     = round(res$r2, 4),
    R2_Gap      = round(train_r2 - res$r2, 4)         # large positive gap = overfitting
  )
}) %>% bind_rows()

cat("\n=========================================================\n")
cat("  BLOCK 17: TRAIN vs TEST PERFORMANCE GAP\n")
cat("=========================================================\n")
print(overfit_gap_table)
write.csv(overfit_gap_table, "lstm_overfit_gap_table.csv", row.names = FALSE)

# Interpretation guide printed for convenience
cat("\nInterpretation:\n")
cat(" - RMSE_Ratio close to 1 = good generalisation.\n")
cat(" - RMSE_Ratio >> 1.5-2   = overfitting to the training window.\n")
cat(" - R2_Gap near 0         = good. R2_Gap > 0.3-0.4 = overfitting.\n")
cat(" - Negative Test_R2      = model performs worse than predicting the mean;\n")
cat("   strong sign it is fitting noise, not signal.\n")

# =============================================================================
# BLOCK 18: LEARNING CURVE GAP (train vs val loss at convergence)
# =============================================================================
learning_curve_gap <- lapply(names(all_results), function(nm) {
  h <- all_results[[nm]]$history$metrics
  n <- length(h$loss)
  # average of last 10% of epochs, to avoid noise from a single epoch
  tail_n <- max(1, floor(0.1 * n))
  final_train_loss <- mean(tail(h$loss, tail_n))
  final_val_loss   <- mean(tail(h$val_loss, tail_n))
  data.frame(
    Spec            = nm,
    Epochs_Run      = n,
    Final_Train_Loss = round(final_train_loss, 5),
    Final_Val_Loss   = round(final_val_loss, 5),
    Val_Train_Ratio  = round(final_val_loss / final_train_loss, 2)
  )
}) %>% bind_rows()

cat("\n=========================================================\n")
cat("  BLOCK 18: LEARNING CURVE GAP (VAL LOSS / TRAIN LOSS)\n")
cat("=========================================================\n")
print(learning_curve_gap)
write.csv(learning_curve_gap, "lstm_learning_curve_gap.csv", row.names = FALSE)
cat("\nVal_Train_Ratio near 1 = good. Consistently >1.5-2 across training = overfitting.\n")
cat("If val_loss diverges upward while train_loss keeps falling, that's the\n")
cat("textbook overfitting signature - check your existing loss plots for this shape.\n")

# =============================================================================
# BLOCK 19: NAIVE BENCHMARK COMPARISON (random-walk / persistence forecast)
# =============================================================================
# If your LSTM cannot beat "next quarter = this quarter" on the SAME test
# window, its apparent skill is likely noise, not learned signal.

naive_benchmark <- df_full[[target_var]]
naive_test_actual <- naive_benchmark[(n_train + 1):n_obs]
# naive prediction = value at t-1 (lagged by 1 quarter), aligned to test_dates
naive_test_pred <- naive_benchmark[n_train:(n_obs - 1)]

naive_rmse <- rmse(naive_test_actual, naive_test_pred)
naive_mae  <- mae(naive_test_actual, naive_test_pred)

cat("\n=========================================================\n")
cat("  BLOCK 19: LSTM vs NAIVE (RANDOM-WALK) BENCHMARK\n")
cat("=========================================================\n")
cat(sprintf("Naive persistence RMSE: %.4f | MAE: %.4f\n", naive_rmse, naive_mae))

naive_comparison <- lapply(names(all_results), function(nm) {
  res <- all_results[[nm]]
  data.frame(
    Spec        = nm,
    LSTM_RMSE   = round(res$rmse, 4),
    Naive_RMSE  = round(naive_rmse, 4),
    Skill_Ratio = round(1 - (res$rmse / naive_rmse), 4)  # >0 = beats naive; <=0 = doesn't
  )
}) %>% bind_rows()

print(naive_comparison)
write.csv(naive_comparison, "lstm_naive_benchmark_comparison.csv", row.names = FALSE)
cat("\nSkill_Ratio > 0 means the LSTM beats a trivial persistence forecast.\n")
cat("Skill_Ratio <= 0 is a red flag: the model adds no value over 'no change',\n")
cat("meaning any apparent in-sample fit is likely noise.\n")

# =============================================================================
# BLOCK 20: RESIDUAL AUTOCORRELATION (Ljung-Box test)
# =============================================================================
# Tests whether residuals are white noise (good) vs. still contain
# structure the model failed to capture (bad fit, not necessarily overfit).

ljung_box_table <- lapply(names(all_results), function(nm) {
  res <- all_results[[nm]]
  resid <- res$y_true - res$y_pred
  lb <- Box.test(resid, lag = min(4, length(resid) - 1), type = "Ljung-Box")
  data.frame(Spec = nm, LB_statistic = round(lb$statistic, 3),
             p_value = round(lb$p.value, 4))
}) %>% bind_rows()

cat("\n=========================================================\n")
cat("  BLOCK 20: LJUNG-BOX TEST ON TEST RESIDUALS\n")
cat("=========================================================\n")
print(ljung_box_table)
write.csv(ljung_box_table, "lstm_ljung_box_residuals.csv", row.names = FALSE)
cat("\np-value < 0.05: residuals show significant autocorrelation - the model\n")
cat("left exploitable structure on the table (underfitting some signal).\n")
cat("p-value > 0.05: residuals look like white noise - no leftover structure.\n")

# =============================================================================
# BLOCK 21: PLACEBO / SHUFFLED-TARGET TEST (the strongest overfitting check)
# =============================================================================
# Refit the SAME architecture on a randomly shuffled target. A model that is
# genuinely learning signal should perform close to random (R2 near 0 or
# negative) on shuffled data. If it still achieves a suspiciously good test
# R2 here, the architecture/training setup is capable of fitting pure noise,
# which means your real result may be doing the same thing.
#
# NOTE: only run this for your best/main specification (e.g. Spec 4) - it
# retrains a model so it is slow. Loop over all specs if you have time.

run_placebo_test <- function(predictor_vars, df_full, target_var,
                             continuous_vars, n_train, n_obs, lookback,
                             n_repeats = 3) {
  
  set.seed(123)
  placebo_r2s <- numeric(n_repeats)
  
  for (r in 1:n_repeats) {
    
    df_shuffled <- df_full
    # shuffle ONLY the target column, within the training portion, to break
    # any true relationship with predictors while preserving predictor structure
    df_shuffled[[target_var]] <- sample(df_full[[target_var]])
    
    all_vars <- c(target_var, predictor_vars)
    train_raw <- df_shuffled[1:n_train, ]
    scale_params <- list()
    for (v in c(target_var, continuous_vars)) {
      scale_params[[v]] <- list(mean = mean(train_raw[[v]], na.rm = TRUE),
                                sd = sd(train_raw[[v]], na.rm = TRUE))
    }
    df_scaled <- df_shuffled
    for (v in c(target_var, continuous_vars)) {
      df_scaled[[v]] <- (df_shuffled[[v]] - scale_params[[v]]$mean) / scale_params[[v]]$sd
    }
    
    create_sequences <- function(data_matrix, target_col_idx, lookback) {
      n <- nrow(data_matrix); n_features <- ncol(data_matrix); n_samples <- n - lookback
      X <- array(0, dim = c(n_samples, lookback, n_features)); y <- numeric(n_samples)
      for (i in seq_len(n_samples)) {
        X[i, , ] <- data_matrix[i:(i + lookback - 1), ]
        y[i] <- data_matrix[i + lookback, target_col_idx]
      }
      list(X = X, y = y)
    }
    
    data_matrix <- as.matrix(df_scaled[, all_vars])
    target_col_idx <- which(colnames(data_matrix) == target_var)
    seqs <- create_sequences(data_matrix, target_col_idx, lookback)
    
    n_train_seq <- n_train - lookback
    X_train <- seqs$X[1:n_train_seq, , , drop = FALSE]
    y_train <- seqs$y[1:n_train_seq]
    X_test  <- seqs$X[(n_train_seq + 1):dim(seqs$X)[1], , , drop = FALSE]
    y_test  <- seqs$y[(n_train_seq + 1):length(seqs$y)]
    
    tensorflow::set_random_seed(42)
    model <- keras_model_sequential() %>%
      layer_lstm(units = 64, input_shape = c(lookback, dim(X_train)[3]),
                 dropout = 0.1, recurrent_dropout = 0.1, return_sequences = FALSE) %>%
      layer_dense(units = 1, activation = "linear")
    
    model %>% compile(optimizer = optimizer_adam(learning_rate = 0.001),
                      loss = "mean_squared_error")
    
    invisible(capture.output(
      model %>% fit(x = X_train, y = y_train, epochs = 150, batch_size = 16,
                    validation_split = 0.15,
                    callbacks = list(
                      callback_early_stopping(monitor = "val_loss", patience = 20,
                                              restore_best_weights = TRUE),
                      callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5,
                                                    patience = 10, min_lr = 1e-6)
                    ),
                    shuffle = FALSE, verbose = 0)
    ))
    
    pred <- as.vector(predict(model, X_test, verbose = 0))
    inv <- function(x) x * scale_params[[target_var]]$sd + scale_params[[target_var]]$mean
    pred_o <- inv(pred); y_o <- inv(y_test)
    
    ss_res <- sum((y_o - pred_o)^2); ss_tot <- sum((y_o - mean(y_o))^2)
    placebo_r2s[r] <- 1 - ss_res / ss_tot
    
    cat(sprintf("  Placebo repeat %d/%d: shuffled-target Test R2 = %.4f\n",
                r, n_repeats, placebo_r2s[r]))
  }
  placebo_r2s
}

cat("\n=========================================================\n")
cat("  BLOCK 21: PLACEBO TEST (SHUFFLED TARGET) - Spec 4\n")
cat("=========================================================\n")
cat("Retraining on a randomly shuffled target - this should perform near-random.\n")

placebo_r2_spec4 <- run_placebo_test(predictors_spec4, df_full, target_var,
                                     continuous_vars, n_train, n_obs, lookback,
                                     n_repeats = 3)

cat(sprintf("\nMean placebo Test R2 (shuffled target): %.4f (SD: %.4f)\n",
            mean(placebo_r2_spec4), sd(placebo_r2_spec4)))
cat(sprintf("Actual model Test R2 (Spec 4, real target): %.4f\n", result4$r2))
cat("\nInterpretation: placebo R2 should be near 0 or negative. If it comes back\n")
cat("meaningfully positive (e.g. > 0.15-0.2), the architecture can fit pure noise\n")
cat("under your current setup (lookback, sample size, epochs), which means your\n")
cat("real R2 needs to be read with real skepticism, not taken at face value.\n")

# =============================================================================
# BLOCK 22: PARAMETER COUNT vs SAMPLE SIZE (structural red-flag check)
# =============================================================================
cat("\n=========================================================\n")
cat("  BLOCK 22: PARAMETER COUNT vs TRAINING SAMPLE SIZE\n")
cat("=========================================================\n")

param_table <- lapply(names(all_results), function(nm) {
  res <- all_results[[nm]]
  n_params <- res$model$count_params()
  n_train_seq <- n_train - lookback
  data.frame(Spec = nm, N_Params = n_params, N_Train_Sequences = n_train_seq,
             Params_per_Sample = round(n_params / n_train_seq, 2))
}) %>% bind_rows()

print(param_table)
write.csv(param_table, "lstm_param_vs_sample_size.csv", row.names = FALSE)
cat("\nAs a rough guideline, having more than ~1 parameter per training sample\n")
cat("(let alone several times more) is a classic overfitting risk in small-data\n")
cat("deep learning, regardless of what regularisation (dropout) is applied.\n")
cat("Given quarterly UK data, your N_Train_Sequences is likely well under 200,\n")
cat("so check this ratio carefully.\n")

# =============================================================================
# SUMMARY: WHAT TO LOOK AT, IN ORDER
# =============================================================================
cat("\n\n=========================================================\n")
cat("  HOW TO READ THESE DIAGNOSTICS TOGETHER\n")
cat("=========================================================\n")
cat("1. Block 22 (Params vs N): if params/sample >> 1, treat everything below\n")
cat("   with caution from the outset - the model has the CAPACITY to overfit.\n")
cat("2. Block 17+18 (Train/Test gap, Val/Train loss gap): a large gap is\n")
cat("   overfitting to the specific training window.\n")
cat("3. Block 19 (Naive benchmark): if you don't beat 'no change', any fit\n")
cat("   is noise, whether or not Blocks 17-18 look fine.\n")
cat("4. Block 21 (Placebo/shuffled test): the decisive test - if the SAME\n")
cat("   setup gets a good R2 on a target with no real relationship to the\n")
cat("   predictors, your real R2 is not trustworthy on its own.\n")
cat("5. Block 20 (Ljung-Box): separate issue - tells you about remaining\n")
cat("   signal, not about overfitting per se.\n")

# =============================================================================
# BLOCK 23: COMBINED LEARNING CURVE PLOT - ALL 4 SPECS TOGETHER
# =============================================================================
# Unlike Block 14 (which splits into Group A / Group B), this puts all four
# specifications' training and validation loss curves in a single figure
# (2x2 facet grid) for direct side-by-side comparison of convergence
# behaviour and overfitting shape (val_loss diverging while train_loss falls).

build_loss_data_all <- function(result, spec_label) {
  bind_rows(
    data.frame(Epoch = seq_along(result$history$metrics$loss),
               Loss = result$history$metrics$loss,
               Type = "Training", Spec = spec_label),
    data.frame(Epoch = seq_along(result$history$metrics$val_loss),
               Loss = result$history$metrics$val_loss,
               Type = "Validation", Spec = spec_label)
  )
}

loss_data_all <- bind_rows(
  build_loss_data_all(result1, "Spec 1: No Dummy"),
  build_loss_data_all(result2, "Spec 2: + Crisis Dummies"),
  build_loss_data_all(result3, "Spec 3: + Seasonal Dummies"),
  build_loss_data_all(result4, "Spec 4: + Crisis + Seasonal Dummies")
)

loss_data_all$Spec <- factor(
  loss_data_all$Spec,
  levels = c("Spec 1: No Dummy", "Spec 2: + Crisis Dummies",
             "Spec 3: + Seasonal Dummies", "Spec 4: + Crisis + Seasonal Dummies")
)

p_loss_all <- ggplot(loss_data_all, aes(x = Epoch, y = Loss, colour = Type)) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = c("Training" = "steelblue", "Validation" = "tomato")) +
  facet_wrap(~ Spec, ncol = 2, scales = "free_x") +
  labs(
    title = "LSTM Learning Curves: All 4 Specifications",
    subtitle = "Training vs Validation Loss by Epoch",
    x = "Epoch", y = "MSE Loss", colour = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        strip.text = element_text(face = "bold"))

print(p_loss_all)
ggsave("lstm_learning_curves_all_specs.png", p_loss_all,
       width = 11, height = 8, dpi = 150)

cat("\nSaved: lstm_learning_curves_all_specs.png\n")

