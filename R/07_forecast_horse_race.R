# ============================================================
# FORECAST HORSE RACE: Naive vs VECM vs ARDL
# England, pseudo out-of-sample from 2020 Q1
# ============================================================

library(urca); library(vars); library(ARDL); library(dynlm); library(forecast)

# ---- 1. Split ----
cutoff      <- c(2019, 4)
test_end    <- c(2023, 4)   # England lstock real through 2023 Q4

eng_train   <- window(eng_tf, end = cutoff)
eng_test    <- window(eng_tf, start = c(2020, 1), end = test_end)
h           <- nrow(eng_test)
actual      <- as.numeric(eng_test[, "lhstarts"])

cat("Training obs:", nrow(eng_train), "| Test obs:", h, "\n")

# ---- 2. Naive — random walk ----
fc_naive <- rep(tail(as.numeric(eng_train[, "lhstarts"]), 1), h)

# ---- 3. VECM — unconditional ----
jo_tr <- ca.jo(eng_train, type = "eigen", ecdet = "const", K = 2, season = 4)
var_tr <- vec2var(jo_tr, r = 1)

fc_vecm <- predict(var_tr, n.ahead = h)$fcst$lhstarts[, "fcst"]
# ---- 4. ARDL — conditional on actual future regressors ----
eng_df_tr   <- to_ardl_df(eng_train)
ardl_tr_fit <- auto_ardl(
  lhstarts ~ lrprc + lvol + r3 + lstock + lrcc | d1 + d2 + d3,
  data = eng_df_tr, max_order = 8, selection = "AIC"
)
cat("Training ARDL order:", ardl_tr_fit$best_order, "\n")

# Extract UECM coefficients
b   <- coef(uecm(ardl_tr_fit$best_model))
ord <- ardl_tr_fit$best_order   # c(p, q_lrprc, q_lvol, q_r3, q_lstock, q_lrcc)
p   <- ord[1]
vars_order <- c("lrprc", "lvol", "r3", "lstock", "lrcc")

# Full data matrix — actual values used for regressors in forecast
full_mat  <- as.matrix(eng_tf)
seas_full <- make_seas_dummies(eng_tf)
n_tr      <- nrow(eng_train)

# lhstarts_ext: actual in training, will fill with forecasts in test
lhstarts_ext <- as.numeric(full_mat[, "lhstarts"])

for (i in 1:h) {
  t <- n_tr + i
  
  # ECT — level terms at t-1 (actual regressors, forecast lhstarts)
  ect <- b["(Intercept)"] +
    b["L(lhstarts, 1)"] * lhstarts_ext[t-1]
  for (k in seq_along(vars_order)) {
    v  <- vars_order[k]
    qk <- ord[k + 1]
    nm  <- paste0("L(", v, ", 1)")
    val <- full_mat[t - 1, v]
    if (nm %in% names(b)) ect <- ect + b[nm] * val
  
  # Short-run lhstarts lags (uses forecasted values for t-j)
  sr_lhs <- 0
  for (j in 1:(p - 1)) {
    nm <- paste0("d(L(lhstarts, ", j, "))")
    if (nm %in% names(b))
      sr_lhs <- sr_lhs + b[nm] * (lhstarts_ext[t-j] - lhstarts_ext[t-j-1])
  }
  
  # Short-run regressor lags (actual values throughout)
  sr_x <- 0
  for (k in seq_along(vars_order)) {
    v  <- vars_order[k]
    qk <- ord[k + 1]
    if (qk >= 1) {
      nm0 <- paste0("d(", v, ")")
      if (nm0 %in% names(b))
        sr_x <- sr_x + b[nm0] * (full_mat[t, v] - full_mat[t-1, v])
    }
    if (qk >= 2) {
      for (j in 1:(qk - 1)) {
        nmj <- paste0("d(L(", v, ", ", j, "))")
        if (nmj %in% names(b))
          sr_x <- sr_x + b[nmj] * (full_mat[t-j, v] - full_mat[t-j-1, v])
      }
    }
  }
  
  # Seasonal
  s_t <- b["d1"] * seas_full[t, "d1"] +
    b["d2"] * seas_full[t, "d2"] +
    b["d3"] * seas_full[t, "d3"]
  
  # Level forecast: lhstarts_t = lhstarts_{t-1} + predicted delta
  lhstarts_ext[t] <- lhstarts_ext[t-1] + ect + sr_lhs + sr_x + s_t
}

fc_ardl <- lhstarts_ext[(n_tr + 1):(n_tr + h)]

# ---- 5. Accuracy metrics ----
rmse <- function(a, f) sqrt(mean((a - f)^2))
mae  <- function(a, f) mean(abs(a - f))

results <- data.frame(
  Model = c("Naive RW", "VECM", "ARDL (conditional)"),
  RMSE  = round(c(rmse(actual, fc_naive),
                  rmse(actual, fc_vecm),
                  rmse(actual, fc_ardl)), 4),
  MAE   = round(c(mae(actual, fc_naive),
                  mae(actual, fc_vecm),
                  mae(actual, fc_ardl)), 4)
)
cat("\n=== Forecast accuracy (log starts) ===\n")
print(results)

dm.test(
  e1 = actual - fc_vecm,
  e2 = actual - fc_ardl,
  alternative = "greater",   # H1: VECM errors > ARDL errors
  h = 1,
  power = 2
)

# ---- 6. Plot ----
dates_num  <- as.numeric(time(eng_test))
ylim_range <- range(c(actual, fc_naive, fc_vecm, fc_ardl), na.rm = TRUE)

plot(dates_num, actual, type = "l", lwd = 2,
     ylim = ylim_range + c(-0.05, 0.05),
     main = "England lhstarts: Pseudo OOS Forecast 2020 Q1–2023 Q4",
     ylab = "log private starts", xlab = "")
lines(dates_num, fc_naive, col = "grey50",    lty = 2, lwd = 1.5)
lines(dates_num, fc_vecm,  col = "firebrick", lty = 1, lwd = 1.5)
lines(dates_num, fc_ardl,  col = "steelblue", lty = 1, lwd = 1.5)
legend("topright", bty = "n",
       legend = c("Actual", "Naive RW", "VECM", "ARDL (cond.)"),
       col    = c("black", "grey50", "firebrick", "steelblue"),
       lty    = c(1, 2, 1, 1), lwd = 1.5)
