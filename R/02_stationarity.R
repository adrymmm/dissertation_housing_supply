library(tseries)
library(urca)

schwert_pmax <- function(x) floor(12 * (length(x) / 100)^(1/4))

run_adf <- function(x, type) {
  T_full <- length(x)
  pmax   <- schwert_pmax(x)
  
  best_aic <- Inf
  best_lag <- 0
  
  for (p in 0:pmax) {
    fit <- ur.df(x, type = type, lags = p)
    # Fix sample at pmax so AIC is comparable across lag lengths
    res_fixed <- tail(fit@res, T_full - pmax - 1)
    n_fixed   <- length(res_fixed)
    k         <- p + ifelse(type == "trend", 3, 2)
    aic       <- log(sum(res_fixed^2) / n_fixed) + 2 * k / n_fixed
    if (aic < best_aic) { best_aic <- aic; best_lag <- p }
  }
  
  fit   <- ur.df(x, type = type, lags = best_lag)
  tstat <- fit@teststat[1]
  list(tstat = round(tstat, 3),
       lag   = best_lag,
       obs   = T_full - best_lag - 1)
}

# Also fix the first-differences sprintf: r$lag -> r$tstat
run_adf_table <- function(tf, label) {
  vars  <- colnames(tf)
  types <- ifelse(vars == "r3", "drift", "trend")
  cat("\n---", label, "--- levels ---\n")
  cat(sprintf("%-12s %8s %6s %6s\n", "variable", "t-stat", "lags", "obs"))
  for (i in seq_along(vars)) {
    r <- run_adf(tf[, i], types[i])
    cat(sprintf("%-12s %8.3f %6d %6d\n", vars[i], r$tstat, r$lag, r$obs))
  }
  cat("\n---", label, "--- first differences ---\n")
  cat(sprintf("%-12s %8s %6s %6s\n", "variable", "t-stat", "lags", "obs"))
  for (i in seq_along(vars)) {
    r <- run_adf(diff(tf[, i]), "drift")
    cat(sprintf("%-12s %8.3f %6d %6d\n", vars[i], r$tstat, r$lag, r$obs))  # fixed
  }
}

run_adf_table(uk_tf,  "UK")
run_adf_table(eng_tf, "England")

kpss.test(uk_tf[, "lhstarts"],  null = "Trend")
kpss.test(eng_tf[, "lhstarts"], null = "Trend")
kpss.test(diff(eng_tf[, "lstock"]), null = "Trend")
