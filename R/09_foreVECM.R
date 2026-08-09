library(vars)
library(zoo)

var_eng       <- readRDS("R/models/var_eng.rds")
final_dummies <- readRDS("R/models/final_dummies.rds")
eng_tf        <- readRDS("R/models/eng_tf.rds")
eng_ts        <- ts(eng_tf, start = c(1975, 1), frequency = 4)

# Horizon and dating derived, not hardcoded 
obr <- read.csv("data/python_master/OBR/obr_scenario.csv")
obr <- obr[obr$period > "2025Q4", ]
h <- nrow(obr)

last_obs <- as.yearqtr(tail(time(eng_ts), 1))
fc_start <- last_obs + 1 / 4
periods  <- format(fc_start + (0:(h - 1)) / 4, "%YQ%q")

cat("Sample ends", format(last_obs, "%YQ%q"),
    "| forecasting", h, "quarters to", periods[h], "\n")
# Alignment guard
stopifnot(identical(periods, obr$period))

# Forecast
dv <- matrix(0, h, ncol(final_dummies),
             dimnames = list(NULL, colnames(final_dummies)))

fc <- predict(var_eng, n.ahead = h, dumvar = dv)

lh <- fc$fcst$lhstarts

out <- data.frame(
  period      = periods,
  vecm_log    = lh[, "fcst"],
  vecm_starts = exp(lh[, "fcst"]),
  vecm_lower  = exp(lh[, "lower"]),
  vecm_upper  = exp(lh[, "upper"])
)

# Jump-off sanity check
last_actual <- exp(tail(eng_tf[, "lhstarts"], 1))
cat(sprintf("Jump-off: last actual %.0f -> first forecast %.0f (%.1f%%)\n",
            last_actual, out$vecm_starts[1],
            100 * (out$vecm_starts[1] / last_actual - 1)))

# Implied covariate paths 
cov_paths <- data.frame(period = periods)
for (v in setdiff(colnames(eng_tf), "lhstarts")) {
  cov_paths[[v]] <- fc$fcst[[v]][, "fcst"]
}

obr_lrprc <- obr$lrprc
cat(sprintf("lrprc over horizon: VECM %+.3f log-points, OBR %+.3f\n",
            tail(cov_paths$lrprc, 1) - cov_paths$lrprc[1],
            tail(obr_lrprc, 1) - obr_lrprc[1]))

dir.create("data/outputs/forecasts", recursive = TRUE, showWarnings = FALSE)
write.csv(out, "data/outputs/forecasts/vecm_unconditional_forecast.csv",
          row.names = FALSE)
write.csv(cov_paths, "data/outputs/forecasts/vecm_implied_covariates.csv",
          row.names = FALSE)

print(out)