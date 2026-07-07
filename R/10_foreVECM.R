library(ARDL)
library(zoo)
library(vars)

var_eng       <- readRDS("R/models/var_eng.rds")
final_dummies <- readRDS("R/models/final_dummies.rds")

h  <- 13   # 2026Q1–2029Q1; set to match your net-additions horizon
dv <- matrix(0, h, ncol(final_dummies),
             dimnames = list(NULL, colnames(final_dummies)))

fc <- predict(var_eng, n.ahead = h, dumvar = dv)

# lhstarts path (fcst / lower / upper are log)
vecm_log    <- fc$fcst$lhstarts[, "fcst"]
vecm_starts <- exp(vecm_log)

out <- data.frame(
  period       = as.character(zoo::as.yearqtr("2026 Q1") + (0:(h-1))/4),
  vecm_log     = vecm_log,
  vecm_starts  = vecm_starts,
  vecm_lower   = exp(fc$fcst$lhstarts[, "lower"]),
  vecm_upper   = exp(fc$fcst$lhstarts[, "upper"])
)
write.csv(out, "data/outputs/forecasts/vecm_unconditional_forecast.csv", row.names = FALSE)