library(urca)
eng_tf <- readRDS("R/models/eng_tf.rds")

pmax_eng <- floor(12 * (nrow(eng_tf) / 100)^(1/4))

# Helper that extracts adf lags to use in ZA
adf_lag <- function(adf_obj) {
  # number of lagged differences in the ADF regression
  rn <- rownames(adf_obj@testreg$coefficients)
  sum(grepl("^z\\.diff\\.lag", rn))}

# Same for England
for (v in colnames(eng_tf)) {
  type_adf  <- if (v == "r3") "drift"     else "trend"
  type_kpss <- if (v == "r3") "mu"        else "tau"
  type_za   <- if (v == "r3") "intercept" else "both"
    
  cat("\n---", v, "(levels) ---\n")
  
  adf_lev <- ur.df(eng_tf[, v], type = type_adf, lags = pmax_eng, selectlags = "AIC")
  cat("\n---ADF", v, "---\n"); print(summary(adf_lev))
  
  cat("\n---KPSS short", v, "---\n")
  print(summary(ur.kpss(eng_tf[, v], type = type_kpss, lags = "short")))
  cat("\n---KPSS long", v, "---\n")
  print(summary(ur.kpss(eng_tf[, v], type = type_kpss, lags = "long")))
  
  cat("\n--- ZA", v, "(lag =", adf_lag(adf_lev), ") ---\n")
  print(summary(ur.za(eng_tf[, v], model = type_za, lag = adf_lag(adf_lev))))
  
  cat("\n---", v, "(diff) ---\n")
  adf_d <- ur.df(diff(eng_tf[, v]), type = "drift", lags = pmax_eng, selectlags = "AIC")
  cat("\n---ADF", v, "---\n"); print(summary(adf_d))
  cat("\n---KPSS short", v, "---\n")
  print(summary(ur.kpss(diff(eng_tf[, v]), type = "mu", lags = "short")))
  cat("\n---KPSS long", v, "---\n")
  print(summary(ur.kpss(diff(eng_tf[, v]), type = "mu", lags = "long")))
}

# -- TWO-BREAK LM TEST --
source("../vendor/LeeStrazicichUnitRoot/LeeStrazicichUnitRootTestParallelization.R")  # defines ur.ls.bootstrap

library(foreach); library(doSNOW); library(parallel)
cl <- makeCluster(max(1, detectCores() - 1))   # all cores but one
registerDoSNOW(cl)

trend_vars <- setdiff(colnames(eng_tf), "r3")

for (v in trend_vars) {
  cat("\n---Two-Break (Model C, trend) for", v, "---\n")
  ur.ls.bootstrap(y = as.vector(eng_tf[, v]),
                  model = "break", breaks = 2, lags = pmax_eng,
                  method = "GTOS", critval = "theoretical",
                  pn = 0.1, print.results = "print")
}

cat("\n---Two-Break (Model A, level) for rate ---\n")
ur.ls.bootstrap(y = as.vector(eng_tf[, "r3"]),
                model = "crash", breaks = 2, lags = pmax_eng,
                method = "GTOS", critval = "theoretical",
                pn = 0.1, print.results = "print")

stopCluster(cl)

# Lee-Strazicich two-break LM test (Michalis runs neither ZA nor LS; robustness extension):
#   Model C (intercept + trend break) for logs; Model A (intercept break only) for r3.
#   Lags by general-to-specific from Schwert pmax = 14. Break positions on full 204-obs grid (pos 1 = 1975Q1).
#   - lhstarts: tau = -5.71, rejects UR at 10% (misses 5% by 0.02; crit -5.73); breaks 2008Q1 (GFC), 2016Q2
#              NB: FLIPS vs fixed-14 run (-4.67, could not reject). Now all four tests (ADF -4.07@1%,
#                  KPSS fails to reject stationarity, ZA -5.51@5%, LS -5.71@10%) lean (trend-)stationary.
#                  I(1) classification rests on Michalis precedent only, not on own evidence.
#   - lrprc:    tau = -4.90, cannot reject UR; breaks 2001Q2, 2012Q1                    [12 lags]
#   - lvol:     tau = -4.42, cannot reject UR; breaks 2008Q1 (GFC), 2013Q4 (Help-to-Buy) [13 lags]
#   - r3:       tau = -2.75 (Model A), cannot reject UR; breaks 1984Q4, 1987Q1          [1 lag]
#              NB: LM-minimising breaks cluster in mid-80s rate volatility, NOT GFC/ZLB/2022 splice.
#                  Do not interpret these as policy dates.
#   - lstock:   tau = -5.56, rejects UR at 10% (misses 5%; crit -5.74); breaks 1991Q3, 2010Q1  [2 lags]
#              NB: rejection survives correct lag spec (-5.79@5% under fixed-14) -> robust, not a low-power
#                  artefact. Near-trend-stationary; weak stochastic content (Denton-interpolated series).
#   - lrcc:     tau = -4.73, cannot reject UR; breaks 2000Q4, 2020Q1 (COVID)           [4 lags]
#
# Synthesis: lrprc, lvol, r3, lrcc are clean I(1) — fail to reject UR across ADF + KPSS + ZA + LS.
# lhstarts and lstock are NOT clean I(1): break-robust tests (ZA and/or LS) reject the unit root.
# Both treated as I(1) for VECM comparability with Michalis (rank r=1), but they are the documented
# mechanical source of rank > 1 in an unrestricted Johansen system — each contributes a spurious
# cointegrating direction by loading ~1 on itself.