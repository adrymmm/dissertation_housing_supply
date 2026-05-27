library(urca)

pmax_uk  <- floor(12 * (nrow(uk_tf)  / 100)^(1/4))
pmax_eng <- floor(12 * (nrow(eng_tf) / 100)^(1/4))

# Helper that extracts adf lags to use in ZA
adf_lag <- function(adf_obj) {
  # number of lagged differences in the ADF regression
  rn <- rownames(adf_obj@testreg$coefficients)
  sum(grepl("^z\\.diff\\.lag", rn))
}

for (v in colnames(uk_tf)) {
  type_adf  <- if (v == "r3") "drift"     else "trend"
  type_kpss <- if (v == "r3") "mu"        else "tau"
  type_za   <- if (v == "r3") "intercept" else "both"
  
  cat("\n---", v, "(levels) ---\n")
  
  adf_lev <- ur.df(uk_tf[, v], type = type_adf, lags = pmax_uk, selectlags = "AIC")
  cat("\n---ADF", v, "---\n"); print(summary(adf_lev))
  
  cat("\n---KPSS short", v, "---\n")
  print(summary(ur.kpss(uk_tf[, v], type = type_kpss, lags = "short")))
  cat("\n---KPSS long", v, "---\n")
  print(summary(ur.kpss(uk_tf[, v], type = type_kpss, lags = "long")))
  
  cat("\n--- ZA", v, "(lag =", adf_lag(adf_lev), ") ---\n")
  print(summary(ur.za(uk_tf[, v], model = type_za, lag = adf_lag(adf_lev))))
  
  cat("\n---", v, "(diff) ---\n")
  adf_d <- ur.df(diff(uk_tf[, v]), type = "drift", lags = pmax_uk, selectlags = "AIC")
  cat("\n---ADF", v, "---\n"); print(summary(adf_d))
  cat("\n---KPSS short", v, "---\n")
  print(summary(ur.kpss(diff(uk_tf[, v]), type = "mu", lags = "short")))
  cat("\n---KPSS long", v, "---\n")
  print(summary(ur.kpss(diff(uk_tf[, v]), type = "mu", lags = "long")))
}

# Same for England
for (v in colnames(eng_tf)) {
  type_adf  <- if (v == "r3") "drift"     else "trend"
  type_kpss <- if (v == "r3") "mu"        else "tau"
  type_za   <- if (v == "r3") "intercept" else "both"
  
  cat("\n---", v, "(levels) ---\n")
  
  adf_lev <- ur.df(eng_tf[, v], type = type_adf, lags = pmax_uk, selectlags = "AIC")
  cat("\n---ADF", v, "---\n"); print(summary(adf_lev))
  
  cat("\n---KPSS short", v, "---\n")
  print(summary(ur.kpss(eng_tf[, v], type = type_kpss, lags = "short")))
  cat("\n---KPSS long", v, "---\n")
  print(summary(ur.kpss(eng_tf[, v], type = type_kpss, lags = "long")))
  
  cat("\n--- ZA", v, "(lag =", adf_lag(adf_lev), ") ---\n")
  print(summary(ur.za(eng_tf[, v], model = type_za, lag = adf_lag(adf_lev))))
  
  cat("\n---", v, "(diff) ---\n")
  adf_d <- ur.df(diff(eng_tf[, v]), type = "drift", lags = pmax_uk, selectlags = "AIC")
  cat("\n---ADF", v, "---\n"); print(summary(adf_d))
  cat("\n---KPSS short", v, "---\n")
  print(summary(ur.kpss(diff(eng_tf[, v]), type = "mu", lags = "short")))
  cat("\n---KPSS long", v, "---\n")
  print(summary(ur.kpss(diff(eng_tf[, v]), type = "mu", lags = "long")))
}


# ZA single-break test (Michalis does not run this; robustness extension):
#   - lhstarts: rejects UR at 5%, break at 2007 Q4 (GFC)
#   - r3: identifies 1992 Q3 break (ERM exit) but does not reject UR at 5%
#   - Other variables: cannot reject UR
# Results essentially identical for UK and England series.
# Combined with ADF + KPSS evidence, all six treated as I(1) for VECM analysis.