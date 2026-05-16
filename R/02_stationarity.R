library(urca)

pmax_uk  <- floor(12 * (nrow(uk_tf)  / 100)^(1/4))
pmax_eng <- floor(12 * (nrow(eng_tf) / 100)^(1/4))

for (v in colnames(uk_tf)) {
  type_adf  <- if (v == "r3") "drift" else "trend"
  type_kpss <- if (v == "r3") "mu"    else "tau"
  
  cat("\n---", v, "(levels) ---\n")
  print(summary(ur.df(uk_tf[, v], type = type_adf, lags = pmax_uk, selectlags = "AIC")))
  print(summary(ur.kpss(uk_tf[, v], type = type_kpss, lags = "short")))
  
  cat("\n---", v, "(diff) ---\n")
  print(summary(ur.df(diff(uk_tf[, v]), type = "drift", lags = pmax_uk, selectlags = "AIC")))
  print(summary(ur.kpss(diff(uk_tf[, v]), type = "mu", lags = "short")))
}

# Same for England
for (v in colnames(eng_tf)) {
  type_adf  <- if (v == "r3") "drift" else "trend"
  type_kpss <- if (v == "r3") "mu"    else "tau"
  
  cat("\n---", v, "(levels) ---\n")
  print(summary(ur.df(eng_tf[, v], type = type_adf, lags = pmax_eng, selectlags = "AIC")))
  print(summary(ur.kpss(eng_tf[, v], type = type_kpss, lags = "short")))
  
  cat("\n---", v, "(diff) ---\n")
  print(summary(ur.df(diff(eng_tf[, v]), type = "drift", lags = pmax_eng, selectlags = "AIC")))
  print(summary(ur.kpss(diff(eng_tf[, v]), type = "mu", lags = "short")))
}