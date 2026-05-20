library(urca)

pmax_uk  <- floor(12 * (nrow(uk_tf)  / 100)^(1/4))
pmax_eng <- floor(12 * (nrow(eng_tf) / 100)^(1/4))

for (v in colnames(uk_tf)) {
  type_adf  <- if (v == "r3") "drift" else "trend"
  type_kpss <- if (v == "r3") "mu"    else "tau"
  
  cat("\n---", v, "(levels) ---\n")
  cat("\n---ADF", v, "---\n")
  print(summary(ur.df(uk_tf[, v], type = type_adf, lags = pmax_uk, selectlags = "AIC")))
  cat("\n---KPSS short", v, "---\n")
  print(summary(ur.kpss(uk_tf[, v], type = type_kpss, lags = "short")))
  cat("\n---KPSS long", v, "---\n")
  print(summary(ur.kpss(uk_tf[, v], type = type_kpss, lags = "long")))
  cat("\n--- ZA", v,"---\n")
  print(summary(ur.za(uk_tf[, v], model = "both", lag = pmax_uk)))
  
  cat("\n---", v, "(diff) ---\n")
  cat("\n---ADF", v, "---\n")
  print(summary(ur.df(diff(uk_tf[, v]), type = "drift", lags = pmax_uk, selectlags = "AIC")))
  cat("\n---KPSS short", v, "---\n")
  print(summary(ur.kpss(diff(uk_tf[, v]), type = "mu", lags = "short")))
  cat("\n---KPSS long", v, "---\n")
  print(summary(ur.kpss(diff(uk_tf[, v]), type = "mu", lags = "long")))
}

# Sig structural breaks: r3 (1992 Q3) ERM exit, lhstarts (2007 Q4) GFC Housing Crash

# Same for England
for (v in colnames(eng_tf)) {
  type_adf  <- if (v == "r3") "drift" else "trend"
  type_kpss <- if (v == "r3") "mu"    else "tau"
  
  cat("\n---", v, "(levels) ---\n")
  cat("\n---ADF", v, "---\n")
  print(summary(ur.df(eng_tf[, v], type = type_adf, lags = pmax_eng, selectlags = "AIC")))
  cat("\n---KPSS short", v, "---\n")
  print(summary(ur.kpss(eng_tf[, v], type = type_kpss, lags = "short")))
  cat("\n---KPSS long", v, "---\n")
  print(summary(ur.kpss(eng_tf[, v], type = type_kpss, lags = "long")))
  cat("\n--- ZA", v,"---\n")
  print(summary(ur.za(eng_tf[, v], model = "both", lag = pmax_eng)))
  
  cat("\n---", v, "(diff) ---\n")
  cat("\n---ADF", v, "---\n")
  print(summary(ur.df(diff(eng_tf[, v]), type = "drift", lags = pmax_eng, selectlags = "AIC")))
  cat("\n---KPSS short", v, "---\n")
  print(summary(ur.kpss(diff(eng_tf[, v]), type = "mu", lags = "short")))
  cat("\n---KPSS long", v, "---\n")
  print(summary(ur.kpss(diff(eng_tf[, v]), type = "mu", lags = "long")))
}

# Sig structural breaks: r3 (1992 Q3) ERM exit, lhstarts (2007 Q4) GFC Housing Crash