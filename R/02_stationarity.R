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

# Testing manually for seasonality ADF
adf_seas <- function(x, p = 4, trend = TRUE) {
  x <- as.vector(x); n <- length(x); dx <- diff(x)
  E <- embed(dx, p + 1)
  d <- data.frame(dy = E[, 1],
                  L1 = x[(p + 1):(n - 1)],
                  tt = (p + 2):n,
                  q  = factor(rep_len(1:4, n)[(p + 2):n]))   # assumes series starts in Q1
  for (i in 1:p) d[[paste0("dl", i)]] <- E[, i + 1]
  f <- as.formula(paste("dy ~ L1 +", if (trend) "tt +" else "", "q +",
                        paste0("dl", 1:p, collapse = " + ")))
  coef(summary(lm(f, data = d)))["L1", ]
}

adf_seas(eng_tf[, "lhstarts"], p = 12)   # compare t to tau3: -3.99 / -3.43 / -3.13