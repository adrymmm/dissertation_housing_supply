library(urca)

eng_tf <- df %>%
  select(lhstarts, lrprc, lvol, r3, lstock, lrcc) %>%
  as.matrix()

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

res <- ur.ls.bootstrap(y = as.vector(eng_tf[,"lhstarts"]),
                       model = "break", breaks = 2,
                       lags = pmax_eng, method = "Fixed",
                       critval = "theoretical",         
                       pn = 0.1, print.results = "print")

stopCluster(cl)

# ZA single-break test (Michalis does not run this; robustness extension):
#   - lhstarts: tau = -5.51, rejects UR at 5%, break at pos 132 (2008Q4, GFC)
#   - lrprc:    tau = -4.07, cannot reject UR; break at pos 105 (2002Q1)
#   - lvol:     tau = -3.78, cannot reject UR; break at pos 132 (2008Q4)
#   - r3:       tau = -4.50, rejects at 10% only; break at pos 188 (2022Q1, post-ZLB)
#              NB: UK r3 break at 1992Q3 (ERM); divergence likely Bank Rate vs LIBOR splice
#   - lstock:   tau = -5.25, rejects at 5%; break at pos 69 (1992Q3)
#              NB: inter-censal stock revision artefact; outside estimation window
#   - lrcc:     tau = -4.12, cannot reject UR; break at pos 103 (2001Q3)
# Combined with ADF + KPSS evidence, all six treated as I(1) for VECM analysis.