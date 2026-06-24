library(urca)
library(vars)
library(car)

eng_tf <- readRDS("R/models/eng_tf.rds")

eng_ts    <- ts(eng_tf, start = c(1975, 1), frequency = 4)
time_axis <- time(eng_ts)
dd <- function(yr) which.min(abs(time_axis - yr))

# Integration order
int_order <- function(x) {
  c(ADF_lvl  = ur.df(x,        type = "trend", selectlags = "AIC")@teststat[1],
    KPSS_lvl = ur.kpss(x,      type = "tau")@teststat[1],
    ADF_diff = ur.df(diff(x),  type = "drift", selectlags = "AIC")@teststat[1])
}
int_tab <- t(sapply(colnames(eng_tf), function(v) int_order(eng_tf[, v])))
print(round(int_tab, 2))

# Zivot-Andrews (break-robust) confirms lhstarts stationary; run individually

# Seasonality check
for (v in colnames(eng_tf)) {
  m <- lm(diff(eng_ts[, v]) ~ factor(cycle(eng_ts[, v])[-1]))   # on diffs
  f <- summary(m)$fstatistic
  p <- pf(f[1], f[2], f[3], lower.tail = FALSE)
  cat(sprintf("%-9s seasonal F p = %.4f  %s\n",
              v, p, ifelse(p < 0.05, "<- seasonal", "smooth")))
}

# Impulse dummies - genuine transitory shocks only
final_dummies <- matrix(0, nrow(eng_ts), 4)
colnames(final_dummies) <- c("gfc_2008Q3", "covid_2020Q2", "covid_2020Q3", "saw_2023Q2")
final_dummies[dd(2008.50), 1] <- 1
final_dummies[dd(2020.25), 2] <- 1
final_dummies[dd(2020.50), 3] <- 1
final_dummies[dd(2023.25), 4] <- 1

# Lag order
pmax_eng <- floor(12 * (nrow(eng_tf) / 100)^(1/4))
lagsel   <- VARselect(eng_tf, lag.max = pmax_eng, type = "both")
print(lagsel$selection)          # report SC(n) = BIC 
K <- 5L                          # imposed (Michalis 2023)

# Johansen rank - 6-variable system (Case 3, season = 4)
jo_eng <- ca.jo(eng_tf, type = "trace", ecdet = "none", K = K,
                spec = "transitory", season = 4, dumvar = final_dummies)
summary(jo_eng)

jo_eng_eig <- ca.jo(eng_tf, type = "eigen", ecdet = "none", K = K,
                    spec = "transitory", season = 4, dumvar = final_dummies)
summary(jo_eng_eig)
# -> trace selects r = 2 (r<=1 rejected)

# Robustness - 5-variable system excluding lstock
eng_tf_5 <- eng_tf[, c("lhstarts", "lrprc", "lvol", "r3", "lrcc")]
jo_5 <- ca.jo(eng_tf_5, type = "trace", ecdet = "none", K = K,
              spec = "transitory", season = 4, dumvar = final_dummies)
summary(jo_5)
# -> still r = 2: lstock is not the cause; lhstarts stationarity is.

# ------------------------------------------------------------
#    Rank interpretation
#    r = 2 = supply relation + lhstarts-stationarity direction.
#    Optional formal check: test that the unit vector on lhstarts
#    spans one cointegrating direction (partial beta restriction).
#    NB verify the H/r setup for blrtest before quoting in the text.
#      H <- ...                     # unit vector e_lhstarts + free directions
#      summary(blrtest(jo_eng, H = H, r = 2))
#    Primary model proceeds at r = 1 (the economic supply relation).
# ------------------------------------------------------------
var_eng <- vec2var(jo_eng, r = 1)
var_5   <- vec2var(jo_5,   r = 1)

# Residual diagnostics
normality.test(var_eng); serial.test(var_eng, lags.pt = 16, type = "PT.adjusted")
normality.test(var_5);   serial.test(var_5,   lags.pt = 16, type = "PT.adjusted")

# Outlier locator used for dummies
res       <- residuals(var_eng)
res_dates <- tail(time(eng_ts), nrow(res))
z   <- scale(res)
idx <- which(abs(z) > 3, arr.ind = TRUE)
print(data.frame(date = round(res_dates[idx[, 1]], 2),
                 eq   = colnames(res)[idx[, 2]],
                 z    = round(z[idx], 2))[order(-abs(z[idx])), ])

#    Weak exogeneity (at r = 1)
vecm_u <- cajorls(jo_eng, r = 1)
srlm   <- summary(vecm_u$rlm)
t_lrprc <- srlm$`Response lrprc.d`$coefficients["ect1", "t value"]
t_r3    <- srlm$`Response r3.d`$coefficients["ect1", "t value"]
cat(sprintf("WE  lrprc: t = %.3f   r3: t = %.3f\n", t_lrprc, t_r3))

# r3 t-value carries the rejection in joint test

b <- coef(vecm_u$rlm)["ect1", ]          # ect1 loading in every equation
E <- residuals(vecm_u$rlm)               # residual matrix, cols = equations
X <- model.matrix(vecm_u$rlm)
g <- solve(crossprod(X))["ect1", "ect1"] # (X'X)^-1 element for ect1

eqs <- c("lrprc.d", "r3.d")
a <- b[eqs]
S <- crossprod(E[, eqs]) / (nrow(E) - ncol(X))   # 2x2 residual cov (Sigma block)
W <- as.numeric(t(a) %*% solve(g * S) %*% a)     # Sigma-aware joint Wald
cat(sprintf("Joint WE: chi2(2) = %.3f  p = %.4f\n",
            W, pchisq(W, df = 2, lower.tail = FALSE)))

saveRDS(jo_eng, "R/models/jo_eng.rds")
saveRDS(var_eng, "R/models/var_eng.rds")
saveRDS(final_dummies, "R/models/final_dummies.rds")