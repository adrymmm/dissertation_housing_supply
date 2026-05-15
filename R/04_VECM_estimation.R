library(vars)
library(urca)

# Estimate VECM with r=1, K=2, seasonal dummies
jo_uk  <- ca.jo(uk_tf,  type = "eigen", ecdet = "const",
                K = 2, dumvar = make_seas_dummies(uk_tf))
jo_eng <- ca.jo(eng_tf, type = "eigen", ecdet = "const",
                K = 2, dumvar = make_seas_dummies(eng_tf))

vecm_uk  <- cajorls(jo_uk,  r = 1)
vecm_eng <- cajorls(jo_eng, r = 1)

# Cointegrating vectors (normalised on lhstarts)
cat("=== UK cointegrating vector ===\n")
print(vecm_uk$beta)

cat("\n=== England cointegrating vector ===\n")
print(vecm_eng$beta)

# lhstarts equation only — the one we care about
cat("\n=== UK lhstarts equation ===\n")
print(summary(vecm_uk$rlm)$"Response lhstarts.d")

cat("\n=== England lhstarts equation ===\n")
print(summary(vecm_eng$rlm)$"Response lhstarts.d")

# Diagnostics
# Convert ca.jo objects to VAR representation for diagnostic tools
var_uk  <- vec2var(jo_uk,  r = 1)
var_eng <- vec2var(jo_eng, r = 1)

# Serial correlation — Portmanteau test
cat("=== Serial correlation ===\n")
print(serial.test(var_uk,  lags.pt = 16, type = "PT.asymptotic"))
print(serial.test(var_eng, lags.pt = 16, type = "PT.asymptotic"))

# Normality — JB test on residuals
cat("\n=== Normality ===\n")
print(normality.test(var_uk,  multivariate.only = FALSE))
print(normality.test(var_eng, multivariate.only = FALSE))

# Stability — CUSUM on lhstarts residuals directly
cat("\n=== Structural stability (lhstarts residuals) ===\n")
resid_uk  <- residuals(var_uk)[,  "resids of lhstarts"]
resid_eng <- residuals(var_eng)[, "resids of lhstarts"]

par(mfrow = c(1, 2))
plot(cumsum(resid_uk),  type = "l", main = "CUSUM - UK lhstarts",
     ylab = "", xlab = "")
abline(h = 0, lty = 2)
plot(cumsum(resid_eng), type = "l", main = "CUSUM - England lhstarts",
     ylab = "", xlab = "")
abline(h = 0, lty = 2)
par(mfrow = c(1, 1))

# Evidence for NARDL
