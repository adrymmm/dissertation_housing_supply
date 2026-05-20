library(urca)
library(vars)

var_eng_k5 <- vec2var(jo_eng_k5, r = 1) 

# --Estimating VECM--
# Serial Correlation test with K=5, r=1
serial.test(var_eng_k5, lags.pt = 16, type = "PT.adjusted")

# Extracting cointegrating vector
cajorls(jo_eng_k5, r = 1)$beta  

# Extracting ECM short run equation
summary(cajorls(jo_eng_k5, r = 1)$rlm)

#Normalising cointegrating vector on lhstarts
beta_eng <- cajorls(jo_eng_k5, r = 1)$beta
beta_eng / beta_eng["lhstarts.l1", ]   # normalise so lhstarts coef = 1, read elasticities directly

# --Residual Diagnostics--
# Diagnostics on K=5
normality.test(var_eng_k5, multivariate.only = FALSE) # Reject normality for all - BAD

arch.test(var_eng_k5, lags.multi = 5, multivariate.only = FALSE)
# ARCH effects detected in lvol, r3, lstock, multivariate - BAD

# --Stability--
# Testing roots
var_lev <- VAR(eng_tf, p = 5, type = "both", season = 4)
roots(var_lev)
# All roots lie in unit circle- GOOD

# Testing CUSUM on lhstarts
res <- residuals(var_eng_k5)[, "resids of lhstarts"]   # match exact colname
plot(cumsum(res), type = "l", main = "CUSUM — lhstarts resids")
abline(h = 0, lty = 2)
# Evidence for NARDL, coefficients for lhstarts unstable around downturns

# --Impulse Response Functions--
irf_eng <- irf(var_eng_k5,
               impulse  = "lrprc",        # shock to real price
               response = "lhstarts",     # supply response
               n.ahead  = 20,             # 20 quarters = 5 years
               boot     = TRUE, runs = 500, ci = 0.95)
plot(irf_eng)

# Cumulative response = long-run supply effect of a price shock
irf_cum <- irf(var_eng_k5, impulse = "lrprc", response = "lhstarts",
               n.ahead = 20, cumulative = TRUE, boot = TRUE, runs = 500)
plot(irf_cum)

# --Weak exogeneity--
# alpha restriction: zero rows = weakly exogenous to long-run relation
# free: lhstarts, lvol, lstock, lrcc | restricted: lrprc, r3
A_we <- matrix(c(1,0,0,0,
                 0,0,0,0,
                 0,1,0,0,
                 0,0,0,0,
                 0,0,1,0,
                 0,0,0,1), nrow = 6, byrow = TRUE)
summary(alrtest(jo_eng_k5, A = A_we, r = 1))
# CANNOT REJECT -> GOOD! LPRC and R3 WEAKLY EXOGENOUS MEANING R=1 IS DEFENSIBLE