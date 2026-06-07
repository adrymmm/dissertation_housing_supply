library(urca)
library(vars)
library(car)

var_eng <- vec2var(jo_eng, r = 1) 

# --Estimating VECM--
# Serial Correlation test with K=5, r=1
serial.test(var_eng, lags.pt = 16, type = "PT.adjusted")

# Extracting cointegrating vector
cajorls(jo_eng, r = 1)$beta  
  
# Extracting ECM short run equation
summary(cajorls(jo_eng, r = 1)$rlm)

#Normalising cointegrating vector on lhstarts
beta_eng <- cajorls(jo_eng, r = 1)$beta
beta_eng / beta_eng["lhstarts.l1", ]   # normalise so lhstarts coef = 1, read elasticities directly
elas <- -beta_eng / beta_eng["lhstarts.l1", ]
elas["lhstarts.l1", ] <- 1    # keep the normalised 1
elas

# --Residual Diagnostics--
# Diagnostics on K=5
normality.test(var_eng, multivariate.only = FALSE) # Reject normality for all - BAD

arch.test(var_eng, lags.multi = 5, multivariate.only = FALSE)
# ARCH effects detected in lvol, r3, lstock, multivariate - BAD

# --Stability--
# Testing roots
var_lev <- VAR(eng_tf, p = 5, type = "both", season = 4)
roots(var_lev)
# All roots lie in unit circle- GOOD

# Testing CUSUM on lhstarts
res <- residuals(var_eng)[, "resids of lhstarts"]   # match exact colname
plot(cumsum(res), type = "l", main = "CUSUM — lhstarts resids")
abline(h = 0, lty = 2)
# Evidence for NARDL, coefficients for lhstarts unstable around downturns

# --Impulse Response Functions--
# Orthogonalizing
eng_tf2 <- eng_tf[, c("lrprc","lvol","r3","lstock","lrcc","lhstarts")]
jo2  <- ca.jo(eng_tf2, type="trace", ecdet="none", K=K_eng, spec="transitory", season=4)
v2   <- vec2var(jo2, r = 1)
irf2 <- irf(v2, impulse="lrprc", response="lhstarts", # shock to real price, supply response
            n.ahead=20, ortho=TRUE, boot=TRUE, runs=1000) # 5 years, ortho
plot(irf2)

# Cumulative response = long-run supply effect of a price shock
irf_cum <- irf(v2, impulse = "lrprc", response = "lhstarts",
               n.ahead = 20, cumulative = TRUE, boot = TRUE, runs = 500)
plot(irf_cum)

# -- FEVD --
fevd_eng <- fevd(v2, n.ahead = 20)
fevd_eng$lhstarts        # the row that matters: variance decomp OF starts
plot(fevd_eng)           