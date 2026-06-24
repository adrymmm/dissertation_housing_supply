library(urca)
library(vars)
library(car)

eng_tf       <- readRDS("R/models/eng_tf.rds")
jo_eng       <- readRDS("R/models/jo_eng.rds")
final_dummies <- readRDS("R/models/final_dummies.rds")

eng_ts <- ts(eng_tf, start = c(1975, 1), frequency = 4)

var_eng <- vec2var(jo_eng, r = 1)
serial.test(var_eng, lags.pt = 16, type = "PT.adjusted")

# Cointegrating vector
cj       <- cajorls(jo_eng, r = 1)
beta_eng <- cj$beta
elas     <- -beta_eng / beta_eng["lhstarts.l1", ]
elas["lhstarts.l1", ] <- 1
elas

# Short-run ECM
summary(cj$rlm)

# Residual diagnostics
normality.test(var_eng, multivariate.only = FALSE)
arch.test(var_eng, lags.multi = 5, multivariate.only = FALSE)

# Stability
var_lev <- VAR(eng_tf, p = 5, type = "const", season = 4, exogen = final_dummies)
roots(var_lev)

# CUSUM on lhstarts
res <- residuals(var_eng)[, "resids of lhstarts"]
plot(cumsum(res), type = "l", main = "CUSUM — lhstarts resids")
abline(h = 0, lty = 2)

saveRDS(var_eng, "R/models/var_eng.rds")

# Evidence for NARDL, coefficients for lhstarts unstable around downturns

# UNCOMMENT TO USE THEY TAKE TIME TO RUN

# # Impulse Response Functions
# eng_tf2 <- eng_tf[, c("lrprc","lvol","r3","lstock","lrcc","lhstarts")]
# jo2  <- ca.jo(eng_tf2, type="trace", ecdet="none", K=5, spec="transitory",
#               season=4, dumvar=final_dummies)
# 
# v2   <- vec2var(jo2, r = 1)
# irf2 <- irf(v2,
#             impulse="lrprc",
#             response="lhstarts", # shock to real price, supply response
#             n.ahead=20, # 5 years
#             ortho=TRUE, # ortho
#             boot=TRUE,
#             runs=1000) #
# plot(irf2)
# 
# # Cumulative response = long-run supply effect of a price shock
# irf_cum <- irf(v2,
#                impulse = "lrprc",
#                response = "lhstarts",
#                n.ahead = 20,
#                cumulative = TRUE,
#                boot = TRUE,
#                runs = 500)
# plot(irf_cum)
#
# # FEVD 
# fevd_eng <- fevd(v2, n.ahead = 20)
# fevd_eng$lhstarts        # the row that matters: variance decomp OF starts
# plot(fevd_eng)           