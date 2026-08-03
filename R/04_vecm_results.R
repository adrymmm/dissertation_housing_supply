library(urca)
library(vars)

jo_eng <- readRDS("R/models/jo_eng.rds")

# Long-run elasticities, normalised on lhstarts
cj       <- cajorls(jo_eng, r = 1)
beta_eng <- cj$beta
elas     <- -beta_eng / beta_eng["lhstarts.l1", ]
elas["lhstarts.l1", ] <- 1
elas

# Short-run ECM
summary(cj$rlm)

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