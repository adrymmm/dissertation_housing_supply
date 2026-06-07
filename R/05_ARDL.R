library(ARDL)
library(strucchange)

# Recreating ts object
eng_ts <- ts(eng_tf, start = c(1975, 1), frequency = 4)

# Auto-select lag orders with AIC
mod <- auto_ardl(lhstarts ~ lrprc + lvol + r3 + lstock + lrcc,
                 data = eng_ts, max_order = 4)

# Top lag structures
mod$top_orders
ardl_best <- mod$best_model
summary(ardl_best)

# Bounds test for cointegration (Pesaran-Shin-Smith F-bounds)
bounds_f_test(ardl_best, case = 3)
bounds_t_test(ardl_best, case = 3)
# Reject null of no cointegration

# Reparametrize to the conditional ECM
ardl_ecm <- recm(ardl_best, case = 3) 

# Short-run ecm, speed of adjustment
summary(ardl_ecm)

# Long-run elasticities
multipliers(ardl_best)                

d <- ardl_best$model
names(d) <- make.names(names(d))                 # "L(lhstarts, 1)" -> "L.lhstarts..1."
f <- reformulate(names(d)[-1], response = names(d)[1])

# CUSUM
cusum <- efp(f, data = d, type = "Rec-CUSUM")
plot(cusum, main = "ARDL CUSUM"); sctest(cusum)

# CUSUMSQ (recursive residuals -> cumulative squared)
n <- length(rr); i <- 1:n
s <- cumsum(rr^2)/sum(rr^2)
c0 <- 0.10                      # 5% bound for n~190 — read exact value from a
# CUSUMSQ table (Durbin 1969) or Edgerton-Wells (1994)
plot(i/n, s, type="l", xlab="t/n", ylab="CUSUMSQ", main="ARDL CUSUMSQ")
abline(0, 1, lty=2)             # expected line under stability
lines(i/n, i/n + c0, col="red")
lines(i/n, i/n - c0, col="red")
