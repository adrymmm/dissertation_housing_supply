library(ARDL)
library(strucchange)

eng_tf <- readRDS("R/models/eng_tf.rds")

eng_ts <- ts(eng_tf, start = c(1975, 1), frequency = 4)

# Creating impulse dummies
ta <- time(eng_ts); dd <- function(yr) which.min(abs(ta - yr))
D <- matrix(0, nrow(eng_ts), 4,
            dimnames = list(NULL, c("d08Q3","d20Q2","d20Q3","d23Q2")))
D[dd(2008.50),1] <- 1
D[dd(2020.25),2] <- 1
D[dd(2020.50),3] <- 1
D[dd(2023.25),4] <- 1
eng_ts <- ts(cbind(as.matrix(eng_tf), D), start = c(1975,1), frequency = 4)

mod <- auto_ardl(lhstarts ~ lrprc + lvol + r3 + lstock + lrcc |
                   d08Q3 + d20Q2 + d20Q3 + d23Q2,
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
names(d) <- make.names(names(d))
dum  <- c("d08Q3","d20Q2","d20Q3","d23Q2")
core <- setdiff(names(d), dum)

# FWL - partialling out dummies
Dm    <- cbind(1, as.matrix(d[, dum]))
purge <- function(z) lm.fit(Dm, z)$residuals
dp    <- as.data.frame(lapply(d[, core], purge)) 
names(dp) <- core

f     <- reformulate(names(dp)[-1], response = names(dp)[1])
cusum <- efp(f, data = dp, type = "Rec-CUSUM")
plot(cusum)
sctest(cusum)

rr <- recresid(f, data = dp)
n <- length(rr)
i <- 1:n
s <- cumsum(rr^2)/sum(rr^2)
c0 <- 0.08498
plot(i/n, s, type="l", xlab="t/n", ylab="CUSUMSQ", main="ARDL CUSUMSQ")
abline(0, 1, lty=2)
lines(i/n, i/n + c0, col="red")
lines(i/n, i/n - c0, col="red")

ect <- ardl_ecm$model[, "ect"]            # equilibrium errors from recm
oc  <- efp(ect ~ 1, type = "OLS-CUSUM"); plot(oc); sctest(oc)
bp  <- breakpoints(ect ~ 1, h = 0.15)
print(sctest(oc))     # report this p-value
summary(bp)           # "Optimal number of breakpoints: 0"
if (length(na.omit(breakdates(bp)))) confint(bp)

# Saving fitted models
saveRDS(ardl_best, "R/models/ardl_best.rds")
saveRDS(ardl_ecm, "R/models/ardl_ecm.rds")