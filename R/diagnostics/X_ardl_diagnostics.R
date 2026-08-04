library(ARDL)
library(strucchange)
library(lmtest)

eng_tf <- readRDS("R/models/eng_tf.rds")
eng_ts <- ts(eng_tf, start = c(1975, 1), frequency = 4)

ta <- time(eng_ts); dd <- function(yr) which.min(abs(ta - yr))
D <- matrix(0, nrow(eng_ts), 5,
            dimnames = list(NULL, c("d08Q3","d20Q2","d20Q3","d23Q2","d23Q3")))
D[dd(2008.50),1] <- 1
D[dd(2020.25),2] <- 1
D[dd(2020.50),3] <- 1
D[dd(2023.25),4] <- 1
D[dd(2023.50),5] <- 1

q <- cycle(eng_ts)
S <- outer(as.numeric(q), 1:3, "==") - 1/4
colnames(S) <- c("sd1","sd2","sd3")

eng_zoo <- as.zooreg(ts(cbind(as.matrix(eng_tf), D, S), start = c(1975, 1), frequency = 4))

mod       <- readRDS("R/models/ardl_mod.rds")
ardl_best <- readRDS("R/models/ardl_best.rds")
ardl_ecm  <- readRDS("R/models/ardl_ecm.rds")

# --- lag order: was max_order actually interior? ---
print(mod$top_orders)

# --- bounds test for cointegration ---
print(bounds_f_test(ardl_best, case = 3))
print(bounds_t_test(ardl_best, case = 3))

# --- residual autocorrelation: justifies including seasonals at all ---
print(bgtest(ardl_best, order = 4))
print(bgtest(ardl_best, order = 8))

# --- FWL-purged stability tests ---
d <- ardl_best$model
names(d) <- make.names(names(d))
dum  <- c("d08Q3","d20Q2","d20Q3","d23Q2","d23Q3","sd1","sd2","sd3")
core <- setdiff(names(d), dum)

Dm    <- cbind(1, as.matrix(d[, dum]))
purge <- function(z) lm.fit(Dm, z)$residuals
dp    <- as.data.frame(lapply(d[, core], purge))
names(dp) <- core
f     <- reformulate(names(dp)[-1], response = names(dp)[1])

cusum <- efp(f, data = dp, type = "Rec-CUSUM")
plot(cusum)
print(sctest(cusum))

rr <- recresid(f, data = dp)
n <- length(rr)
i <- 1:n
s <- cumsum(rr^2)/sum(rr^2)
cat("n =", n, "\n")
c0 <- 0.1377979  # simulated 5% CUSUMSQ boundary for n = 179 (B = 100,000 reps)
cat("CUSUMSQ max |s - i/n| =", max(abs(s - i/n)), " vs c0 =", c0, "\n")
plot(i/n, s, type = "l", xlab = "t/n", ylab = "CUSUMSQ", main = "ARDL CUSUMSQ")
abline(0, 1, lty = 2)
lines(i/n, i/n + c0, col = "red")
lines(i/n, i/n - c0, col = "red")

ect <- ardl_ecm$model[, "ect"]
oc  <- efp(ect ~ 1, type = "OLS-CUSUM"); plot(oc)
print(sctest(oc))
bp  <- breakpoints(ect ~ 1, h = 0.15)
print(summary(bp))
if (length(na.omit(breakdates(bp)))) print(confint(bp))

# --- lstock robustness: is exclusion principled, or just convenient? ---
mod_full <- auto_ardl(lhstarts ~ lrprc + lvol + r3 + lrcc + lstock |
                        d08Q3 + d20Q2 + d20Q3 + d23Q2 + d23Q3 + sd1 + sd2 + sd3,
                      data = eng_zoo, max_order = 6)
ardl_full <- mod_full$best_model
cat("With-lstock order:", paste(ardl_full$order, collapse = ","), "\n")
print(bounds_f_test(ardl_full, case = 3))
print(bounds_t_test(ardl_full, case = 3))
print(multipliers(ardl_full))
saveRDS(ardl_full, "R/diagnostics/ardl_full_robustness.rds")