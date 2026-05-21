library(ARDL)
library(strucchange)

# Converting to dataframe
dat <- as.data.frame(eng_tf)

# Auto-select lag orders with AIC
mod <- auto_ardl(lhstarts ~ lrprc + lvol + r3 + lstock + lrcc,
                 data = dat, max_order = 4)

# Top lag structures
mod$top_orders
ardl_best <- mod$best_model
summary(ardl_best)

# Bounds test for cointegration (Pesaran-Shin-Smith F-bounds)
bounds_f_test(ardl_best, case = 3)   # case 3 likely misspecified

# Reparametrize to the conditional ECM
ardl_ecm <- recm(ardl_best, case = 3) 
# Short-run ecm, speed of adjustment
summary(ardl_ecm)                       
# Long-run elasticities
multipliers(ardl_best)                

df <- ardl_best$model
names(df) <- make.names(names(df))   # L(lhstarts, 1) -> L.lhstarts..1.
f  <- as.formula(paste(names(df)[1], "~", paste(names(df)[-1], collapse = " + ")))

# CUSUM
plot(efp(f, data = df, type = "Rec-CUSUM"), main = "CUSUM")

# CUSUMSQ
rr <- recresid(f, data = df)
n  <- length(rr); sq <- cumsum(rr^2)/sum(rr^2); i <- seq_along(sq)
plot(i, sq, type = "l", main = "CUSUMSQ", xlab = "", ylab = "")
lines(i, i/n + 0.948/sqrt(n), lty = 2, col = "red")
lines(i, i/n - 0.948/sqrt(n), lty = 2, col = "red")