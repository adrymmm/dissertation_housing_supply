library(ARDL)
library(strucchange)

# Recreating ts object
eng_ts <- ts(eng_tf, start = c(1975, 1), frequency = 4)

# Auto-select lag orders with AIC
mod <- auto_ardl(lhstarts ~ lrprc + lvol + r3 + lstock + lrcc + trend(lhstarts),
                 data = eng_ts,
                 max_order = 4)

# Top lag structures
mod$top_orders
ardl_best <- mod$best_model
summary(ardl_best)

# Bounds test for cointegration (Pesaran-Shin-Smith F-bounds)
bounds_f_test(ardl_best, case = 5)   # case 3 likely misspecified now testing case 5
# Reject null of no cointegration

# Reparametrize to the conditional ECM
ardl_ecm <- recm(ardl_best, case = 5) 

# Short-run ecm, speed of adjustment
summary(ardl_ecm)                       
# Long-run elasticities
multipliers(ardl_best)                

# Prepare data for strucchange by adding a plain numeric trend column
dat_clean <- as.data.frame(eng_ts)
dat_clean$linear_trend <- 1:nrow(dat_clean)

# CUSUM PLOT
ardl_cusum <- efp(lhstarts ~ lrprc + lvol + r3 + lstock + lrcc + linear_trend, 
                  data = dat_clean, 
                  type = "Rec-CUSUM")

# Draws the CUSUM process and boundaries automatically
plot(ardl_cusum, main = "ARDL Cumulative Sum of Recursive Residuals (CUSUM)")


# CUSUM SQUARED (CUSUMSQ) PLOT
# Extract the true recursive residuals from estimated ardl_best model
ardl_cusumsq <- efp(lhstarts ~ lrprc + lvol + r3 + lstock + lrcc + linear_trend, 
                    data = dat_clean, 
                    type = "OLS-CUSUM")

plot(ardl_cusumsq, main = "ARDL CUSUM of Squares (CUSUMSQ)")