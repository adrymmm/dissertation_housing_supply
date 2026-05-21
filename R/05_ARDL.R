library(ARDL)

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