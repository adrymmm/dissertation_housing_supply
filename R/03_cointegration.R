library(urca)
library(vars)


# Checking for seasonality
eng_ts <- ts(eng_tf, start=c(1975,1), frequency=4)
for (v in colnames(eng_tf)) {
  q  <- factor(cycle(eng_ts[, v]))
  m  <- lm(diff(eng_ts[, v]) ~ factor(cycle(eng_ts[, v])[-1]))  # on diffs, avoids trend
  f  <- summary(m)$fstatistic
  p  <- pf(f[1], f[2], f[3], lower.tail = FALSE)
  cat(sprintf("%-9s seasonal F p-value = %.4f  %s\n",
              v, p, ifelse(p < 0.05, "<- seasonal (NSA-like)", "smooth (SA-like)")))
}

# Lag order var in levels BIC
lagsel_eng <- VARselect(eng_tf, lag.max = pmax_eng, type = "both")
print(lagsel_eng$selection)          
K_eng <- lagsel_eng$selection["SC(n)"]

# Johansen trace test, restricted trend + centred seasonals
jo_eng <- ca.jo(eng_tf,
                type    = "trace",      
                ecdet   = "trend",      # restricted linear trend (Case 4)
                K       = K_eng,        
                spec    = "transitory", 
                season  = 4)            
summary(jo_eng)

# Eigen test
jo_eng_eig <- ca.jo(eng_tf, type = "eigen", ecdet = "trend",
                    K = K_eng, spec = "transitory", season = 4)
summary(jo_eng_eig)

# Serial Correlation test
var_eng <- vec2var(jo_eng, r = 3) 
serial.test(var_eng, lags.pt = 16, type = "PT.asymptotic")

# Considerable serial correlation need to test with higher lag order

# Testing serial correlation over K=2-6
for (k in 2:6) {
  jo_k  <- ca.jo(eng_tf, type="trace", ecdet="trend", K=k, spec="transitory", season=4)
  var_k <- vec2var(jo_k, r = 1)          
  pt    <- serial.test(var_k, lags.pt = 16, type = "PT.asymptotic")
  cat(sprintf("K=%d  PT chi2=%.1f  df=%d  p=%.4g\n",
              k, pt$serial$statistic, pt$serial$parameter, pt$serial$p.value))
}
# K=5 gives best result

jo_eng_k5 <- ca.jo(eng_tf, type="trace", ecdet="trend", K=5, spec="transitory", season=4)
summary(jo_eng_k5)

jo_eng_k5_eig <- ca.jo(eng_tf, type="eigen", ecdet="trend", K=5, spec="transitory", season=4)
summary(jo_eng_k5_eig)