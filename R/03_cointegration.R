library(urca)
library(vars)
library(strucchange)

# Converting to timeseries
eng_ts <- ts(eng_tf, start=c(1975,1), frequency=4)
time_axis <- time(eng_ts)

for (v in colnames(eng_tf)) {
  q  <- factor(cycle(eng_ts[, v]))
  m  <- lm(diff(eng_ts[, v]) ~ factor(cycle(eng_ts[, v])[-1]))  # on diffs, avoids trend
  f  <- summary(m)$fstatistic
  p  <- pf(f[1], f[2], f[3], lower.tail = FALSE)
  cat(sprintf("%-9s seasonal F p-value = %.4f  %s\n",
              v, p, ifelse(p < 0.05, "<- seasonal (NSA-like)", "smooth (SA-like)")))
}

# Find optimal structural breaks for a variable over time Bai & Perron (2003)
bp_test <- breakpoints(eng_ts ~ 1, breaks = 1)
summary(bp_test)

cat("Break Date:", format(time_axis[133]), "\n")


# Initialising dummy matrix
final_dummies <- matrix(0, nrow = nrow(eng_ts), ncol = 1)

# First Bai & Perron Break (2008 Q1)
final_dummies[time_axis >= 2008.00, 1] <- 1
colnames(final_dummies) <- c("shock_2008Q1")

# Lag order var in levels BIC
lagsel_eng <- VARselect(eng_tf, lag.max = pmax_eng, type = "both")
print(lagsel_eng$selection)          
K_eng <- lagsel_eng$selection["AIC(n)"]

# Johansen trace test, restricted trend + centred seasonals
jo_eng <- ca.jo(eng_tf,
                type    = "trace",      
                ecdet   = "none",      # unrestricted constant (Case 3)
                K       = K_eng,        
                spec    = "transitory", 
                season  = 4,
                dumvar = final_dummies)            
summary(jo_eng)

# Eigen test
jo_eng_eig <- ca.jo(eng_tf, type = "eigen", ecdet = "none",
                    K = K_eng, spec = "transitory", season = 4,
                    dumvar = final_dummies)
summary(jo_eng_eig)

# Serial Correlation test
var_eng <- vec2var(jo_eng, r = 1) 
serial.test(var_eng, lags.pt = 16, type = "PT.asymptotic")
serial.test(var_eng, lags.bg = 4, type = "BG")

# Considerable serial correlation need to test with higher lag order

# Testing serial correlation over K=2-6
for (k in 2:12) {
  jo_k  <- ca.jo(eng_tf, type="trace", ecdet="none", K=k, spec="transitory", season=4, dumvar=final_dummies)
  var_k <- vec2var(jo_k, r = 2)          
  pt    <- serial.test(var_k, lags.pt = 16, type = "PT.asymptotic")
  cat(sprintf("K=%d  PT chi2=%.1f  df=%d  p=%.4g\n",
              k, pt$serial$statistic, pt$serial$parameter, pt$serial$p.value))
}
# K=6 gives best result

# Diagnostic: 5-var Johansen excluding lstock 
eng_tf_5 <- eng_tf[, c("lhstarts","lrprc","lvol","r3","lrcc")]

lagsel_5 <- VARselect(eng_tf_5, lag.max = pmax_eng, type = "both")
print(lagsel_5$selection)
K_5 <- as.integer(unname(lagsel_5$selection["AIC(n)"]))

jo_5 <- ca.jo(eng_tf_5, type="trace", ecdet="none",
              K=K_5, spec="transitory", season=4, dumvar = final_dummies)
summary(jo_5)

jo_5_eig <- ca.jo(eng_tf_5, type="eigen", ecdet="none",
                  K=K_5, spec="transitory", season=4, dumvar = final_dummies)
summary(jo_5_eig)




# Formally checking deterministic terms

specs <- c(case2="const", case3="none", case4="trend")
jo <- lapply(specs, function(e)
  ca.jo(eng_tf, type="trace", ecdet=e, K=5, spec="transitory", season=4))

# tabulate trace stat vs 5% crit, by rank, for each case
sapply(jo, function(z) z@teststat)          # trace stats
sapply(jo, function(z) z@cval[,"5pct"])      # critical values

# Define the number of observations (T) and variables (n)
T <- nrow(eng_tf) - 5  # Total observations minus your lag length K=5
n <- ncol(eng_tf)

# Extract the eigenvalues from both models
lambda_case2 <- jo$case2@lambda
lambda_case3 <- jo$case3@lambda

# Choose the rank (r) you want to test under. 
# Based on your output, let's test at r = 1
r <- 1

# Calculating lr stat
lr_stat <- T * sum(log((1 - lambda_case3[1:r]) / (1 - lambda_case2[1:r])))
df <- r
p_value <- pchisq(lr_stat, df = df, lower.tail = FALSE)

# Print results
cat("LR Statistic:", lr_stat, "\n")
cat("Degrees of Freedom:", df, "\n")
cat("P-value:", p_value, "\n")