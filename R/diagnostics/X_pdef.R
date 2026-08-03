# Robustness: house prices and construction costs deflated with p_def
# (household disposable income deflator) instead of the headline gdp_def.
# Standalone - reads the master CSV directly, writes nothing the main pipeline reads.
# r3 stays NOMINAL throughout - this checks the price/cost deflator only,
# kept separate from the real-rate robustness check (R/models/rrate/) so any
# shift here can't be conflated with the rate-definition effect found there.

library(dplyr)
library(readr)
library(zoo)
library(urca)
library(vars)
library(ARDL)

df <- read_csv("data/python_master/england_master.csv")
colnames(df)[1] <- "Date"
df <- na.omit(df)
df$Date <- as.Date(zoo::as.yearqtr(df$Date, format = "%YQ%q"))

df <- df %>%
  arrange(Date) %>%
  mutate(
    r3       = rate,
    lrprc    = log(hprice / p_def),
    lrcc     = log(cc / p_def),
    lhstarts = log(starts),
    lvol     = log(vol),
    lstock   = log(hstock)
  )

start_q <- as.numeric(format(df$Date[1], "%Y"))
start_qtr <- (as.numeric(format(df$Date[1], "%m")) + 2) / 3
cat(sprintf("Sample: %dQ%d - %s, n = %d\n\n", start_q, start_qtr,
            format(tail(df$Date, 1), "%Y-%m"), nrow(df)))

# Unit root: confirm lrprc/lrcc under p_def are still I(1), same order as headline.
adf <- function(x, type, lags = 4) summary(ur.df(na.omit(x), type = type, lags = lags,
                                                selectlags = "AIC"))
cat("\n--- ADF, lrprc (p_def) ---\n"); print(adf(df$lrprc, "drift"))
cat("\n--- ADF, lrcc (p_def) ---\n"); print(adf(df$lrcc, "drift"))

eng_tf <- df %>%
  dplyr::select(lhstarts, lrprc, lvol, r3, lstock, lrcc) %>%
  as.matrix()
eng_ts <- ts(eng_tf, start = c(start_q, start_qtr), frequency = 4)

ta <- time(eng_ts); dd <- function(yr) which.min(abs(ta - yr))
final_dummies <- matrix(0, nrow(eng_ts), 5,
                        dimnames = list(NULL, c("gfc_2008Q3","covid_2020Q2","covid_2020Q3","regstd_2023Q2","regstd_2023Q3")))
final_dummies[dd(2008.50), 1] <- 1
final_dummies[dd(2020.25), 2] <- 1
final_dummies[dd(2020.50), 3] <- 1
final_dummies[dd(2023.25), 4] <- 1
final_dummies[dd(2023.50), 5] <- 1

# VECM at the headline spec: r=1, K=5, ecdet="none" (case 3), season=4
jo <- ca.jo(eng_ts, type = "trace", ecdet = "none", K = 5, spec = "transitory", season = 4, dumvar = final_dummies)
cat("\n--- Johansen trace, p_def ---\n"); print(summary(jo))

vecm <- cajorls(jo, r = 1)
beta <- jo@V[, 1] / jo@V[1, 1]        # normalise on lhstarts
alpha <- vecm$rlm$coefficients["ect1", "lhstarts.d"]

cat("\n--- VECM long-run vector (normalised on lhstarts, p_def) ---\n")
print(round(-beta[-1], 4))            # sign-flipped to read as elasticities
cat(sprintf("\nAdjustment speed (alpha): %.4f\n", alpha))
cat(sprintf("Half-life (quarters): %.2f\n", log(0.5) / log(1 + alpha)))

# restrict alpha to zero for lrprc and r3, as in the headline test
vecm_u <- cajorls(jo, r = 1)
srlm <- summary(vecm_u$rlm)
b <- coef(vecm_u$rlm)["ect1", ]
E <- residuals(vecm_u$rlm)
X <- model.matrix(vecm_u$rlm)
g <- solve(crossprod(X))["ect1", "ect1"]

for (eqs in list(c("lrprc.d","r3.d"), c("lrprc.d","lvol.d","r3.d","lrcc.d"))) {
  a <- b[eqs]
  S <- crossprod(E[, eqs]) / (nrow(E) - ncol(X))
  W <- as.numeric(t(a) %*% solve(g * S) %*% a)
  cat(sprintf("p_def WE (%s): chi2(%d) = %.3f  p = %.4f\n",
              paste(eqs, collapse=","), length(eqs), W, pchisq(W, df=length(eqs), lower.tail=FALSE)))
}

# ARDL at the headline spec: lstock dropped, same dummies, same max_order
ta <- time(eng_ts); dd <- function(yr) which.min(abs(ta - yr))
D <- matrix(0, nrow(eng_ts), 4,
            dimnames = list(NULL, c("d08Q3", "d20Q2", "d20Q3", "d23Q2")))
D[dd(2008.50), 1] <- 1
D[dd(2020.25), 2] <- 1
D[dd(2020.50), 3] <- 1
D[dd(2023.25), 4] <- 1

eng_zoo <- as.zooreg(ts(cbind(eng_tf, D), start = c(start_q, start_qtr), frequency = 4))

mod <- auto_ardl(lhstarts ~ lrprc + lvol + r3 + lrcc | d08Q3 + d20Q2 + d20Q3 + d23Q2,
                 data = eng_zoo, max_order = 4)
cat("\n--- ARDL top orders, p_def ---\n"); print(mod$top_orders)

ardl_best <- mod$best_model
cat("\n--- ARDL bounds tests, p_def ---\n")
print(bounds_f_test(ardl_best, case = 3))
print(bounds_t_test(ardl_best, case = 3))

ardl_ecm <- recm(ardl_best, case = 3)
cat("\n--- ARDL ECM, p_def ---\n"); print(summary(ardl_ecm))

m <- multipliers(ardl_best)
cat("\n--- ARDL long-run multipliers, p_def ---\n"); print(m)

# ARDL fixed at the headline order for direct comparison
ardl_fixed <- ardl(lhstarts ~ lrprc + lvol + r3 + lrcc | d08Q3 + d20Q2 + d20Q3 + d23Q2,
                   data = eng_zoo, order = c(4,4,4,4,4))

# Side-by-side against the headline (gdp_def) results
headline <- c(lrprc = 0.845691, lvol = 0.729380, r3 = -0.022061, lrcc = -1.985059)
lr_pdef  <- setNames(m$Estimate, m$Term)[names(headline)]

m_fixed  <- multipliers(ardl_fixed)
lr_fixed <- setNames(m_fixed$Estimate, m_fixed$Term)[names(headline)]

cat("\n--- ARDL long-run elasticities: gdp_def vs p_def ---\n")
print(round(data.frame(
  gdp_def   = headline,
  pdef_auto = lr_pdef,
  pdef_fixed = lr_fixed
), 4))

cat(sprintf("\nECT: gdp_def -0.4823, p_def auto %.4f, p_def fixed %.4f\n",
            coef(ardl_ecm)["ect"], coef(recm(ardl_fixed, case = 3))["ect"]))

saveRDS(ardl_best, "R/models/diagnostics/ardl_best_pdef.rds")
saveRDS(ardl_ecm,  "R/models/diagnostics/ardl_ecm_pdef.rds")
saveRDS(jo,        "R/models/diagnostics/jo_pdef.rds")