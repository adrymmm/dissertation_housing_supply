# Robustness: nominal Bank Rate replaced with an ex-post real rate.
# Standalone - reads the master CSV directly, writes nothing the main pipeline reads.

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

# real rate = nominal Bank Rate - YoY inflation from the GDP/consumption deflator.
# p_def used here rather than gdp_def: the relevant deflator for a discount rate is a
# general price index, not construction costs. Costs 4 obs, so sample starts 1976Q1.
df <- df %>%
  arrange(Date) %>%
  mutate(
    infl_yoy = 100 * (log(p_def) - dplyr::lag(log(p_def), 4)),
    r3_nom   = rate,
    r3       = rate - infl_yoy,
    lrprc    = log(hprice / gdp_def),
    lrcc     = log(cc / gdp_def),
    lhstarts = log(starts),
    lvol     = log(vol),
    lstock   = log(hstock)
  ) %>%
  filter(!is.na(infl_yoy))

start_q <- as.numeric(format(df$Date[1], "%Y"))
start_qtr <- (as.numeric(format(df$Date[1], "%m")) + 2) / 3
cat(sprintf("Sample: %dQ%d - %s, n = %d\n\n", start_q, start_qtr,
            format(tail(df$Date, 1), "%Y-%m"), nrow(df)))

summary(df[, c("r3_nom", "r3", "infl_yoy")])

# Unit root: is the real rate still I(1)? If the order differs from the nominal rate
# that is itself a finding, not a nuisance.
adf <- function(x, type, lags = 4) summary(ur.df(na.omit(x), type = type, lags = lags,
                                                selectlags = "AIC"))
cat("\n--- ADF, real rate (levels, drift) ---\n"); print(adf(df$r3, "drift"))
cat("\n--- ADF, real rate (first difference, none) ---\n"); print(adf(diff(df$r3), "none"))
cat("\n--- ADF, nominal rate (levels, drift), for reference ---\n"); print(adf(df$r3_nom, "drift"))

eng_tf <- df %>%
  dplyr::select(lhstarts, lrprc, lvol, r3, lstock, lrcc) %>%
  as.matrix()
eng_ts <- ts(eng_tf, start = c(start_q, start_qtr), frequency = 4)

# VECM at the headline spec: r=1, K=5, ecdet="none" (case 3), season=4
jo <- ca.jo(eng_ts, type = "trace", ecdet = "none", K = 5, spec = "transitory", season = 4)
cat("\n--- Johansen trace, real rate ---\n"); print(summary(jo))

vecm <- cajorls(jo, r = 1)
beta <- jo@V[, 1] / jo@V[1, 1]        # normalise on lhstarts
alpha <- vecm$rlm$coefficients["ect1", "lhstarts.d"]

cat("\n--- VECM long-run vector (normalised on lhstarts, real rate) ---\n")
print(round(-beta[-1], 4))            # sign-flipped to read as elasticities
cat(sprintf("\nAdjustment speed (alpha): %.4f\n", alpha))
cat(sprintf("Half-life (quarters): %.2f\n", log(0.5) / log(1 + alpha)))

# restrict alpha to zero for lrprc and r3, as in the headline test
DA <- matrix(c(1,0,0,0,   # lhstarts  free
               0,0,0,0,   # lrprc     restricted
               0,1,0,0,   # lvol      free
               0,0,0,0,   # r3        restricted
               0,0,1,0,   # lstock    free
               0,0,0,1),  # lrcc      free
             nrow = 6, byrow = TRUE)
summary(alrtest(jo, A = DA, r = 1))

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
cat("\n--- ARDL top orders, real rate ---\n"); print(mod$top_orders)

ardl_best <- mod$best_model
cat("\n--- ARDL bounds tests, real rate ---\n")
print(bounds_f_test(ardl_best, case = 3))
print(bounds_t_test(ardl_best, case = 3))

ardl_ecm <- recm(ardl_best, case = 3)
cat("\n--- ARDL ECM, real rate ---\n"); print(summary(ardl_ecm))

m <- multipliers(ardl_best)
cat("\n--- ARDL long-run multipliers, real rate ---\n"); print(m)

# ARDL copying previous spec
ardl_fixed <- ardl(lhstarts ~ lrprc + lvol + r3 + lrcc | d08Q3 + d20Q2 + d20Q3 + d23Q2,
                   data = eng_zoo, order = c(4,4,4,4,4))

# Side-by-side against the headline (nominal-rate) results
headline <- c(lrprc = 0.845691, lvol = 0.729380, r3 = -0.022061, lrcc = -1.985059)
lr_real <- setNames(m$Estimate, m$Term)[names(headline)]

m_fixed <- multipliers(ardl_fixed)
lr_fixed <- setNames(m_fixed$Estimate, m_fixed$Term)[names(headline)]

cat("\n--- ARDL long-run elasticities: nominal vs real rate ---\n")
print(round(data.frame(
  nominal    = headline,
  real_auto  = lr_real,
  real_fixed = lr_fixed
), 4))

cat(sprintf("\nECT: nominal -0.4823, real auto %.4f, real fixed %.4f\n",
            coef(ardl_ecm)["ect"], coef(recm(ardl_fixed, case = 3))["ect"]))

saveRDS(ardl_best, "R/models/rrate/ardl_best_rrate.rds")
saveRDS(ardl_ecm,  "R/models/rrate/ardl_ecm_rrate.rds")
saveRDS(jo,        "R/models/rrate/jo_rrate.rds")


# --- Findings -----------------------------------------------------------------
# Real rate is I(1) like the nominal rate (ADF -2.49 in levels, -9.06 in
# differences), so the specification is a valid substitution. Weak exogeneity of
# lrprc and r3 still holds (LR = 1.96, p = 0.38, vs 1.78/0.41 in the headline).
# Elasticities keep their signs and ordering: at the headline lag order the price
# elasticity rises 0.85 -> 1.14 and cost -1.99 -> -2.39, so cost remains roughly
# twice price in magnitude. The real-rate ARDL price elasticity (1.14) sits almost
# exactly on the VECM headline (~1.13), i.e. the two methods agree more closely
# under this specification than under the nominal one. Adjustment is faster
# (ECT -0.48 -> -0.67 at fixed lags, -0.89 under re-selected lags), about half
# attributable to the rate definition and half to lag order; a half-life under one
# quarter is implausibly fast for starts and likely reflects the higher volatility
# of the ex-post real rate loading into the correction term. Rank evidence is
# weaker: trace rejects r <= 1 at 5% but not 1%. Conclusion: the structural
# results are robust to the rate definition; nominal is retained as the headline
# for comparability with Anastasiou (2023), not because the real rate fails.
# Note the price elasticity shifts +35% across these two defensible
# specifications - larger than the +25% reform uplift in the elasticity
# counterfactual - so that counterfactual's baseline should be read as a
# proportional shift, not a precisely known starting value.
# ------------------------------------------------------------------------------