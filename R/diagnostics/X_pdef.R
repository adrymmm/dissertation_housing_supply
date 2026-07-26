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

# VECM at the headline spec: r=1, K=5, ecdet="none" (case 3), season=4
jo <- ca.jo(eng_ts, type = "trace", ecdet = "none", K = 5, spec = "transitory", season = 4)
cat("\n--- Johansen trace, p_def ---\n"); print(summary(jo))

vecm <- cajorls(jo, r = 1)
beta <- jo@V[, 1] / jo@V[1, 1]        # normalise on lhstarts
alpha <- vecm$rlm$coefficients["ect1", "lhstarts.d"]

cat("\n--- VECM long-run vector (normalised on lhstarts, p_def) ---\n")
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

# --- Summary --------------------------------------------------------------
# Substituting p_def for gdp_def in lrprc/lrcc leaves lrprc and lrcc I(1)
# (ADF -1.59, -1.53), so the swap is mechanically valid. ARDL is robust:
# auto_ardl selects the same (4,4,4,4,4) order, bounds test moves from
# borderline to decisive (F=5.19 p=0.008, t=-4.71 p=0.007), and elasticities
# hold sign/ordering within 6-14% (price 0.85->0.78, cost -1.99->-1.70,
# ECT -0.48->-0.51) - a second specification, alongside the real rate check,
# in which the headline's borderline bounds result resolves cleanly. The
# VECM is not robust: price roughly halves (1.13->0.54) because price and
# cost load on the CE at -1.45 net rather than equal-and-opposite, so the
# ~0.14 log-point trend in the gdp_def/p_def wedge does not cancel and
# instead redistributes across the loadings (ecdet="none" leaves no trend
# term to absorb it). Decisive: weak exogeneity of lrprc/r3 is REJECTED
# under p_def (LR=9.2, df=2, p=0.01), driven by r3.d loading +1.85 on the
# first CE, vs. held comfortably under gdp_def (1.78, p=0.41) and the real
# rate (1.96, p=0.38). Since weak exogeneity underpins both the ARDL
# conditional specification and the OBR-conditioned forward projections,
# gdp_def is retained as headline on this basis, not merely for
# Michalis comparability. Net effect: price elasticity now spans ~0.54-1.14
# across defensible specifications - wider than the +/-25%/50% elasticity
# counterfactual grid - but the counterfactual already showed a 50% swing
# moves cumulative net additions by ~1,300 homes against a ~498k shortfall,
# so this imprecision does not threaten the missed-target conclusion.