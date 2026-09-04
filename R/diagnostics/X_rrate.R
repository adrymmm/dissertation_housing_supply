# Robustness: nominal Bank Rate replaced with an ex-post real rate.
# Standalone - reads the master CSV directly, writes nothing the main pipeline reads.
# Mirrors the current headline specs in R/03_vecm_core.R and R/05_ARDL.R -
# dummies, lag orders, and WE test all reread from those scripts/their saved
# outputs so this stays in sync automatically as the headline pipeline moves.

library(dplyr)
library(readr)
library(zoo)
library(urca)
library(vars)
library(ARDL)
source("R/functions/vecm_functions.R")   # trace_rank_decision

# Headline reference, pulled live so this file can't silently go stale
headline_coefs  <- read_csv("R/models/ardl_coefs_full.csv", show_col_types = FALSE)
headline        <- setNames(headline_coefs$estimate[headline_coefs$type == "longrun"],
                            headline_coefs$term[headline_coefs$type == "longrun"])
headline_ect    <- headline_coefs$estimate[headline_coefs$type == "ecm" & headline_coefs$term == "ect"]
headline_order  <- readRDS("R/models/ardl_best.rds")$order

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

# Impulse dummies, same 5 as the headline (regstd_2023Q3 included)
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
cat("\n--- Johansen trace, real rate ---\n"); print(summary(jo))
trace_rank_decision(jo)

vecm <- cajorls(jo, r = 1)
beta <- jo@V[, 1] / jo@V[1, 1]        # normalise on lhstarts
alpha <- vecm$rlm$coefficients["ect1", "lhstarts.d"]

cat("\n--- VECM long-run vector (normalised on lhstarts, real rate) ---\n")
print(round(-beta[-1], 4))            # sign-flipped to read as elasticities
cat(sprintf("\nAdjustment speed (alpha): %.4f\n", alpha))
cat(sprintf("Half-life (quarters): %.2f\n", log(0.5) / log(1 + alpha)))

# Weak exogeneity, same alrtest specification as the headline (R/03_vecm_core.R):
# full conditioning set (lrprc,lvol,r3,lrcc jointly restricted, lhstarts+lstock free)
# and the (lrprc,lrcc) pair alone.
make_A <- function(free_vars, all_vars) {
  A <- matrix(0, length(all_vars), length(free_vars))
  A[match(free_vars, all_vars), ] <- diag(length(free_vars))
  A
}
vars6  <- colnames(eng_tf)
A_full <- make_A(c("lhstarts", "lstock"), vars6)
A_pair <- make_A(setdiff(vars6, c("lrprc", "lrcc")), vars6)

cat("\n--- real-rate weak exogeneity (headline spec) ---\n")
cat("Full conditioning set (lrprc,lvol,r3,lrcc):\n"); print(summary(alrtest(jo, A = A_full, r = 1)))
cat("(lrprc, lrcc) pair:\n");                          print(summary(alrtest(jo, A = A_pair, r = 1)))

# ARDL at the headline spec: lstock dropped, same 5 dummies + centred
# seasonals, same max_order (R/05_ARDL.R)
D <- matrix(0, nrow(eng_ts), 5,
            dimnames = list(NULL, c("d08Q3", "d20Q2", "d20Q3", "d23Q2", "d23Q3")))
D[dd(2008.50), 1] <- 1
D[dd(2020.25), 2] <- 1
D[dd(2020.50), 3] <- 1
D[dd(2023.25), 4] <- 1
D[dd(2023.50), 5] <- 1

q <- cycle(eng_ts)
S <- outer(as.numeric(q), 1:3, "==") - 1/4
colnames(S) <- c("sd1", "sd2", "sd3")

eng_zoo <- as.zooreg(ts(cbind(eng_tf, D, S), start = c(start_q, start_qtr), frequency = 4))

mod <- auto_ardl(lhstarts ~ lrprc + lvol + r3 + lrcc |
                   d08Q3 + d20Q2 + d20Q3 + d23Q2 + d23Q3 + sd1 + sd2 + sd3,
                 data = eng_zoo, max_order = 6)
cat("\n--- ARDL top orders, real rate ---\n"); print(mod$top_orders)

ardl_best <- mod$best_model
cat("\n--- ARDL bounds tests, real rate ---\n")
print(bounds_f_test(ardl_best, case = 3))
print(bounds_t_test(ardl_best, case = 3))

ardl_ecm <- recm(ardl_best, case = 3)
cat("\n--- ARDL ECM, real rate ---\n"); print(summary(ardl_ecm))

m <- multipliers(ardl_best)
cat("\n--- ARDL long-run multipliers, real rate ---\n"); print(m)

# ARDL fixed at the current headline-selected order for direct comparison
ardl_fixed <- ardl(lhstarts ~ lrprc + lvol + r3 + lrcc |
                     d08Q3 + d20Q2 + d20Q3 + d23Q2 + d23Q3 + sd1 + sd2 + sd3,
                   data = eng_zoo, order = headline_order)

# Side-by-side against the headline (nominal-rate) results
lr_real <- setNames(m$Estimate, m$Term)[names(headline)]

m_fixed <- multipliers(ardl_fixed)
lr_fixed <- setNames(m_fixed$Estimate, m_fixed$Term)[names(headline)]

cat("\n--- ARDL long-run elasticities: nominal vs real rate ---\n")
print(round(data.frame(
  nominal    = headline,
  real_auto  = lr_real,
  real_fixed = lr_fixed
), 4))

cat(sprintf("\nECT: nominal %.4f, real auto %.4f, real fixed %.4f\n",
            headline_ect, coef(ardl_ecm)["ect"], coef(recm(ardl_fixed, case = 3))["ect"]))

saveRDS(ardl_best, "R/models/diagnostics/ardl_best_rrate.rds")
saveRDS(ardl_ecm,  "R/models/diagnostics/ardl_ecm_rrate.rds")
saveRDS(jo,        "R/models/diagnostics/jo_rrate.rds")