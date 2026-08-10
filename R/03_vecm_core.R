library(urca)
library(vars)

eng_tf <- readRDS("R/models/eng_tf.rds")
eng_ts <- ts(eng_tf, start = c(1975, 1), frequency = 4)
time_axis <- time(eng_ts)
dd <- function(yr) which.min(abs(time_axis - yr))

# Seasonality Tests
for (v in colnames(eng_tf)) {
  m <- lm(diff(eng_ts[, v]) ~ factor(cycle(eng_ts[, v])[-1]))
  f <- summary(m)$fstatistic
  p <- pf(f[1], f[2], f[3], lower.tail = FALSE)
  cat(sprintf("%-9s seasonal F p = %.4f  %s\n",
              v, p, ifelse(p < 0.05, "<- seasonal", "smooth")))
}

# ---- Impulse dummies (a priori, documented events) -----------------------
final_dummies <- matrix(0, nrow(eng_ts), 5)
colnames(final_dummies) <- c("gfc_2008Q3", "covid_2020Q2", "covid_2020Q3",
                             "regstd_2023Q2", "regstd_2023Q3")
final_dummies[dd(2008.50), 1] <- 1
final_dummies[dd(2020.25), 2] <- 1
final_dummies[dd(2020.50), 3] <- 1
final_dummies[dd(2023.25), 4] <- 1
final_dummies[dd(2023.50), 5] <- 1

# ---- Lag order: K=5 selected by FPE, corroborated by a residual
# serial-correlation search across the full lag range (U-shape minimum,
# confirmed independently on the five-variable system). HQ/SC prefer K=2;
# AIC hits the pmax boundary (uninformative). Diagnostics in diagnostics script.
pmax_eng <- floor(12 * (nrow(eng_tf) / 100)^(1/4))
lagsel <- VARselect(eng_tf, lag.max = pmax_eng, type = "const",
                    season = 4, exogen = final_dummies)
print(lagsel$selection)
K <- 5L

# ---- Headline VECM: Case 3 (ecdet="none"), rank test ---------------------
jo_eng <- ca.jo(eng_tf, type = "trace", ecdet = "none", K = K,
                spec = "transitory", season = 4, dumvar = final_dummies)
summary(jo_eng)

jo_eng_eig <- ca.jo(eng_tf, type = "eigen", ecdet = "none", K = K,
                    spec = "transitory", season = 4, dumvar = final_dummies)
summary(jo_eng_eig)

r <- 1L
var_eng <- vec2var(jo_eng, r = r)

# ---- Weak exogeneity: alrtest (headline method) --------------------------
make_A <- function(free_vars, all_vars) {
  A <- matrix(0, length(all_vars), length(free_vars))
  A[match(free_vars, all_vars), ] <- diag(length(free_vars))
  A
}
vars6   <- colnames(eng_tf)
A_full  <- make_A(c("lhstarts", "lstock"), vars6)             # tests (lrprc,lvol,r3,lrcc)
A_pair  <- make_A(setdiff(vars6, c("lrprc", "lrcc")), vars6)  # tests (lrprc,lrcc) only

cat("Full conditioning set:\n"); print(summary(alrtest(jo_eng, A = A_full, r = r)))
cat("(lrprc, lrcc) pair:\n");    print(summary(alrtest(jo_eng, A = A_pair, r = r)))

# Individual ect1 loadings (t-stats) for the write-up's decomposition
vecm_u <- cajorls(jo_eng, r = r)
srlm <- summary(vecm_u$rlm)
for (v in c("lrprc.d", "lvol.d", "r3.d", "lrcc.d")) {
  cat(sprintf("WE %-9s t = %.3f\n", v,
              srlm[[paste0("Response ", v)]]$coefficients["ect1", "t value"]))
}

saveRDS(jo_eng, "R/models/jo_eng.rds")
saveRDS(var_eng, "R/models/var_eng.rds")
saveRDS(final_dummies, "R/models/final_dummies.rds")