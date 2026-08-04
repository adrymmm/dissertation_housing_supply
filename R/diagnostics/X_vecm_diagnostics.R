library(urca)
library(vars)

eng_tf         <- readRDS("R/models/eng_tf.rds")
eng_ts         <- ts(eng_tf, start = c(1975, 1), frequency = 4)
final_dummies  <- readRDS("R/models/final_dummies.rds")
jo_eng         <- readRDS("R/models/jo_eng.rds")
var_eng        <- readRDS("R/models/var_eng.rds")
K <- 5L
r <- 1L

# ---- Serial Autocorrelation Lag order test ---------------
serial_scan <- function(data, r, dummies, k_max, lags.pt = 16) {
  for (k in 2:k_max) {
    var_k <- vec2var(ca.jo(data, type = "trace", ecdet = "none", K = k,
                           spec = "transitory", season = 4, dumvar = dummies), r = r)
    st <- serial.test(var_k, lags.pt = lags.pt, type = "PT.adjusted")$serial
    cat(sprintf("K=%2d  chi2=%.1f  df=%d  ratio=%.2f  p=%.4g\n",
                k, st$statistic, st$parameter, st$statistic / st$parameter, st$p.value))
  }
}

serial_scan(eng_tf,   r = 1, dummies = final_dummies, k_max = pmax_eng)
normality.test(var_eng)

# ---- 5-variable system excluding lstock ----------------------
eng_tf_5 <- eng_tf[, c("lhstarts", "lrprc", "lvol", "r3", "lrcc")]
jo_5 <- ca.jo(eng_tf_5, type = "trace", ecdet = "none", K = K,
              spec = "transitory", season = 4, dumvar = final_dummies)
summary(jo_5)

serial_scan(eng_tf_5, r = 2, dummies = final_dummies, k_max = pmax_eng)
var_5 <- vec2var(jo_5, r = 2)
normality.test(var_5)

# Is the second cointegrating vector just lhstarts stationarity?
# bh5lrtest, r1=1 known vector, r=2 total, one free vector remains.
# CAUTION: eigenvectors/weights for the free column print as NaN/Inf -
# possible normalisation degeneracy at r1=1. LR stat/df/p may still be
# valid but treat as unconfirmed until cross-checked or urca source read.
# Result so far: chi2=8.7, df=3, p=0.03 -> REJECTS. Second vector is NOT
# just the stationarity direction.
H_stat <- matrix(c(1, 0, 0, 0, 0), c(5, 1))
summary(bh5lrtest(z = jo_5, H = H_stat, r = 2))

# Independent sanity check, not relying on bh5lrtest's own eigenvector output
round(jo_5@V[, 1:2], 3)

# ---- Outlier scan - POST-ESTIMATION check only, not the dummy-selection
# method (dummies are chosen a priori in the core script; dummied dates
# cannot appear here by construction since dumvar already absorbs them).
res       <- residuals(var_eng)
res_dates <- tail(time(eng_ts), nrow(res))
z   <- scale(res)
idx <- which(abs(z) > 3, arr.ind = TRUE)
print(data.frame(date = round(res_dates[idx[, 1]], 2),
                 eq   = colnames(res)[idx[, 2]],
                 z    = round(z[idx], 2))[order(-abs(z[idx])), ])

# ---- Weak exogeneity: rank sensitivity (r=2) -----------------------------
# At r=2, even the (lrprc,lrcc) pair rejects weak exogeneity - the
# conclusion is NOT robust to rank choice. See bh5lrtest result above for
# why the r=1/r=2 tension can't be dismissed as an artefact.
make_A <- function(free_vars, all_vars) {
  A <- matrix(0, length(all_vars), length(free_vars))
  A[match(free_vars, all_vars), ] <- diag(length(free_vars))
  A
}
vars6  <- colnames(eng_tf)
A_full <- make_A(c("lhstarts", "lstock"), vars6)
A_pair <- make_A(setdiff(vars6, c("lrprc", "lrcc")), vars6)
cat("r=2, full set:\n"); print(summary(alrtest(jo_eng, A = A_full, r = 2)))
cat("r=2, pair:\n");     print(summary(alrtest(jo_eng, A = A_pair, r = 2)))

# ---- Michalis comparability: does dropping impulse dummies alone move
# the VECM toward his figures? K held at 5 - isolates dummy treatment,
# not lag order (his implied VAR order is 2, per lag-order memo).
pmax_eng <- floor(12 * (nrow(eng_tf) / 100)^(1/4))

jo_nodum <- ca.jo(eng_tf, type = "trace", ecdet = "none", K = 5L,
                  spec = "transitory", season = 4)
summary(jo_nodum)

jo_nodum_eig <- ca.jo(eng_tf, type = "eigen", ecdet = "none", K = 5L,
                      spec = "transitory", season = 4)
summary(jo_nodum_eig)
# check trace/eigen above before trusting r=1 - rank has already flipped
# with dummy count in this exact system (r=2 four-dummy, r=1 three/five)

r_nodum <- 1L  # revise if trace/eigen disagree
vecm_nodum <- cajorls(jo_nodum, r = r_nodum)
beta_nodum <- vecm_nodum$beta
print(round(beta_nodum / -beta_nodum["lhstarts.l1", ], 3))  # normalise lhstarts=1

srlm_nodum <- summary(vecm_nodum$rlm)
ect_nodum  <- srlm_nodum[["Response lhstarts.d"]]$coefficients["ect1", ]
cat(sprintf("no-dummy ECT: coef=%.4f  t=%.3f  half-life=%.2fq\n",
            ect_nodum["Estimate"], ect_nodum["t value"],
            log(0.5) / log(1 + ect_nodum["Estimate"])))

# five-dummy comparator for the same three figures, side by side
vecm_5dum  <- cajorls(jo_eng, r = 1)
beta_5dum  <- vecm_5dum$beta
print(round(beta_5dum / -beta_5dum["lhstarts.l1", ], 3))
srlm_5dum  <- summary(vecm_5dum$rlm)
ect_5dum   <- srlm_5dum[["Response lhstarts.d"]]$coefficients["ect1", ]
cat(sprintf("five-dummy ECT: coef=%.4f  t=%.3f  half-life=%.2fq\n",
            ect_5dum["Estimate"], ect_5dum["t value"],
            log(0.5) / log(1 + ect_5dum["Estimate"])))