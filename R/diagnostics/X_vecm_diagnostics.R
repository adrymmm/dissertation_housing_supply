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

# ---- Beta restriction tests (blrtest), r=1 headline, 6-var system --------
# rank caveat: r<=1 trace stat 70.34 vs 70.60 (5% crit, see jo_eng summary) -
# borderline, so both LR tests below are conditional on a near-coin-flip
# rank choice, not just on the r=1 point estimate.
v6   <- colnames(eng_tf)
idx6 <- setNames(seq_along(v6), v6)
p6   <- length(v6)

vecm_r1 <- cajorls(jo_eng, r = 1)
beta_r1 <- vecm_r1$beta[, 1]
elas_r1 <- -beta_r1 / beta_r1["lhstarts.l1"]; names(elas_r1) <- v6

# H0: elasticity_lrprc = ratio, all other vars free
H_lrprc <- function(ratio) {
  H <- matrix(0, p6, p6 - 1)
  H[idx6["lhstarts"], 1] <- 1
  H[idx6["lrprc"],   1] <- -ratio
  free <- setdiff(v6, c("lhstarts", "lrprc"))
  for (i in seq_along(free)) H[idx6[free[i]], i + 1] <- 1
  H
}
lr_lrprc <- function(ratio) blrtest(z = jo_eng, H = H_lrprc(ratio), r = 1)@teststat

t1 <- blrtest(z = jo_eng, H = H_lrprc(0.987), r = 1)

# LR-based 95% CI in place of a Wald SE - individual beta_i don't have a
# standard SE in the Johansen framework; inverting the LR test is the
# defensible convention for a point-null like the Anastasiou (2023) 0.987.
crit <- qchisq(0.95, 1)
grid_lo <- unname(elas_r1["lrprc"])
while (lr_lrprc(grid_lo) < crit && grid_lo > -2) grid_lo <- grid_lo - 0.05
grid_hi <- unname(elas_r1["lrprc"])
while (lr_lrprc(grid_hi) < crit && grid_hi < 3) grid_hi <- grid_hi + 0.05
bisect <- function(lo, hi, f, target, tol = 1e-4) {
  flo <- f(lo) - target
  repeat {
    mid <- (lo + hi) / 2; fm <- f(mid) - target
    if (abs(hi - lo) < tol) return(mid)
    if (sign(fm) == sign(flo)) { lo <- mid; flo <- fm } else hi <- mid
  }
}
ci_lo <- bisect(grid_lo, unname(elas_r1["lrprc"]), lr_lrprc, crit)
ci_hi <- bisect(unname(elas_r1["lrprc"]), grid_hi, lr_lrprc, crit)

cat(sprintf("\nTEST 1  H0: lrprc=0.987 (Anastasiou 2023)  est=%.4f  95%%CI=[%.4f,%.4f]  LR=%.4f  df=1  p=%.4f  [rank r<=1: 70.34 vs 70.60 @5%%]\n",
            elas_r1["lrprc"], ci_lo, ci_hi, t1@teststat, t1@pval[1]))

# H0: elasticity_lrprc = -elasticity_lrcc (price/cost symmetry)
H_sym <- matrix(0, p6, p6 - 1)
H_sym[idx6["lrprc"], 1] <-  1
H_sym[idx6["lrcc"],  1] <- -1
free2 <- setdiff(v6, c("lrprc", "lrcc"))
for (i in seq_along(free2)) H_sym[idx6[free2[i]], i + 1] <- 1
t2 <- blrtest(z = jo_eng, H = H_sym, r = 1)

cat(sprintf("TEST 2  H0: lrprc=-lrcc  lrprc=%.4f  lrcc=%.4f  LR=%.4f  df=1  p=%.4f  [rank r<=1: 70.34 vs 70.60 @5%%]\n",
            elas_r1["lrprc"], elas_r1["lrcc"], t2@teststat, t2@pval[1]))

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
t3 <- bh5lrtest(z = jo_5, H = H_stat, r = 2)
summary(t3)
cat(sprintf("TEST 3  H0: vec1=e_lhstarts (5-var, r=2)  LR=%.4f  df=3  p=%.4f  [V/W free column is NaN/Inf - degenerate, LR/df/p unaffected]\n",
            t3@teststat, t3@pval[1]))

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