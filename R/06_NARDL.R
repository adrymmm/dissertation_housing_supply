# =============================================================================
# NARDL — Nonlinear ARDL (Shin, Yu & Greenwood-Nimmo, 2014)
# Dependent variable: lhstarts
# Asymmetric variable: lrprc (decomposed into positive/negative partial sums)
# =============================================================================

library(dynlm)
library(lmtest)
library(sandwich)
library(car)

# -----------------------------------------------------------------------------
# 1. Partial sum decomposition of lrprc
# -----------------------------------------------------------------------------
make_partial_sums <- function(tf) {
  d_lrprc   <- diff(tf[, "lrprc"])
  lrprc_pos <- ts(cumsum(pmax(d_lrprc, 0)),
                  start = start(d_lrprc), frequency = 4)
  lrprc_neg <- ts(cumsum(pmin(d_lrprc, 0)),
                  start = start(d_lrprc), frequency = 4)
  list(pos = lrprc_pos, neg = lrprc_neg)
}

ps_uk  <- make_partial_sums(uk_tf)
ps_eng <- make_partial_sums(eng_tf)

# -----------------------------------------------------------------------------
# 2. Build data frames for dynlm
# -----------------------------------------------------------------------------
make_nardl_df <- function(tf, ps) {
  qq <- cycle(tf)
  df <- ts(cbind(
    lhstarts  = tf[,  "lhstarts"],
    lrprc_pos = c(NA, ps$pos),     # lag-align: diff loses first obs
    lrprc_neg = c(NA, ps$neg),
    lvol      = tf[,  "lvol"],
    r3        = tf[,  "r3"],
    lstock    = tf[,  "lstock"],
    lrcc      = tf[,  "lrcc"],
    d1        = as.numeric(qq == 1),
    d2        = as.numeric(qq == 2),
    d3        = as.numeric(qq == 3)
  ), start = start(tf), frequency = 4)
  df
}

nardl_uk_df  <- make_nardl_df(uk_tf,  ps_uk)
nardl_eng_df <- make_nardl_df(eng_tf, ps_eng)

# -----------------------------------------------------------------------------
# 3. Estimate NARDL — use same lag orders as ARDL best models
#    UK:      ARDL(4, -, 4, 0, 1, 4) — lrprc split into pos/neg, same lags
#    England: ARDL(4, -, 3, 0, 5, 2)
# -----------------------------------------------------------------------------
nardl_uk <- dynlm(
  d(lhstarts) ~
    L(lhstarts, 1) + L(lrprc_pos, 1) + L(lrprc_neg, 1) +
    L(lvol, 1) + L(r3, 1) + L(lstock, 1) + L(lrcc, 1) +
    d(L(lhstarts, 1)) + d(L(lhstarts, 2)) + d(L(lhstarts, 3)) +
    d(lrprc_pos) + d(lrprc_neg) +
    d(lvol) + d(L(lvol, 1)) + d(L(lvol, 2)) + d(L(lvol, 3)) +
    L(r3, 0) +
    d(lstock) +
    d(lrcc) + d(L(lrcc, 1)) + d(L(lrcc, 2)) + d(L(lrcc, 3)) +
    d1 + d2 + d3,
  data = nardl_uk_df
)

nardl_eng <- dynlm(
  d(lhstarts) ~
    L(lhstarts, 1) + L(lrprc_pos, 1) + L(lrprc_neg, 1) +
    L(lvol, 1) + L(r3, 1) + L(lstock, 1) + L(lrcc, 1) +
    d(L(lhstarts, 1)) + d(L(lhstarts, 2)) + d(L(lhstarts, 3)) +
    d(lrprc_pos) + d(lrprc_neg) +
    d(lvol) + d(L(lvol, 1)) + d(L(lvol, 2)) +
    d(lstock) + d(L(lstock, 1)) + d(L(lstock, 2)) +
    d(L(lstock, 3)) + d(L(lstock, 4)) +
    d(lrcc) + d(L(lrcc, 1)) +
    d1 + d2 + d3,
  data = nardl_eng_df
)

# -----------------------------------------------------------------------------
# 4. Long-run asymmetric coefficients
#    L+ = -coef(lrprc_pos.l1) / coef(lhstarts.l1)
#    L- = -coef(lrprc_neg.l1) / coef(lhstarts.l1)
# -----------------------------------------------------------------------------
lr_coefs <- function(mod, label) {
  b  <- coef(mod)
  rho    <- b["L(lhstarts, 1)"]
  L_pos  <- -b["L(lrprc_pos, 1)"] / rho
  L_neg  <- -b["L(lrprc_neg, 1)"] / rho
  cat(label, "— long-run: L+ =", round(L_pos, 4),
      " L- =", round(L_neg, 4),
      " diff =", round(L_pos - L_neg, 4), "\n")
}

lr_coefs(nardl_uk,  "UK")
lr_coefs(nardl_eng, "England")

# -----------------------------------------------------------------------------
# 5. Wald test for long-run asymmetry: H0: L+ = L-
#    Equivalent to: coef(lrprc_pos.l1) = coef(lrprc_neg.l1)
# -----------------------------------------------------------------------------
cat("\n=== Wald test — long-run asymmetry ===\n")
cat("UK:\n")
print(linearHypothesis(nardl_uk,
                       "L(lrprc_pos, 1) = L(lrprc_neg, 1)",
                       vcov = vcovHAC(nardl_uk)))

cat("\nEngland:\n")
print(linearHypothesis(nardl_eng,
                       "L(lrprc_pos, 1) = L(lrprc_neg, 1)",
                       vcov = vcovHAC(nardl_eng)))


# Short run asymemtry
cat("UK short-run asymmetry:\n")
print(linearHypothesis(nardl_uk,
                       "d(lrprc_pos) = d(lrprc_neg)",
                       vcov = vcovHAC(nardl_uk)))

cat("\nEngland short-run asymmetry:\n")
print(linearHypothesis(nardl_eng,
                       "d(lrprc_pos) = d(lrprc_neg)",
                       vcov = vcovHAC(nardl_eng)))
