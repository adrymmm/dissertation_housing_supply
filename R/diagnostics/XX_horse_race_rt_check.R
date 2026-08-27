# Real-time specification re-selection: a sensitivity check, NOT part of the
# headline h=1 horse race (08_horse_race.R). 08_horse_race.R's structural
# forecasts use lag orders / K / rank chosen once on the full 1975-2025
# sample -- i.e. a frozen spec that saw the evaluation window. This script
# re-selects those choices at each origin using only data up to that origin,
# to quantify how much of the frozen spec's apparent edge is specification
# leakage.
#
# Standalone by design: duplicates the setup and frozen-forecast machinery
# from 08_horse_race.R rather than sourcing it, so it can be run
# independently of (and without re-running or overwriting the output of) the
# headline pipeline. Writes its own output file so it never collides with
# h1_forecasts.csv / h1_model_roles.csv.

library(urca)
library(vars)
library(zoo)
library(ARDL)

eng_zoo    <- readRDS("R/models/nardl_eng_zoo.rds")
jo_eng     <- readRDS("R/models/jo_eng.rds")
ardl_best  <- readRDS("R/models/ardl_best.rds")
nardl_fits <- readRDS("R/models/nardl_fits_full.rds")

dat <- as.data.frame(eng_zoo)
dts <- as.yearqtr(as.numeric(zoo::index(eng_zoo)))
n   <- nrow(dat)

# CONFIG (mirrors 08_horse_race.R)
y_name    <- "lhstarts"
K         <- 5L
R_RANK    <- 1L
nardl_var <- "lrcc"
dum       <- c("d08Q3", "d20Q2", "d20Q3", "d23Q2", "d23Q3")
seas      <- c("sd1", "sd2", "sd3")

# Real-time-selection-only config
MAX_ORDER_ARDL <- 6L
MAX_LAG_NARDL  <- 8L
GTS_THRESH     <- 0.10
base_x         <- c("lrprc", "lvol", "r3", "lrcc")

stopifnot(all(c(y_name, dum, seas) %in% names(dat)))
yv <- dat[[y_name]]

# Expanding evaluation window
eval0 <- which(dts == as.yearqtr("2010 Q1"))
stopifnot(length(eval0) == 1)
origin_set <- (eval0 - 1):(n - 1)

# VECM: six variables in jo_eng column order
vecm_vars <- colnames(jo_eng@x)
stopifnot(all(vecm_vars %in% names(dat)), vecm_vars[1] == y_name)

# ARDL: lstock is dropped in 05, so take the regressors from $order's names
ord_ardl <- as.integer(ardl_best$order)
ardl_x   <- names(ardl_best$order)[-1]
stopifnot(!is.null(names(ardl_best$order)),
          names(ardl_best$order)[1] == y_name,
          all(ardl_x %in% names(dat)))

# NARDL: pull the gts-selected fit
as_ardl <- function(x) {
  if (inherits(x, "ardl")) return(x)
  if (!is.null(x$fit) && inherits(x$fit, "ardl")) return(x$fit)
  hit <- Filter(function(e) inherits(e, "ardl"), x)
  if (!length(hit)) stop("no ardl object inside nardl_fits_full[['", nardl_var, "']]")
  hit[[1]]
}
nardl_fit <- as_ardl(nardl_fits[[nardl_var]])
ord_nardl <- as.integer(nardl_fit$order)
nardl_x   <- names(nardl_fit$order)[-1]
stopifnot(names(nardl_fit$order)[1] == y_name, all(nardl_x %in% names(dat)))

# ---- ARDL-family one-step forecasts ------------------------------------
lagv <- function(x, k) if (k == 0L) x else c(rep(NA_real_, k), head(x, -k))

design <- function(dat, y, xs, ord, fixed, contemp = TRUE) {
  stopifnot(length(ord) == 1L + length(xs))
  cl <- list()
  for (k in seq_len(ord[1])) cl[[sprintf("L(%s, %d)", y, k)]] <- lagv(dat[[y]], k)
  k0 <- if (contemp) 0L else 1L
  for (j in seq_along(xs)) {
    if (ord[j + 1L] >= k0)
      for (k in k0:ord[j + 1L])
        cl[[sprintf("L(%s, %d)", xs[j], k)]] <- lagv(dat[[xs[j]]], k)
  }
  for (f in fixed) cl[[f]] <- dat[[f]]
  cbind(`(Intercept)` = 1, as.matrix(as.data.frame(cl, check.names = FALSE)))
}

fit_fc1 <- function(X, yv, o) {
  vary <- apply(X[1:o, , drop = FALSE], 2,
                function(z) length(unique(z[!is.na(z)])) > 1L)
  Xi <- X[, union("(Intercept)", colnames(X)[vary]), drop = FALSE]
  ok <- stats::complete.cases(Xi[1:o, , drop = FALSE]) & !is.na(yv[1:o])
  qf <- qr(Xi[1:o, , drop = FALSE][ok, , drop = FALSE])
  if (qf$rank < ncol(Xi))
    warning(sprintf("rank deficiency at origin %d (%d of %d)",
                    o, qf$rank, ncol(Xi)))
  b <- qr.coef(qf, yv[1:o][ok])
  b[is.na(b)] <- 0
  list(fc = sum(Xi[o + 1L, ] * b), k = ncol(Xi))
}

project_rw <- function(dat, xs, o) {
  for (x in xs) dat[[x]][o + 1L] <- dat[[x]][o]
  dat
}

fc1_rw <- function(dat, y, xs, ord, fixed, yv, o) {
  fit_fc1(design(project_rw(dat, xs, o), y, xs, ord, fixed, contemp = TRUE),
          yv, o)$fc
}

# ---- real-time specification selection -----------------------------------
# All of this reads rows 1..o only. Candidate designs are column subsets of one
# pre-built lag bank, so a full AIC grid costs a few seconds per origin.

nardl_pool <- unique(c(base_x, paste0(rep(base_x, each = 2), c("_pos", "_neg"))))
bank_vars  <- unique(c(y_name, nardl_pool))
MAX_K      <- max(MAX_ORDER_ARDL, MAX_LAG_NARDL)

BANK <- local({
  cl <- list()
  for (v in bank_vars) for (k in 0:MAX_K)
    cl[[sprintf("L(%s, %d)", v, k)]] <- lagv(dat[[v]], k)
  cbind(`(Intercept)` = 1,
        as.matrix(as.data.frame(cl, check.names = FALSE)),
        as.matrix(dat[, c(dum, seas)]))
})
FIX_COLS  <- match(c(dum, seas), colnames(BANK))
EST_START <- MAX_K + 1L          # common sample, so AICs are comparable

cols_for <- function(y, xs, ord, contemp) {
  k0 <- if (contemp) 0L else 1L
  c(sprintf("L(%s, %d)", y, seq_len(ord[1])),
    unlist(lapply(seq_along(xs), function(j)
      if (ord[j + 1L] >= k0)
        sprintf("L(%s, %d)", xs[j], k0:ord[j + 1L]) else character(0))))
}

# Columns of BANK with variation in the window -- drops impulse dummies for
# events that haven't happened yet at this origin.
usable_fix <- function(o) {
  FIX_COLS[apply(BANK[EST_START:o, FIX_COLS, drop = FALSE], 2,
                 function(z) length(unique(z)) > 1L)]
}

rt_window <- function(idx, o) {
  X <- BANK[EST_START:o, idx, drop = FALSE]
  y <- yv[EST_START:o]
  ok <- !is.na(y)
  list(X = X[ok, , drop = FALSE], y = y[ok])
}

# AIC only -- the hot path, run once per grid point.
rt_aic <- function(idx, o) {
  w <- rt_window(idx, o)
  f <- .lm.fit(w$X, w$y)
  if (f$rank < ncol(w$X)) return(Inf)              # rank deficient: reject
  n <- length(w$y)
  n * log(sum(f$residuals^2) / n) + 2 * f$rank
}

# AIC plus coefficient p-values, for the general-to-specific search.
rt_fit <- function(idx, o) {
  w <- rt_window(idx, o)
  qf <- qr(w$X)
  n <- length(w$y); p <- qf$rank
  if (p < ncol(w$X)) return(NULL)
  b <- qr.coef(qf, w$y)
  rss <- sum((w$y - drop(w$X %*% b))^2)
  se <- sqrt(diag(chol2inv(qr.R(qf))) * rss / (n - p))
  list(aic = n * log(rss / n) + 2 * p,
       pval = setNames(2 * stats::pt(-abs(b / se), n - p), colnames(w$X)))
}

# ARDL: AIC over the full order grid, the rule auto_ardl() applies by default.
ardl_grid <- as.matrix(expand.grid(c(list(seq_len(MAX_ORDER_ARDL)),
                                     rep(list(0:MAX_ORDER_ARDL), length(base_x)))))
select_ardl <- function(o, contemp) {
  fx <- usable_fix(o)
  best <- NULL; best_aic <- Inf
  for (r in seq_len(nrow(ardl_grid))) {
    ord <- ardl_grid[r, ]
    idx <- c(1L, match(cols_for(y_name, base_x, ord, contemp), colnames(BANK)), fx)
    a <- rt_aic(idx, o)
    if (a < best_aic) { best_aic <- a; best <- ord }
  }
  as.integer(best)
}

# NARDL: general-to-specific on the tail lags, the rule gts() applies, run for
# each candidate decomposition; the decomposed variable is then the one with
# the lowest AIC (the full-sample choice of lrcc was a judgement call, and a
# real-time analogue has to be mechanical).
gts_rt <- function(v, o, contemp) {
  xs  <- c(paste0(v, c("_pos", "_neg")), setdiff(base_x, v))
  fx  <- usable_fix(o)
  ord <- rep(MAX_LAG_NARDL, 1L + length(xs))
  repeat {
    idx <- c(1L, match(cols_for(y_name, xs, ord, contemp), colnames(BANK)), fx)
    f <- rt_fit(idx, o)
    # rank-deficient at this order: shrink the longest lag and retry
    if (is.null(f)) { ord[which.max(ord)] <- max(ord) - 1L; next }
    tails <- ifelse(c(ord[1] > 1L, ord[-1] > 0L),
                    sprintf("L(%s, %d)", c(y_name, xs), ord), NA_character_)
    keep <- !is.na(tails) & tails %in% names(f$pval)
    if (!any(keep)) return(list(order = ord, aic = f$aic, xs = xs))
    p <- f$pval[tails[keep]]
    if (max(p) <= GTS_THRESH) return(list(order = ord, aic = f$aic, xs = xs))
    ord[which(keep)[which.max(p)]] <- ord[which(keep)[which.max(p)]] - 1L
  }
}
select_nardl <- function(o, contemp) {
  cand <- lapply(base_x, gts_rt, o = o, contemp = contemp)
  cand[[which.min(vapply(cand, `[[`, 0, "aic"))]]
}

# One-step forecast from a spec chosen at this origin. `rw` substitutes
# x_{o+1} = x_o so the forecast stays on the origin's information set.
fc1_rt <- function(xs, ord, o, contemp, rw) {
  d <- if (rw) project_rw(dat, xs, o) else dat
  fit_fc1(design(d, y_name, xs, ord, c(dum, seas), contemp = contemp), yv, o)$fc
}

# VECM: K by FPE and rank by the trace test, both re-run at each origin
# (03_vecm_core.R fixed K = 5 and r = 1 on the full sample).
# Sequential trace test: walk r = 0, 1, 2, ... and stop at the first
# non-rejection. urca labels rows "r = 0  |" / "r <= k |" and lists them in
# descending k, so the r each row refers to is parsed rather than assumed.
johansen_rank <- function(jo, kmax) {
  r_of <- as.integer(sub("^.*r *(?:<=|=) *([0-9]+).*$", "\\1", rownames(jo@cval)))
  stopifnot(!any(is.na(r_of)))
  for (i in order(r_of)) {
    if (jo@teststat[i] <= jo@cval[i, "5pct"])
      return(min(max(r_of[i], 1L), kmax))
  }
  kmax
}

Xv <- as.matrix(dat[, vecm_vars])
Dv <- as.matrix(dat[, dum])

vecm_fit1 <- function(o, K_use, r_use = NULL) {
  keep <- colnames(Dv)[colSums(abs(Dv[1:o, , drop = FALSE])) > 0]
  jo <- ca.jo(Xv[1:o, , drop = FALSE], type = "trace", ecdet = "none", K = K_use,
              spec = "transitory", season = 4,
              dumvar = Dv[1:o, keep, drop = FALSE])

  dv <- tail(jo@dumvar, 4)[1, , drop = FALSE]
  hit <- intersect(colnames(dv), dum)
  stopifnot(ncol(dv) == ncol(jo@dumvar), length(hit) == length(keep))
  dv[, hit] <- 0

  if (is.null(r_use)) r_use <- johansen_rank(jo, ncol(Xv) - 1L)
  i1 <- grep("r <= 1", rownames(jo@cval))
  list(fc = predict(vec2var(jo, r = r_use), n.ahead = 1,
                    dumvar = dv)$fcst[[y_name]][1, "fcst"],
       r = r_use, K = K_use, n_dum = length(keep),
       trace_r1 = if (length(i1) == 1) jo@teststat[i1] else NA_real_,
       cv5_r1   = if (length(i1) == 1) jo@cval[i1, "5pct"] else NA_real_)
}

# K by FPE on rows 1..o
select_K <- function(o) {
  keep <- colnames(Dv)[colSums(abs(Dv[1:o, , drop = FALSE])) > 0]
  lag_max <- floor(12 * (o / 100)^(1 / 4))
  sel <- try(VARselect(Xv[1:o, , drop = FALSE], lag.max = lag_max, type = "const",
                       season = 4, exogen = Dv[1:o, keep, drop = FALSE]),
             silent = TRUE)
  if (inherits(sel, "try-error")) return(K)
  k_fpe <- as.integer(sel$selection[["FPE(n)"]])
  if (k_fpe < lag_max) return(k_fpe)
  k_hq <- as.integer(sel$selection[["HQ(n)"]])
  cat(sprintf("    NOTE origin %d: FPE at lag.max (%d), falling back to HQ (%d)\n",
              o, lag_max, k_hq))
  k_hq
}

# ---- frozen vs real-time, per origin -----------------------------------
struct_fc <- do.call(rbind, lapply(origin_set, function(o) {
  fz <- vecm_fit1(o, K, R_RANK)
  oa <- select_ardl(o, contemp = TRUE)
  na <- select_nardl(o, contemp = TRUE)
  rt <- vecm_fit1(o, select_K(o), NULL)
  row <- data.frame(
    origin         = o,
    # frozen: orders / K / rank chosen once on the full sample (see 08_horse_race.R)
    ardl_rw        = fc1_rw(dat, y_name, ardl_x,  ord_ardl,  c(dum, seas), yv, o),
    nardl_rw       = fc1_rw(dat, y_name, nardl_x, ord_nardl, c(dum, seas), yv, o),
    vecm           = fz$fc,
    # real-time: orders / K / rank re-selected using data up to this origin only
    ardl_rt        = fc1_rt(base_x, oa,       o, contemp = TRUE, rw = TRUE),
    nardl_rt       = fc1_rt(na$xs,  na$order, o, contemp = TRUE, rw = TRUE),
    vecm_rt        = rt$fc,
    rt_ardl_order  = paste(oa, collapse = ","),
    rt_nardl_var   = sub("_pos$", "", na$xs[1]),
    rt_nardl_order = paste(na$order, collapse = ","),
    rt_vecm_K      = rt$K,
    rt_vecm_r      = rt$r
  )
  cat(sprintf("  origin %d (%s) done\n", o, as.character(dts[o + 1])))
  row
}))

struct_fc$date   <- as.character(as.Date(dts[struct_fc$origin + 1]))
struct_fc$actual <- yv[struct_fc$origin + 1]

# ---- how much of the structural models' edge is specification leakage ----
rmse <- function(pred) sqrt(mean((pred - struct_fc$actual)^2))
cmp <- data.frame(
  frozen   = c("ardl_rw", "nardl_rw", "vecm"),
  realtime = c("ardl_rt", "nardl_rt", "vecm_rt"),
  stringsAsFactors = FALSE
)
cmp$rmse_frozen <- vapply(cmp$frozen,   function(m) rmse(struct_fc[[m]]), 0)
cmp$rmse_rt     <- vapply(cmp$realtime, function(m) rmse(struct_fc[[m]]), 0)
cmp$pct_worse   <- 100 * (cmp$rmse_rt / cmp$rmse_frozen - 1)
cat("\nSpecification leakage (full-sample order/K/rank vs re-selected per origin):\n")
print(cmp, row.names = FALSE, digits = 4)

cat("\nARDL orders selected in real time:\n");   print(table(struct_fc$rt_ardl_order))
cat("\nNARDL decomposed variable:\n");           print(table(struct_fc$rt_nardl_var))
cat("\nVECM K x rank:\n"); print(table(K = struct_fc$rt_vecm_K, r = struct_fc$rt_vecm_r))

dir.create("data/outputs/forecasts", recursive = TRUE, showWarnings = FALSE)
write.csv(struct_fc, "data/outputs/forecasts/h1_rt_sensitivity.csv", row.names = FALSE)
