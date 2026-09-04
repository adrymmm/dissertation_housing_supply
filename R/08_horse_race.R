library(urca)
library(vars)
library(zoo)
library(ARDL)
library(tsibble)
library(fable)
library(fabletools)
library(dplyr)
library(tidyr)

eng_zoo    <- readRDS("R/models/nardl_eng_zoo.rds")
jo_eng     <- readRDS("R/models/jo_eng.rds")
ardl_best  <- readRDS("R/models/ardl_best.rds")
nardl_fits <- readRDS("R/models/nardl_fits_full.rds")
if (!file.exists("R/models/mrf_h1.rds")) stop("run R/07_MacroRF.R first")
mrf_h1 <- readRDS("R/models/mrf_h1.rds")

dat <- as.data.frame(eng_zoo)
dts <- as.yearqtr(as.numeric(zoo::index(eng_zoo)))
n   <- nrow(dat)

# CONFIG
y_name    <- "lhstarts"
K         <- 5L
R_RANK    <- 1L
nardl_var <- "lrcc"
dum       <- c("d08Q3", "d20Q2", "d20Q3", "d23Q2", "d23Q3")
seas      <- c("sd1", "sd2", "sd3")

# Frozen spec: ardl_best$order, NARDL order/variable, and VECM K/rank were all
# picked on the full 1975-2025 sample (saw the eval window) -- disclosure-only,
# not real-time OOS. See R/08b_horse_race_rt_check.R for a real-time check.

stopifnot(all(c(y_name, dum, seas) %in% names(dat)))
yv <- dat[[y_name]]

# Expanding evaluation window
eval0 <- which(dts == as.yearqtr("2010 Q1"))
stopifnot(length(eval0) == 1)
origin_set <- (eval0 - 1):(n - 1)

cat(sprintf("Origins %d..%d  ->  targets %s .. %s\n",
            min(origin_set), max(origin_set),
            as.character(dts[min(origin_set) + 1]),
            as.character(dts[max(origin_set) + 1])))


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

cat("ARDL : ", paste(ardl_x,  collapse = " + "), " | ",
    paste(ord_ardl,  collapse = ","), "\n", sep = "")
cat("NARDL: ", paste(nardl_x, collapse = " + "), " | ",
    paste(ord_nardl, collapse = ","), "\n", sep = "")

# ---- ARDL-family one-step forecasts ------------------------------------
lagv <- function(x, k) if (k == 0L) x else c(rep(NA_real_, k), head(x, -k))

# contemp = TRUE -> conditional/ex post (internal design check only);
# contemp = FALSE -> direct h=1 reparam (unused here, kept as a generic switch).
design <- function(dat, y, xs, ord, fixed, contemp = TRUE) {
  stopifnot(length(ord) == 1L + length(xs))
  cl <- list()
  for (k in seq_len(ord[1])) cl[[sprintf("L(%s, %d)", y, k)]] <- lagv(dat[[y]], k)
  k0 <- if (contemp) 0L else 1L
  for (j in seq_along(xs)) {
    if (ord[j + 1L] >= k0)                       # guard: k0:0 would give c(1,0)
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

# RW closure
project_rw <- function(dat, xs, o) {
  for (x in xs) dat[[x]][o + 1L] <- dat[[x]][o]
  dat
}

fc1_rw <- function(dat, y, xs, ord, fixed, yv, o) {
  fit_fc1(design(project_rw(dat, xs, o), y, xs, ord, fixed, contemp = TRUE),
          yv, o)$fc
}

# K/rank fixed to full-sample choices (03_vecm_core.R); johansen_rank() only
# fires via vecm_fit1()'s NULL-r_use fallback, used by 08b's real-time re-selection.
# Sequential trace test (r = 0, 1, 2, ... stop at first non-rejection); parses
# urca's row labels rather than assuming their order.
johansen_rank <- function(jo, kmax) {
  r_of <- as.integer(sub("^.*r *(?:<=|=) *([0-9]+).*$", "\\1", rownames(jo@cval)))
  stopifnot(!any(is.na(r_of)))
  for (i in order(r_of)) {
    if (jo@teststat[i] <= jo@cval[i, "5pct"])
      return(min(max(r_of[i], 1L), kmax))
  }
  kmax
}

# X_ardl_cond is not a reported forecast -- kept only as an internal check,
# validated directly against ardl_best's own coefficients (the rw design can't be).
X_ardl_cond <- design(dat, y_name, ardl_x, ord_ardl, c(dum, seas), contemp = TRUE)

# Validation: manual conditional design must reproduce ardl_best coefs
local({
  ok <- stats::complete.cases(X_ardl_cond) & !is.na(yv)
  b  <- qr.coef(qr(X_ardl_cond[ok, , drop = FALSE]), yv[ok])
  names(b) <- sub("^L\\((.*), 0\\)$", "\\1", names(b))
  ref <- coef(ardl_best)
  sh  <- intersect(names(b), names(ref))
  cat(sprintf("design check: %d/%d terms matched, max |diff| = %.2e\n",
              length(sh), length(ref), max(abs(b[sh] - ref[sh]))))
  if (length(sh) != length(ref) || max(abs(b[sh] - ref[sh])) > 1e-8)
    warning("manual ARDL design does not reproduce ardl_best -- check `design()`")
})

# ---- information-set assertion -----------------------------------------
local({
  o  <- max(origin_set) - 4L
  xp <- intersect(ardl_x, nardl_x)[1]
  stopifnot(!is.na(xp))
  d2 <- dat; d2[[xp]][o + 1L] <- d2[[xp]][o + 1L] + 1

  probe <- function(dd) c(
    # Validates the perturbation is non-vacuous (not itself a reported forecast) --
    # without it, a no-op perturbation bug would pass the leakage checks below trivially.
    cond_a = fit_fc1(design(dd, y_name, ardl_x,  ord_ardl,  c(dum, seas), TRUE),  yv, o)$fc,
    rw_a   = fc1_rw(dd, y_name, ardl_x,  ord_ardl,  c(dum, seas), yv, o),
    rw_n   = fc1_rw(dd, y_name, nardl_x, ord_nardl, c(dum, seas), yv, o)
  )
  d <- probe(d2) - probe(dat)
  cat(sprintf("leak probe on %s at target: cond %+.4f (validity check, not reported) | rw %.2e/%.2e\n",
              xp, d[["cond_a"]], d[["rw_a"]], d[["rw_n"]]))
  stopifnot(abs(d[["cond_a"]]) > 1e-6)                     # perturbation not vacuous
  stopifnot(all(abs(d[c("rw_a", "rw_n")]) < 1e-10))        # rw designs see no leakage
})

# ---- VECM one-step forecast --------------------------------------------
Xv <- as.matrix(dat[, vecm_vars])
Dv <- as.matrix(dat[, dum])

vecm_fit1 <- function(o, K_use, r_use = NULL) {
  keep <- colnames(Dv)[colSums(abs(Dv[1:o, , drop = FALSE])) > 0]
  jo <- ca.jo(Xv[1:o, , drop = FALSE], type = "trace", ecdet = "none", K = K_use,
              spec = "transitory", season = 4,
              dumvar = Dv[1:o, keep, drop = FALSE])

  # @dumvar holds impulse dummies only (seasonals are rebuilt separately by
  # predict.vec2var); all are zero at the target -- length check guards against a silent urca rename.
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

# ---- unconditional benchmarks ------------------------------------------
eng_tsbl <- tibble(Date = yearquarter(as.Date(dts)), lhstarts = yv) %>%
  as_tsibble(index = Date)

uncond_fc <- eng_tsbl %>%
  filter(row_number() <= max(origin_set)) %>%
  stretch_tsibble(.init = min(origin_set), .step = 1, .id = "origin_id") %>%
  model(rw     = RW(lhstarts),
        snaive = SNAIVE(lhstarts ~ lag("year")),
        ar     = AR(lhstarts ~ order(p = 0:8), ic = "aic"),
        tslm   = TSLM(lhstarts ~ trend()),
        tslm_s = TSLM(lhstarts ~ trend() + season())) %>%
  fabletools::forecast(h = 1) %>%
  as_tibble() %>%
  mutate(origin = min(origin_set) - 1L + origin_id) %>%
  dplyr::select(origin, model = .model, fc = .mean) %>%
  pivot_wider(names_from = model, values_from = fc)

# ---- ARDL-family and VECM (frozen, full-sample spec) --------------------
struct_fc <- do.call(rbind, lapply(origin_set, function(o) {
  fz <- vecm_fit1(o, K, R_RANK)
  row <- data.frame(
    origin   = o,
    ardl_rw  = fc1_rw(dat, y_name, ardl_x,  ord_ardl,  c(dum, seas), yv, o),
    nardl_rw = fc1_rw(dat, y_name, nardl_x, ord_nardl, c(dum, seas), yv, o),
    vecm     = fz$fc,
    trace_r1 = fz$trace_r1, cv5_r1 = fz$cv5_r1, n_dum = fz$n_dum
  )
  cat(sprintf("  origin %d (%s) done\n", o, as.character(dts[o + 1])))
  row
}))

# ---- assemble ----------------------------------------------------------
BENCH  <- c("rw", "snaive", "ar", "tslm", "tslm_s")
# Frozen: orders/K/rank chosen once on the full sample incl. eval window -- disclosure-only, not real-time OOS.
STRUCT <- c("vecm", "ardl_rw", "nardl_rw")
# MRF: spec fixed a priori (no selection on the evaluation window) and refit at
# every origin, so unlike STRUCT these are genuine real-time forecasts.
MRF_COLS <- c("arrf", "gtvp")
mcols  <- c(BENCH, STRUCT, MRF_COLS)

stopifnot(setequal(mrf_h1$origin, origin_set))

res <- full_join(uncond_fc, struct_fc, by = "origin") %>%
  full_join(mrf_h1, by = "origin") %>%
  mutate(target = origin + 1,
         actual = yv[target],
         date = as.character(as.Date(dts[target])),
         dummy_target = rowSums(Dv[target, , drop = FALSE]) > 0) %>%
  arrange(target) %>%
  dplyr::select(target, date, actual, all_of(mcols), dummy_target)

stopifnot(!any(is.na(res[, mcols])))

cat(sprintf("\nr<=1 rejected at 5%% in %d of %d origins\n",
            sum(struct_fc$trace_r1 > struct_fc$cv5_r1, na.rm = TRUE),
            nrow(struct_fc)))

cat("\nRMSE (full window):\n")
print(round(sqrt(colMeans((res[, mcols] - res$actual)^2)), 4))
cat("\nRMSE (ex dummy targets):\n")
print(round(sqrt(colMeans((res[!res$dummy_target, mcols] -
                             res$actual[!res$dummy_target])^2)), 4))

dir.create("data/outputs/forecasts", recursive = TRUE, showWarnings = FALSE)
write.csv(res, "data/outputs/forecasts/h1_forecasts.csv", row.names = FALSE)
write.csv(
  data.frame(
    model = mcols,
    role  = ifelse(mcols %in% STRUCT, "frozen_spec", "model_set"),
    benchmark = mcols %in% "rw"
  ),
  "data/outputs/forecasts/h1_model_roles.csv", row.names = FALSE
)

write.csv(struct_fc %>% mutate(date = as.character(as.Date(dts[origin + 1]))),
          "data/outputs/forecasts/h1_vecm_rank_trace.csv", row.names = FALSE)
