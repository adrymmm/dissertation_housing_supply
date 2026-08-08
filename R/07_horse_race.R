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

dat <- as.data.frame(eng_zoo)
dts <- as.yearqtr(as.numeric(zoo::index(eng_zoo)))
n   <- nrow(dat)

# CONFIG
# K, R_RANK, ord_* fixed from full sample; coefficients re-estimated per origin
y_name    <- "lhstarts"
K         <- 5L
R_RANK    <- 1L
nardl_var <- "lrcc"
dum       <- c("d08Q3", "d20Q2", "d20Q3", "d23Q2", "d23Q3")
seas      <- c("sd1", "sd2", "sd3")

stopifnot(all(c(y_name, dum, seas) %in% names(dat)))
yv <- dat[[y_name]]

eval0 <- which(dts == as.yearqtr("2010 Q1"))
stopifnot(length(eval0) == 1)
origin_set <- (eval0 - 1):(n - 1)

cat(sprintf("Origins %d..%d  ->  targets %s .. %s\n",
            min(origin_set), max(origin_set),
            as.character(dts[min(origin_set) + 1]),
            as.character(dts[max(origin_set) + 1])))


# VECM: six variables in jo_eng's own column order (includes lstock).
vecm_vars <- colnames(jo_eng@x)
stopifnot(all(vecm_vars %in% names(dat)), vecm_vars[1] == y_name)

# ARDL: lstock is dropped in 05, so take the regressors from $order's names
ord_ardl <- as.integer(ardl_best$order)
ardl_x   <- names(ardl_best$order)[-1]
stopifnot(!is.null(names(ardl_best$order)),
          names(ardl_best$order)[1] == y_name,
          all(ardl_x %in% names(dat)))

# NARDL: pull the gts-selected fit out of whatever run_nardl() named it, then
# read both the order AND the regressor names off it. That way the positional
# c(x_pos, x_neg, setdiff(base, x)) convention in nardl_functions.R never has to
# be replicated here.
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

design <- function(dat, y, xs, ord, fixed) {
  stopifnot(length(ord) == 1L + length(xs))
  cl <- list()
  for (k in seq_len(ord[1])) cl[[sprintf("L(%s, %d)", y, k)]] <- lagv(dat[[y]], k)
  for (j in seq_along(xs)) for (k in 0:ord[j + 1L])
    cl[[sprintf("L(%s, %d)", xs[j], k)]] <- lagv(dat[[xs[j]]], k)
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

X_ardl  <- design(dat, y_name, ardl_x,  ord_ardl,  c(dum, seas))
X_nardl <- design(dat, y_name, nardl_x, ord_nardl, c(dum, seas))

# Validation: manual design must reproduce ardl_best coefs
local({
  f  <- fit_fc1(X_ardl, yv, n - 1L)  # coefficients only; the forecast is discarded
  ok <- stats::complete.cases(X_ardl) & !is.na(yv)
  b  <- qr.coef(qr(X_ardl[ok, , drop = FALSE]), yv[ok])
  names(b) <- sub("^L\\((.*), 0\\)$", "\\1", names(b))
  ref <- coef(ardl_best)
  sh  <- intersect(names(b), names(ref))
  cat(sprintf("design check: %d/%d terms matched, max |diff| = %.2e\n",
              length(sh), length(ref), max(abs(b[sh] - ref[sh]))))
  if (length(sh) != length(ref) || max(abs(b[sh] - ref[sh])) > 1e-8)
    warning("manual ARDL design does not reproduce ardl_best -- check `design()`")
})

# ---- VECM one-step forecast --------------------------------------------
Xv <- as.matrix(dat[, vecm_vars])
Dv <- as.matrix(dat[, dum])

vecm_fc1 <- function(o) {
  keep <- colnames(Dv)[colSums(abs(Dv[1:o, , drop = FALSE])) > 0]
  jo <- ca.jo(Xv[1:o, , drop = FALSE], type = "trace", ecdet = "none", K = K,
              spec = "transitory", season = 4,
              dumvar = Dv[1:o, keep, drop = FALSE])

  dv <- tail(jo@dumvar, 4)[1, , drop = FALSE]
  dv[, intersect(colnames(dv), dum)] <- 0
  i1 <- grep("r <= 1", rownames(jo@cval))
  data.frame(
    vecm     = predict(vec2var(jo, r = R_RANK), n.ahead = 1,
                       dumvar = dv)$fcst[[y_name]][1, "fcst"],
    trace_r1 = if (length(i1) == 1) jo@teststat[i1] else NA_real_,
    cv5_r1   = if (length(i1) == 1) jo@cval[i1, "5pct"] else NA_real_,
    n_dum    = length(keep)
  )
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

# ---- conditional models ------------------------------------------------
cond_fc <- do.call(rbind, lapply(origin_set, function(o) {
  cbind(data.frame(origin = o,
                   ardl  = fit_fc1(X_ardl,  yv, o)$fc,
                   nardl = fit_fc1(X_nardl, yv, o)$fc),
        vecm_fc1(o))
}))

# ---- assemble ----------------------------------------------------------
mcols <- c("rw", "snaive", "ar", "tslm", "tslm_s", "vecm", "ardl", "nardl")

res <- full_join(uncond_fc, cond_fc, by = "origin") %>%
  mutate(target = origin + 1,
         actual = yv[target],
         date = as.character(as.Date(dts[target])),
         dummy_target = rowSums(Dv[target, , drop = FALSE]) > 0) %>%
  arrange(target) %>%
  dplyr::select(target, date, actual, all_of(mcols), dummy_target)

stopifnot(!any(is.na(res[, mcols])))

cat(sprintf("\nr<=1 rejected at 5%% in %d of %d origins\n",
            sum(cond_fc$trace_r1 > cond_fc$cv5_r1, na.rm = TRUE),
            nrow(cond_fc)))

dir.create("data/outputs/forecasts", recursive = TRUE, showWarnings = FALSE)
write.csv(res, "data/outputs/forecasts/h1_forecasts.csv", row.names = FALSE)
write.csv(cond_fc %>% mutate(date = as.character(as.Date(dts[origin + 1]))),
          "data/outputs/forecasts/h1_vecm_rank_trace.csv", row.names = FALSE)

# NOTE ON THE CONDITIONING ASYMMETRY (unchanged from the previous version, still
# unresolved). ardl/nardl see realised covariates at o+1; vecm and the naives
# project everything. A DM test of ardl vs vecm therefore compares a conditional
# forecast to an unconditional one, not two forecasting procedures. Either label
# the ardl/nardl columns as scenario-conditional -- which is the bridge/OBR use
# case and defensible if stated -- or add an unconditional ARDL column driven by
# RW-projected covariates before running DM across the two groups.