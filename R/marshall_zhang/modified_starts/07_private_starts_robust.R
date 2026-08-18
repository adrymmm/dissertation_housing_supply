# rdrobust robustness check for the private-starts redirect
# Standalone: does not modify 06_private_starts.R or the validated baseline.
# Estimates the same fuzzy design via CCT (Calonico-Cattaneo-Titiunik) local
# polynomial methods with data-driven bandwidths and bias-corrected inference,
# as an external check on the rddtools/my_fuzzy_rd + IK + HC0 pipeline.

library(dplyr); library(rdrobust)

options(width = 200)

load("../original/working/rdata/dat_1920.Rdata")

CUT <- 50

# NOTE: per_1000_private_starts_01 is an exact linear rescale of
# per_1000_private_starts (cor = 1, factor 1/23.06), so coefficients and SEs
# scale by that factor while t-stats and p-values are identical. There is no
# per_1000_sr_01, so the social-rent baseline can only be run unscaled; both
# private-starts scales are reported so the baseline comparison is like-for-like.
outcomes <- list(
  list(lab = "social rent (baseline)",      y = dat_1920$per_1000_sr),
  list(lab = "private starts (unscaled)",   y = dat_1920$per_1000_private_starts),
  list(lab = "private starts (_01 scaled)", y = dat_1920$per_1000_private_starts_01)
)

# pull conventional / bias-corrected / robust rows out of an rdrobust fit
extract <- function(o, lab, spec, bw_lab) {
  tibble(
    outcome   = lab,
    spec      = spec,
    bwsel     = bw_lab,
    h         = round(o$bws["h", "left"], 2),
    b         = round(o$bws["b", "left"], 2),
    n         = sum(o$N_h),
    convD     = round(o$coef["Conventional", 1], 3),
    convSE    = round(o$se["Conventional", 1], 3),
    convP     = round(o$pv["Conventional", 1], 3),
    bcD       = round(o$coef["Bias-Corrected", 1], 3),
    robSE     = round(o$se["Robust", 1], 3),
    robP      = round(o$pv["Robust", 1], 3)
  )
}

fit <- function(y, lab, spec, kernel, p, ...) {
  o <- rdrobust(y = y, x = dat_1920$afford_gap_median, c = CUT,
                fuzzy = dat_1920$funded_binary,
                kernel = kernel, p = p, ...)
  extract(o, lab, spec, if (is.null(list(...)$h)) o$bwselect else "manual")
}

## 1. rdrobust defaults: triangular kernel, local linear, MSE-optimal bandwidth --
cat("\n=== A. rdrobust defaults (triangular kernel, p = 1, bwselect = mserd) ===\n")
tab_default <- bind_rows(lapply(outcomes, function(o)
  fit(o$y, o$lab, "triangular, p=1", "triangular", 1, bwselect = "mserd")))
print(as.data.frame(tab_default), row.names = FALSE)

## 2. kernel/order matched to the my_fuzzy_rd pipeline (uniform, quadratic) -----
# bridges rdrobust to the 06 pipeline, which uses a uniform kernel and x + x^2
cat("\n=== B. matched to 06 pipeline (uniform kernel, p = 2, bwselect = mserd) ===\n")
tab_matched <- bind_rows(lapply(outcomes, function(o)
  fit(o$y, o$lab, "uniform, p=2", "uniform", 2, bwselect = "mserd")))
print(as.data.frame(tab_matched), row.names = FALSE)

## 3. alternative bandwidth selectors -------------------------------------------
cat("\n=== C. alternative rdrobust bandwidth selectors (triangular, p = 1) ===\n")
tab_bwsel <- bind_rows(lapply(c("mserd", "msetwo", "cerrd"), function(bs)
  bind_rows(lapply(outcomes, function(o)
    fit(o$y, o$lab, "triangular, p=1", "triangular", 1, bwselect = bs)))))
print(as.data.frame(tab_bwsel), row.names = FALSE)

## 4. bandwidth sensitivity grid ------------------------------------------------
# like-for-like with the manual IK bandwidth scan; h fixed, b left to rdrobust
cat("\n=== D. bandwidth grid (h fixed, triangular kernel, p = 1) ===\n")
grid <- c(20, 25, 30, 35, 40, 50)
tab_grid <- bind_rows(lapply(grid, function(hh)
  bind_rows(lapply(outcomes[1:2], function(o)
    fit(o$y, o$lab, "triangular, p=1", "triangular", 1, h = hh, b = hh)))))
print(as.data.frame(tab_grid %>% arrange(outcome, h)), row.names = FALSE)

cat("\n=== E. bandwidth grid, matched kernel/order (uniform, p = 2) ===\n")
tab_grid2 <- bind_rows(lapply(grid, function(hh)
  bind_rows(lapply(outcomes[1:2], function(o)
    fit(o$y, o$lab, "uniform, p=2", "uniform", 2, h = hh, b = hh)))))
print(as.data.frame(tab_grid2 %>% arrange(outcome, h)), row.names = FALSE)

## first stage -----------------------------------------------------------------
# the fuzzy estimate is the reduced form divided by this jump; a small first
# stage inflates both the LATE and its standard error
cat("
=== F. first stage: jump in funded_binary at the cutoff ===
")
fs_rows <- bind_rows(
  lapply(list(NULL, 20, 25, 30, 35, 40, 50), function(hh) {
    o <- if (is.null(hh))
      rdrobust(y = dat_1920$funded_binary, x = dat_1920$afford_gap_median,
               c = CUT, kernel = "triangular", p = 1, bwselect = "mserd")
    else
      rdrobust(y = dat_1920$funded_binary, x = dat_1920$afford_gap_median,
               c = CUT, kernel = "triangular", p = 1, h = hh, b = hh)
    tibble(h = round(o$bws["h", "left"], 2),
           source = if (is.null(hh)) "mserd" else "manual",
           n = sum(o$N_h),
           jump = round(o$coef["Conventional", 1], 4),
           se = round(o$se["Conventional", 1], 4),
           p = round(o$pv["Conventional", 1], 4),
           t = round(o$coef["Conventional", 1] / o$se["Conventional", 1], 2))
  }))
print(as.data.frame(fs_rows), row.names = FALSE)
