library(zoo)
library(pracma)
library(parallel)
source("R/vendor/MacroRF/MRF_v210403.R")   # defines MRF()

B     <- 250L
QRATE <- 0.3
CORES <- max(1L, detectCores() - 2L)
SEED  <- 42L
VI    <- TRUE      # variable importance on the full-sample GTVP fit only

eng_zoo <- readRDS("R/models/nardl_eng_zoo.rds")
dat <- as.data.frame(eng_zoo)
dts <- as.yearqtr(as.numeric(zoo::index(eng_zoo)))
n   <- nrow(dat)

L <- function(x, k) c(rep(NA_real_, k), head(x, -k))
margin <- dat$lrprc - dat$lrcc

D <- data.frame(
  lhstarts  = dat$lhstarts,
  margin    = margin,
  r3        = dat$r3,
  y_l1      = L(dat$lhstarts, 1),
  y_l4      = L(dat$lhstarts, 4),
  margin_l1 = L(margin, 1),
  r3_l1     = L(dat$r3, 1),
  lrprc_l1  = L(dat$lrprc, 1),
  lrcc_l1   = L(dat$lrcc, 1),
  lvol_l1   = L(dat$lvol, 1),
  lstock_l1 = L(dat$lstock, 1),
  trend     = seq_len(n),
  qtr       = as.integer(round(4 * (as.numeric(dts) %% 1))) + 1L
)

S_COLS <- c("y_l1", "y_l4", "margin_l1", "r3_l1", "lrprc_l1", "lrcc_l1",
            "lvol_l1", "lstock_l1", "trend", "qtr")
X_ARRF <- c("y_l1", "y_l4")
X_GTVP <- c("margin_l1", "r3_l1")       # lagged: the h=1 forecast may not see t

first_ok <- max(which(!stats::complete.cases(D))) + 1L
eval0 <- which(dts == as.yearqtr("2010 Q1"))
stopifnot(length(eval0) == 1L)
origin_set <- (eval0 - 1):(n - 1)

cat(sprintf("MRF: B=%d, cores=%d, origins %d..%d -> targets %s .. %s\n",
            B, CORES, min(origin_set), max(origin_set),
            as.character(dts[min(origin_set) + 1]),
            as.character(dts[max(origin_set) + 1])))

fit_mrf <- function(rows, xcols, oos, vi = FALSE) {
  cols <- c("lhstarts", union(xcols, S_COLS))
  M <- as.matrix(D[rows, cols])
  MRF(data = M, y.pos = 1L,
      x.pos = match(xcols, cols), S.pos = match(S_COLS, cols),
      oos.pos = oos, B = B, quantile.rate = QRATE, VI = vi,
      resampling.opt = 2, block.size = 12,
      trend.pos = match("trend", cols), trend.push = 4,
      printb = FALSE, cheap.look.at.GTVPs = FALSE)
}

# h = 1, recursive: refit on first_ok..o, predict o+1. Matches the scheme the
# ARDL/NARDL/VECM columns of the horse race use.
fc1 <- function(o, xcols) {
  rows <- first_ok:(o + 1L)
  fit_mrf(rows, xcols, oos = length(rows))$pred
}

run_spec <- function(xcols) {
  fc <- unlist(mclapply(origin_set, fc1, xcols = xcols,
                        mc.cores = CORES, mc.set.seed = TRUE))
  stopifnot(is.numeric(fc), length(fc) == length(origin_set), !any(is.na(fc)))
  fc
}

RNGkind("L'Ecuyer-CMRG")
set.seed(SEED)
t0 <- Sys.time()
mrf_h1 <- data.frame(origin = origin_set,
                     arrf = run_spec(X_ARRF),
                     gtvp = run_spec(X_GTVP))
cat(sprintf("recursive fits done in %.1f min\n",
            as.numeric(Sys.time() - t0, units = "mins")))

actual <- dat$lhstarts[origin_set + 1L]
cat("\nh=1 RMSE:\n")
print(round(sqrt(colMeans((mrf_h1[, c("arrf", "gtvp")] - actual)^2)), 4))

dir.create("R/models", recursive = TRUE, showWarnings = FALSE)
saveRDS(mrf_h1, "R/models/mrf_h1.rds")

# Full-sample GTVP: contemporaneous regressors, since this fit is the
# elasticity path rather than a forecast.
set.seed(SEED)
gfit <- fit_mrf(first_ok:n, c("margin", "r3"), oos = c(), vi = VI)
bnames <- c("(Intercept)", "margin", "r3")
colnames(gfit$betas) <- bnames
names(gfit$VI_oob) <- gfit$S.names
dimnames(gfit$VI_betas) <- list(bnames, gfit$S.names)
saveRDS(gfit, "R/models/mrf_gtvp.rds")

qb <- function(p) apply(gfit$betas.draws, c(1, 2), quantile, probs = p, na.rm = TRUE)
betas <- data.frame(
  date = as.character(as.Date(dts[first_ok:n])),
  coef = rep(bnames, each = length(first_ok:n)),
  beta = as.vector(gfit$betas),
  lo90 = as.vector(qb(0.05)), lo68 = as.vector(qb(0.16)),
  hi68 = as.vector(qb(0.84)), hi90 = as.vector(qb(0.95))
)
dir.create("data/outputs/forecasts", recursive = TRUE, showWarnings = FALSE)
write.csv(betas, "data/outputs/forecasts/mrf_gtvp_betas.csv", row.names = FALSE)

cat("\nGTVP mean elasticities:\n")
print(round(colMeans(gfit$betas), 4))
if (VI) {
  cat("\nVI_oob (top 5):\n")
  print(round(head(sort(gfit$VI_oob, decreasing = TRUE), 5), 4))
  cat("\ntime variation by coefficient (total VI, sd of path):\n")
  print(round(data.frame(total_VI = rowSums(gfit$VI_betas),
                         sd_beta = apply(gfit$betas, 2, sd)), 4))
  write.csv(as.data.frame(t(gfit$VI_betas)),
            "data/outputs/forecasts/mrf_gtvp_vi.csv")
}
cat("\nWrote R/models/mrf_h1.rds, R/models/mrf_gtvp.rds,",
    "data/outputs/forecasts/mrf_gtvp_betas.csv\n")
