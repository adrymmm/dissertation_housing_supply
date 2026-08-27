# Does the margin restriction (b_price = -b_cost) hold once the ridge, rather
# than the restriction, handles the price/cost collinearity? Full-sample MRF
# with lrprc and lrcc entering separately. Writes R/models/mrf_unrestricted.rds.

library(zoo)
library(pracma)
source("R/vendor/MacroRF/MRF_v210403.R")

B <- 250L; QRATE <- 0.3; SEED <- 42L

eng_zoo <- readRDS("R/models/nardl_eng_zoo.rds")
dat <- as.data.frame(eng_zoo)
dts <- as.yearqtr(as.numeric(zoo::index(eng_zoo)))
n <- nrow(dat)
L <- function(x, k) c(rep(NA_real_, k), head(x, -k))

D <- data.frame(
  lhstarts = dat$lhstarts, lrprc = dat$lrprc, lrcc = dat$lrcc, r3 = dat$r3,
  y_l1 = L(dat$lhstarts, 1), y_l4 = L(dat$lhstarts, 4),
  margin_l1 = L(dat$lrprc - dat$lrcc, 1), r3_l1 = L(dat$r3, 1),
  lrprc_l1 = L(dat$lrprc, 1), lrcc_l1 = L(dat$lrcc, 1),
  lvol_l1 = L(dat$lvol, 1), lstock_l1 = L(dat$lstock, 1),
  trend = seq_len(n), qtr = as.integer(round(4 * (as.numeric(dts) %% 1))) + 1L)

S_COLS <- c("y_l1", "y_l4", "margin_l1", "r3_l1", "lrprc_l1", "lrcc_l1",
            "lvol_l1", "lstock_l1", "trend", "qtr")
X_COLS <- c("lrprc", "lrcc", "r3")
rows <- (max(which(!stats::complete.cases(D))) + 1L):n
cols <- c("lhstarts", X_COLS, S_COLS)

set.seed(SEED)
u <- MRF(data = as.matrix(D[rows, cols]), y.pos = 1L,
         x.pos = match(X_COLS, cols), S.pos = match(S_COLS, cols),
         oos.pos = c(), B = B, quantile.rate = QRATE, VI = TRUE,
         resampling.opt = 2, block.size = 12,
         trend.pos = match("trend", cols), trend.push = 4,
         printb = FALSE, cheap.look.at.GTVPs = FALSE)

bn <- c("(Intercept)", X_COLS)
colnames(u$betas) <- bn
dimnames(u$VI_betas) <- list(bn, u$S.names)
names(u$VI_oob) <- u$S.names
saveRDS(u, "R/models/mrf_unrestricted.rds")

cat("\ntime variation by coefficient:\n")
print(round(data.frame(total_VI = rowSums(u$VI_betas),
                       sd_beta = apply(u$betas, 2, sd),
                       mean_beta = colMeans(u$betas)), 4))

# the restriction, tested pointwise: is b_price + b_cost = 0 at each date?
s <- u$betas.draws[, 2, ] + u$betas.draws[, 3, ]
qs <- apply(s, 1, quantile, probs = c(0.05, 0.5, 0.95), na.rm = TRUE)
cat(sprintf("\nb_price + b_cost: median %.3f, mean %.3f\n",
            median(qs[2, ]), mean(qs[2, ])))
cat(sprintf("90%% band excludes 0 in %.0f%% of quarters\n",
            100 * mean(qs[1, ] > 0 | qs[3, ] < 0)))
cat(sprintf("corr(b_price path, -b_cost path) = %.3f\n",
            cor(u$betas[, "lrprc"], -u$betas[, "lrcc"])))

png("data/outputs/figures/gtvp_unrestricted.png", width = 1000, height = 800, res = 120)
par(mfrow = c(3, 1), mar = c(2.5, 4, 2, 1))
d <- as.Date(dts[rows])
for (v in X_COLS) {
  q <- apply(u$betas.draws[, match(v, bn), ], 1, quantile,
             probs = c(0.05, 0.95), na.rm = TRUE)
  plot(d, u$betas[, v], type = "n", ylim = range(q), xlab = "", ylab = "", main = v)
  polygon(c(d, rev(d)), c(q[1, ], rev(q[2, ])), col = "grey85", border = NA)
  lines(d, u$betas[, v], lwd = 2); abline(h = 0, lty = 2)
}
dev.off()
cat("\nWrote R/models/mrf_unrestricted.rds, data/outputs/figures/gtvp_unrestricted.png\n")
