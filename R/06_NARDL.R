library(ARDL)
library(zoo)

eng_tf <- readRDS("R/models/eng_tf.rds")
eng_ts <- ts(eng_tf, start = c(1975, 1), frequency = 4)

# Impulse dummies
ta <- time(eng_ts); dd <- function(yr) which.min(abs(ta - yr))
D <- matrix(0, nrow(eng_ts), 5,
            dimnames = list(NULL, c("d08Q3","d20Q2","d20Q3","d23Q2","d23Q3")))
D[dd(2008.50),1] <- 1
D[dd(2020.25),2] <- 1
D[dd(2020.50),3] <- 1
D[dd(2023.25),4] <- 1
D[dd(2023.50),5] <- 1

# Centred seasonals (matches ca.jo's season=4 and ardl_main.R)
q <- cycle(eng_ts)
S <- outer(as.numeric(q), 1:3, "==") - 1/4
colnames(S) <- c("sd1","sd2","sd3")

eng_df <- as.data.frame(cbind(as.matrix(eng_tf), D, S))

# Partial sum decomposition — all four variables that may serve as the
# asymmetric regressor x in run_nardl()
nl_vars <- c("lrprc", "lrcc", "r3", "lvol")
for (var in nl_vars) {
  d <- c(0, diff(eng_df[[var]]))
  eng_df[[paste0(var, "_pos")]] <- cumsum(pmax(d, 0))
  eng_df[[paste0(var, "_neg")]] <- cumsum(pmin(d, 0))
}

eng_df <- eng_df[is.finite(rowSums(eng_df)), ]
eng_zoo <- as.zooreg(ts(eng_df, start = c(1975, 1), frequency = 4))

# Impulse dummies and seasonality dums
dum <- c("d08Q3","d20Q2","d20Q3","d23Q2","d23Q3","sd1","sd2","sd3")

run_nardl <- function(x, max_order = 6) {
  base <- c("lrprc", "lvol", "r3", "lrcc")
  rhs  <- c(paste0(x, c("_pos", "_neg")), setdiff(base, x))
  dat  <- eng_zoo[, c("lhstarts", rhs, dum)]
  f    <- as.formula(paste("lhstarts ~", paste(rhs, collapse = " + "),
                           "|", paste(dum, collapse = " + ")))
  
  nardl_fit <- auto_ardl(f, data = dat, max_order = max_order)$best_model
  uecm_fit  <- uecm(nardl_fit)
  cf        <- coef(uecm_fit)
  
  list(
    fit      = nardl_fit,
    uecm     = uecm_fit,
    rhs      = rhs,
    order    = nardl_fit$order,
    bounds_f = bounds_f_test(nardl_fit, case = 3),
    bounds_t = bounds_t_test(nardl_fit, case = 3),
    recm     = recm(nardl_fit, case = 3),
    mult     = multipliers(nardl_fit),
    aliased  = names(which(is.na(cf)))
  )
}

out <- lapply(nl_vars, run_nardl)
names(out) <- nl_vars

for (nm in names(out)) {
  cat(nm, "selected order:", paste(out[[nm]]$order, collapse = ","), "\n")
  if (length(out[[nm]]$aliased)) {
    cat("  WARNING aliased terms:", paste(out[[nm]]$aliased, collapse = ", "), "\n")
  }
}

# Bounds test summary
bounds_tbl <- data.frame(
  var = names(out),
  F   = sapply(out, function(z) unname(z$bounds_f$statistic)),
  Fp  = sapply(out, function(z) z$bounds_f$p.value),
  row.names = NULL
)
bounds_tbl

saveRDS(eng_zoo, "R/models/nardl_eng_zoo.rds")
saveRDS(dum,     "R/models/nardl_dum.rds")
saveRDS(out,     "R/models/nardl_fits.rds")   # named list, one entry per variable