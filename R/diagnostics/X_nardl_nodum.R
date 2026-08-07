library(ARDL)
library(zoo)
library(car)
library(sandwich)

eng_tf <- readRDS("R/models/eng_tf.rds")
eng_ts <- ts(eng_tf, start = c(1975, 1), frequency = 4)
ta <- time(eng_ts); dd <- function(yr) which.min(abs(ta - yr))

# Seasonals
S <- outer(as.numeric(cycle(eng_ts)), 1:3, "==") - 1/4
colnames(S) <- c("sd1","sd2","sd3")

eng_df <- as.data.frame(cbind(as.matrix(eng_tf), S))
nl_vars <- c("lrprc", "lrcc", "r3", "lvol")

# Partial sums
for (v in nl_vars) {
  d <- c(0, diff(eng_df[[v]]))
  eng_df[[paste0(v,"_pos")]] <- cumsum(pmax(d, 0))
  eng_df[[paste0(v,"_neg")]] <- cumsum(pmin(d, 0))
}
stopifnot(all(is.finite(as.matrix(eng_df))))

# Converting to zoo for forecast
eng_zoo <- as.zooreg(ts(eng_df, start = c(1975, 1), frequency = 4))
dum  <- c("sd1","sd2","sd3")
base <- c("lrprc", "lvol", "r3", "lrcc")
# Wrapper for HAC standard errors
hac  <- function(m) NeweyWest(m, lag = 4, prewhite = FALSE)

# PSS Case III 5% I(1) bounds, k=1..7 (Tables CI(iii)/CII(iii)
Fhi <- c(5.73, 4.85, 4.35, 4.01, 3.79, 3.61, 3.50)
Thi <- c(-3.22, -3.53, -3.78, -3.99, -4.19, -4.38, -4.57)

# General to specific lag selection
gts <- function(f, data, o, thresh = 0.10) {
  vars <- all.vars(f)[seq_along(o)]
  repeat {
    m     <- ardl(f, data = data, order = o)
    cf    <- summary(m)$coefficients
    tails <- ifelse(o > 0, paste0("L(", vars, ", ", o, ")"), NA)
    keep  <- !is.na(tails) & tails %in% rownames(cf) & c(o[1] > 1, o[-1] > 0)
    if (!any(keep)) return(list(fit = m, order = o))
    p <- cf[tails[keep], "Pr(>|t|)"]
    if (max(p) <= thresh) return(list(fit = m, order = o))
    i <- which(keep)[which.max(p)]
    o[i] <- o[i] - 1
  }
}

# NARDL with one decomposed, rest symmetric
run_nardl <- function(x, max_lag = 8, thresh = 0.10) {
  reg <- c(paste0(x, c("_pos","_neg")), setdiff(base, x))
  f   <- as.formula(paste("lhstarts ~", paste(reg, collapse = " + "),
                          "|", paste(dum, collapse = " + ")))
  g   <- gts(f, eng_zoo[, c("lhstarts", reg, dum)],
             c(max_lag, rep(max_lag, length(reg))), thresh)
  m   <- g$fit
  ue  <- uecm(m)
  b   <- names(coef(m))
  
  # Coefficient name extractor helper
  pick <- function(v) grep(sprintf("^%s$|^L\\(%s, ", v, v), b, value = TRUE)
  
  # Grab decomposed coeffs
  pl <- pick(paste0(x, "_pos")); ng <- pick(paste0(x, "_neg"))
  
  # LR Asymmetry: Wald Test if sum of positive sums is equal to negative sums
  lr <- linearHypothesis(m, paste(paste(pl, collapse=" + "), "=",
                                  paste(ng, collapse=" + ")), vcov. = hac(m))
  
  # Differenced coefs name extractor helper
  eb <- names(coef(ue))
  ep <- function(v) grep(sprintf("^d\\(%s\\)$|^d\\(L\\(%s, ", v, v), eb, value = TRUE)
  
  # Grab decomposed differenced coeffs
  ps <- ep(paste0(x, "_pos")); nsr <- ep(paste0(x, "_neg"))
  # SR Asymmetry: Additive
  sr <- if (length(ps) && length(nsr))
    linearHypothesis(ue, paste(paste(ps, collapse=" + "), "=",
                               paste(nsr, collapse=" + ")), vcov. = hac(ue))
  
  # SR Asymmetry: Pairwise
  sw <- if (length(ps) && length(ps) == length(nsr))
    linearHypothesis(ue, paste(ps, "=", nsr), vcov. = hac(ue))
  
  list(var = x, fit = m, uecm = ue, reg = reg, order = g$order,
       bounds_f = bounds_f_test(m, case = 3),
       bounds_t = bounds_t_test(m, case = 3),
       recm = recm(m, case = 3), mult = multipliers(m),
       aliased = names(which(is.na(coef(m)))),
       LR_F = lr$F[2], LR_p = lr$`Pr(>F)`[2],
       SRadd_F = if (is.null(sr)) NA else sr$F[2],
       SRadd_p = if (is.null(sr)) NA else sr$`Pr(>F)`[2],
       SRpw_F  = if (is.null(sw)) NA else sw$F[2],
       SRpw_p  = if (is.null(sw)) NA else sw$`Pr(>F)`[2])
}

out <- lapply(nl_vars, run_nardl)
names(out) <- nl_vars

summary_tbl <- do.call(rbind, lapply(out, function(z) {
  k <- length(z$reg)
  data.frame(var = z$var, order = paste(z$order, collapse=","), k_lo = k-1, k_hi = k,
             F_PSS = unname(z$bounds_f$statistic),
             F_crit_lo = Fhi[k - 1], F_crit_hi = Fhi[k],
             t_BDM = unname(z$bounds_t$statistic),
             t_crit_lo = Thi[k - 1], t_crit_hi = Thi[k],
             LR_F = z$LR_F, LR_p = z$LR_p,
             SRadd_F = z$SRadd_F, SRadd_p = z$SRadd_p,
             SRpw_F = z$SRpw_F, SRpw_p = z$SRpw_p)
}))

short_sum <- do.call(rbind, lapply(out, function(z) {
  k <- length(z$reg)
  data.frame(var = z$var, order = paste(z$order, collapse=","),
             LR_p = z$LR_p,
             SRadd_p = z$SRadd_p,
             SRpw_p = z$SRpw_p)
}))

#print(summary_tbl, row.names = FALSE, digits = 3)
cat("================ SPEC: No Impulse Dummies ================ ")
print(short_sum, row.names = FALSE, digits = 3)