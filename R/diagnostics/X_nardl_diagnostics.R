library(ARDL)
library(zoo)
library(lmtest)
library(tseries)
library(car)
library(sandwich)

eng_zoo <- readRDS("R/models/nardl_eng_zoo.rds")
dum     <- readRDS("R/models/nardl_dum.rds")
out     <- readRDS("R/models/nardl_fits.rds")

reset_manual <- function(m, powers = 2:3) {
  y <- as.numeric(model.response(model.frame(m)))
  X <- model.matrix(m)
  f <- as.numeric(fitted(m))
  aux_terms <- poly(f, max(powers), raw = TRUE)[, powers, drop = FALSE]
  aux  <- lm(y ~ X - 1 + aux_terms)
  base <- lm(y ~ X - 1)
  waldtest(base, aux)
}

# ---- residual diagnostics on each selected NARDL --------------------------
diag_tbl <- do.call(rbind, lapply(out, function(z) {
  m <- z$fit
  data.frame(var = z$var, order = paste(z$order, collapse=","),
             n = nobs(m), k = length(coef(m)),
             BG4_p  = bgtest(m, order = 4)$p.value,
             BG8_p  = bgtest(m, order = 8)$p.value,
             BP_p   = bptest(m)$p.value,
             JB_p   = jarque.bera.test(residuals(m))$p.value,
             RESET_p = reset_manual(m)$`Pr(>F)`[2])
}))
cat("== residual diagnostics per NARDL screen ==\n")
print(diag_tbl, row.names = FALSE, digits = 3)

# Run this before selection sensitivity test
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


hac <- function(m) NeweyWest(m, lag = 4, prewhite = FALSE)

test_symmetry <- function(m, x) {
  ue <- uecm(m)
  b  <- names(coef(m))
  pick <- function(v) grep(sprintf("^%s$|^L\\(%s, ", v, v), b, value = TRUE)
  pl <- pick(paste0(x, "_pos")); ng <- pick(paste0(x, "_neg"))
  lr <- linearHypothesis(m, paste(paste(pl, collapse=" + "), "=",
                                  paste(ng, collapse=" + ")), vcov. = hac(m))
  
  eb <- names(coef(ue))
  ep <- function(v) grep(sprintf("^d\\(%s\\)$|^d\\(L\\(%s, ", v, v), eb, value = TRUE)
  ps <- ep(paste0(x, "_pos")); nsr <- ep(paste0(x, "_neg"))
  sr <- if (length(ps) && length(nsr))
    linearHypothesis(ue, paste(paste(ps, collapse=" + "), "=",
                               paste(nsr, collapse=" + ")), vcov. = hac(ue))
  
  c(LR_F = unname(lr$F[2]), LR_p = lr$`Pr(>F)`[2],
    SRadd_F = if (is.null(sr)) NA else unname(sr$F[2]),
    SRadd_p = if (is.null(sr)) NA else sr$`Pr(>F)`[2])
}

for (nm in names(out)) {
  reg <- out[[nm]]$reg
  f   <- as.formula(paste("lhstarts ~", paste(reg, collapse = " + "),
                          "|", paste(dum, collapse = " + ")))
  cat("\n==", nm, "==\n")
  for (ml in c(5, 6, 7, 8, 10)) {
    g   <- gts(f, eng_zoo[, c("lhstarts", reg, dum)], c(ml, rep(ml, length(reg))))
    res <- test_symmetry(g$fit, nm)
    cat("max_lag=", ml, " order=", paste(g$order, collapse=","),
        " LR_p=", round(res["LR_p"], 3),
        " SRadd_p=", round(res["SRadd_p"], 3), "\n", sep = "")
  }
}