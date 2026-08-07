library(ARDL)
library(zoo)
library(car)
library(sandwich)
library(lmtest)
library(tseries)

base <- c("lrprc", "lvol", "r3", "lrcc")
seas <- c("sd1", "sd2", "sd3")

hac <- function(m) NeweyWest(m, lag = 4, prewhite = FALSE)

# PSS Case III 5% I(1) bounds, k=1..7 (Tables CI(iii)/CII(iii))
Fhi <- c(5.73, 4.85, 4.35, 4.01, 3.79, 3.61, 3.50)
Thi <- c(-3.22, -3.53, -3.78, -3.99, -4.19, -4.38, -4.57)

# General-to-specific lag selection
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

# NARDL with one decomposed variable, rest symmetric.
run_nardl <- function(x, dum, eng_zoo, max_lag = 8, thresh = 0.10) {
  reg   <- c(paste0(x, c("_pos", "_neg")), setdiff(base, x))
  fixed <- c(dum, seas)
  
  f <- if (length(fixed) > 0) {
    as.formula(paste("lhstarts ~", paste(reg, collapse = " + "),
                     "|", paste(fixed, collapse = " + ")))
  } else {
    as.formula(paste("lhstarts ~", paste(reg, collapse = " + ")))
  }
  
  g  <- gts(f, eng_zoo[, c("lhstarts", reg, fixed)],
            c(max_lag, rep(max_lag, length(reg))), thresh)
  m  <- g$fit
  ue <- uecm(m)
  b  <- names(coef(m))
  nonalias <- names(which(!is.na(coef(m))))
  
  # Coefficient name extractor helper
  pick <- function(v) grep(sprintf("^%s$|^L\\(%s, ", v, v), b, value = TRUE)
  # Grab decomposed coeffs
  pl <- intersect(pick(paste0(x, "_pos")), nonalias)
  ng <- intersect(pick(paste0(x, "_neg")), nonalias)
  
  # LR Asymmetry: Wald Test if sum of positive sums is equal to negative sums
  lr <- linearHypothesis(m, paste(paste(pl, collapse = " + "), "=",
                                  paste(ng, collapse = " + ")), vcov. = hac(m))
  
  # Differenced coefs name extractor helper
  eb <- names(coef(ue))
  ep <- function(v) grep(sprintf("^d\\(%s\\)$|^d\\(L\\(%s, ", v, v), eb, value = TRUE)
  
  # Grab decomposed differenced coeffs
  ps  <- ep(paste0(x, "_pos"))
  nsr <- ep(paste0(x, "_neg"))
  
  # SR Asymmetry: Additive
  sr <- if (length(ps) && length(nsr))
    linearHypothesis(ue, paste(paste(ps, collapse = " + "), "=",
                               paste(nsr, collapse = " + ")), vcov. = hac(ue))
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

# Ramsey RESET via manual auxiliary regression
reset_manual <- function(m, powers = 2:3) {
  y <- as.numeric(model.response(model.frame(m)))
  X <- model.matrix(m)
  f <- as.numeric(fitted(m))
  aux_terms <- poly(f, max(powers), raw = TRUE)[, powers, drop = FALSE]
  aux  <- lm(y ~ X - 1 + aux_terms)
  base <- lm(y ~ X - 1)
  waldtest(base, aux, vcov = function(z) NeweyWest(z, lag = 4, prewhite = FALSE))
}

# Residual diagnostics per NARDL fit
residual_diag_table <- function(out_all) {
  do.call(rbind, lapply(names(out_all), function(spec_name) {
    do.call(rbind, lapply(out_all[[spec_name]], function(z) {
      m   <- z$fit
      res <- residuals(m)
      
      bg4   <- tryCatch(bgtest(m, order = 4), error = function(e) NULL)
      bg8   <- tryCatch(bgtest(m, order = 8), error = function(e) NULL)
      bp    <- tryCatch(bptest(m), error = function(e) NULL)
      jb    <- tryCatch(jarque.bera.test(as.numeric(res)), error = function(e) NULL)
      sw    <- tryCatch(shapiro.test(as.numeric(res)), error = function(e) NULL)
      reset <- tryCatch(reset_manual(m), error = function(e) NULL)
      
      data.frame(
        spec    = spec_name,
        var     = z$var,
        order   = paste(z$order, collapse = ","),
        n       = nobs(m),
        k       = length(coef(m)),
        BG4_LM  = if (is.null(bg4)) NA_real_ else unname(bg4$statistic),
        BG4_p   = if (is.null(bg4)) NA_real_ else bg4$p.value,
        BG8_LM  = if (is.null(bg8)) NA_real_ else unname(bg8$statistic),
        BG8_p   = if (is.null(bg8)) NA_real_ else bg8$p.value,
        BP_p    = if (is.null(bp))  NA_real_ else bp$p.value,
        JB_p    = if (is.null(jb))  NA_real_ else jb$p.value,
        SW_W    = if (is.null(sw))  NA_real_ else unname(sw$statistic),
        SW_p    = if (is.null(sw))  NA_real_ else sw$p.value,
        RESET_p = if (is.null(reset)) NA_real_ else reset$`Pr(>F)`[2],
        n_aliased = length(z$aliased)
      )
    }))
  }))
}

# Turns every spec into a table
summarize_specs <- function(out_all) {
  do.call(rbind, lapply(names(out_all), function(spec_name) {
    do.call(rbind, lapply(out_all[[spec_name]], function(z) {
      k <- length(z$reg)
      data.frame(spec = spec_name, var = z$var, order = paste(z$order, collapse = ","),
                 k_lo = k - 1, k_hi = k,
                 F_PSS = unname(z$bounds_f$statistic),
                 F_crit_lo = Fhi[k - 1], F_crit_hi = Fhi[k],
                 t_BDM = unname(z$bounds_t$statistic),
                 t_crit_lo = Thi[k - 1], t_crit_hi = Thi[k],
                 LR_F = z$LR_F, LR_p = z$LR_p,
                 SRadd_F = z$SRadd_F, SRadd_p = z$SRadd_p,
                 SRpw_F = z$SRpw_F, SRpw_p = z$SRpw_p)
    }))
  }))
}

print_short_sum <- function(summary_tbl) {
  short_sum <- summary_tbl[, c("spec", "var", "order", "LR_p", "SRadd_p", "SRpw_p")]
  for (s in unique(short_sum$spec)) {
    cat(sprintf("================ SPEC: %s ================\n", s))
    print(short_sum[short_sum$spec == s, ], row.names = FALSE, digits = 3)
  }
}

print_full_sum <- function(summary_tbl) {
  for (s in unique(summary_tbl$spec)) {
    cat(sprintf("================ SPEC: %s ================\n", s))
    print(summary_tbl[summary_tbl$spec == s, ], row.names = FALSE, digits = 3)
  }
}