library(ARDL)
library(car)
library(sandwich)
library(zoo)
library(lmtest)

# RUN NARDL SCRIPT FIRST

eng_zoo <- readRDS("R/models/nardl_eng_zoo.rds")  
dum     <- readRDS("R/models/nardl_dum.rds")       
fits    <- readRDS("R/models/nardl_fits.rds")      

nl_vars  <- names(fits)
dum_nosd <- setdiff(dum, c("sd1","sd2","sd3"))

vcovs <- list(
  HC1   = function(m) vcovHC(m, type = "HC1"),
  HAC   = function(m) vcovHAC(m),
  NWpw  = function(m) NeweyWest(m, prewhite = TRUE),
  NWfix = function(m) NeweyWest(m, lag = floor(4 * (nobs(m)/100)^(2/9)),
                                prewhite = FALSE)
)

# ---- refit only for ablations that change the spec (dummy set / max_order /
# common lag order) — the headline spec is never re-estimated, it's pulled
# straight from `fits`
refit <- function(x, dum_set, max_order, force_common = FALSE) {
  base <- c("lrprc", "lvol", "r3", "lrcc")
  rhs  <- c(paste0(x, c("_pos", "_neg")), setdiff(base, x))
  dat  <- eng_zoo[, c("lhstarts", rhs, dum_set)]
  f    <- as.formula(paste("lhstarts ~", paste(rhs, collapse = " + "),
                           "|", paste(dum_set, collapse = " + ")))
  m <- auto_ardl(f, data = dat, max_order = max_order)$best_model
  if (force_common) {
    o <- m$order
    o[2] <- o[3] <- max(o[2], o[3], 1)
    m <- ardl(f, data = dat, order = o)
  }
  list(fit = m, uecm = uecm(m))
}

# ---- symmetry test across the vcov grid, given a fitted uecm ----------------
sym_grid <- function(u, x, label) {
  cf <- coef(u); k <- length(cf)
  allp <- grep(paste0(x, "_pos"), names(cf), value = TRUE, fixed = TRUE)
  alln <- grep(paste0(x, "_neg"), names(cf), value = TRUE, fixed = TRUE)
  sets <- list(
    lr = list(p = allp[!grepl("^d\\(", allp)], n = alln[!grepl("^d\\(", alln)]),
    sr = list(p = grep("^d\\(", allp, value = TRUE),
              n = grep("^d\\(", alln, value = TRUE))
  )
  rows <- list()
  for (tn in names(sets)) {
    pp <- sets[[tn]]$p; nn <- sets[[tn]]$n
    if (!length(pp) || !length(nn) || anyNA(cf[c(pp, nn)])) {
      rows[[length(rows)+1]] <- data.frame(
        spec = label, var = x, test = tn, vcov = NA, npos = length(pp),
        nneg = length(nn), F = NA, p = NA)
      next
    }
    v <- setNames(numeric(k), names(cf)); v[pp] <- 1; v[nn] <- -1
    C <- matrix(v, nrow = 1)
    for (vn in names(vcovs)) {
      a <- try(linearHypothesis(u, C, rhs = 0, vcov. = vcovs[[vn]](u)),
               silent = TRUE)
      rows[[length(rows)+1]] <- data.frame(
        spec = label, var = x, test = tn, vcov = vn,
        npos = length(pp), nneg = length(nn),
        F = if (inherits(a, "try-error")) NA else a[2, "F"],
        p = if (inherits(a, "try-error")) NA else a[2, "Pr(>F)"])
    }
  }
  do.call(rbind, rows)
}

# ---- long-run asymmetry magnitude with HAC CI --------------------------------
theta_gap <- function(u, x, label) {
  cf  <- coef(u)
  ect <- grep("^L\\(lhstarts", names(cf), value = TRUE)[1]
  lp  <- grep(paste0("L\\(", x, "_pos"), names(cf), value = TRUE)[1]
  ln  <- grep(paste0("L\\(", x, "_neg"), names(cf), value = TRUE)[1]
  if (any(is.na(c(ect, lp, ln)))) return(NULL)
  rho <- cf[ect]
  V   <- vcovHAC(u)
  d   <- cf[lp] - cf[ln]
  sd  <- sqrt(V[lp,lp] + V[ln,ln] - 2*V[lp,ln])
  data.frame(spec = label, var = x,
             theta_pos = as.numeric(-cf[lp]/rho),
             theta_neg = as.numeric(-cf[ln]/rho),
             beta_diff = as.numeric(d),
             lo = as.numeric(d - 1.96*sd), hi = as.numeric(d + 1.96*sd),
             theta_diff = as.numeric(-d/rho), row.names = NULL)
}

res <- list(); thet <- list(); resid_diag <- list()

# ---- headline spec: pulled from fits, no refit --------------------------------
for (x in nl_vars) {
  u <- fits[[x]]$uecm
  res[[length(res)+1]]   <- sym_grid(u, x, "seas_mo6")
  g <- theta_gap(u, x, "seas_mo6")
  if (!is.null(g)) thet[[length(thet)+1]] <- g
  
  rs <- residuals(fits[[x]]$fit)
  resid_diag[[length(resid_diag)+1]] <- data.frame(
    spec = "seas_mo6", var = x,
    shapiro_p = tryCatch(shapiro.test(rs)$p.value, error = function(e) NA),
    bg4_p     = tryCatch(bgtest(fits[[x]]$fit, order = 4)$p.value, error = function(e) NA),
    bg8_p     = tryCatch(bgtest(fits[[x]]$fit, order = 8)$p.value, error = function(e) NA),
    arch4_p   = tryCatch(FinTS::ArchTest(rs, lags = 4)$p.value, error = function(e) NA)
  )
}

# ---- common-lag robustness (equalise pos/neg order), all four ---------------
for (x in nl_vars) {
  z <- refit(x, dum, 6, force_common = TRUE)
  cat(x, "common-lag order:", paste(z$fit$order, collapse = ","), "\n")
  res[[length(res)+1]]  <- sym_grid(z$uecm, x, "seas_mo6_common")
  g <- theta_gap(z$uecm, x, "seas_mo6_common")
  if (!is.null(g)) thet[[length(thet)+1]] <- g
}

# ---- ablations isolating seasonals vs max_order, for lrcc and r3 ------------
for (x in c("lrcc", "r3")) {
  z <- refit(x, dum_nosd, 4)   # old spec: no seasonals, max_order 4
  cat(x, "nosd_mo4 order:", paste(z$fit$order, collapse = ","), "\n")
  res[[length(res)+1]] <- sym_grid(z$uecm, x, "nosd_mo4")
  
  z <- refit(x, dum, 4)        # seasonals, but old max_order
  res[[length(res)+1]] <- sym_grid(z$uecm, x, "seas_mo4")
  
  z <- refit(x, dum_nosd, 6)   # no seasonals, but new max_order
  res[[length(res)+1]] <- sym_grid(z$uecm, x, "nosd_mo6")
}

sym_all    <- do.call(rbind, res)
theta_all  <- do.call(rbind, thet)
resid_all  <- do.call(rbind, resid_diag)

cat("\n=== symmetry tests: spec x vcov ===\n")
print(sym_all, row.names = FALSE, digits = 4)
cat("\n=== long-run asymmetry magnitude (HAC CI on beta_pos - beta_neg) ===\n")
print(theta_all, row.names = FALSE, digits = 4)
cat("\n=== residual diagnostics, headline spec ===\n")
print(resid_all, row.names = FALSE, digits = 4)

cat("\n=== Andrews / Newey-West bandwidths (headline spec) ===\n")
for (x in nl_vars) {
  u <- fits[[x]]$uecm
  cat(x, " n =", nobs(u), " k =", length(coef(u)),
      " bwNW =", round(bwNeweyWest(u), 2), "\n")
}

saveRDS(sym_all,   "R/models/nardl_sym_grid.rds")
saveRDS(theta_all, "R/models/nardl_theta_gap.rds")
saveRDS(resid_all, "R/models/nardl_resid_diag.rds")