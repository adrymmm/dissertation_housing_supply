library(urca)
library(ARDL)

jo_eng    <- readRDS("R/models/jo_eng.rds")
ardl_best <- readRDS("R/models/ardl_best.rds")
ardl_ecm  <- readRDS("R/models/ardl_ecm.rds")
nardl_lrcc <- readRDS("R/models/nardl_lrcc.rds")
sym_tbl <- readRDS("R/models/sym_tbl.rds")
out <- list(lrcc = nardl_lrcc)

# --- VECM ---
cj <- cajorls(jo_eng, r = 1)   # call once, reuse

# Long-run elasticities, normalised on lhstarts (off-diagonals negated)
elas <- -cj$beta / cj$beta["lhstarts.l1", ]
elas["lhstarts.l1", ] <- 1
vecm_lr <- setNames(as.numeric(elas), sub("\\.l1$", "", rownames(elas)))
vecm_lr <- vecm_lr[names(vecm_lr) != "lhstarts"]   # lrprc, lvol, r3, lstock, lrcc

# Adjustment speed (alpha) from the lhstarts equation
srlm  <- summary(cj$rlm)
ect_h <- srlm[["Response lhstarts.d"]]$coefficients["ect1", ]
vecm_alpha <- c(alpha = unname(ect_h["Estimate"]),
                t     = unname(ect_h["t value"]))

# --- ARDL ---
# Long-run elasticities (multipliers carry SE/t/p — keep if you want significance marks)
ardl_mult <- multipliers(ardl_best)
keep    <- c("lrprc","lvol","r3","lstock","lrcc")
ardl_lr <- setNames(ardl_mult$Estimate[match(keep, ardl_mult$Term)], keep)

# Speed of adjustment from the conditional ECM
ardl_ecm  <- recm(ardl_best, case = 3)
ect_a     <- summary(ardl_ecm)$coefficients["ect", ]
ardl_alpha <- c(alpha = unname(ect_a["Estimate"]),
                t     = unname(ect_a["t value"]))

ardl_lr
ardl_alpha

# --- NARDL ---
nardl <- out[["lrcc"]]                 # headline: lrcc decomposed, rest symmetric

# Long-run elasticities (lrcc split +/- , others symmetric)
nardl_mult <- nardl$mult
keep     <- c("lrcc_pos","lrcc_neg","lrprc","lvol","r3","lstock")
nardl_lr <- setNames(nardl_mult$Estimate[match(keep, nardl_mult$Term)], keep)

# Speed of adjustment
ect_n       <- summary(nardl$recm)$coefficients["ect", ]
nardl_alpha <- c(alpha = unname(ect_n["Estimate"]), t = unname(ect_n["t value"]))

# Asymmetry evidence for the slide / write-up
sym_tbl[sym_tbl$var %in% c("lrcc","r3"), c("var","lr_F","lr_p")]

# --- Combined Table ---
vars_order <- c("lrprc","lvol","r3","lstock","lrcc","lrcc_pos","lrcc_neg")
pick <- function(v) unname(v[match(vars_order, names(v))])   # NA where a model lacks that row

lr_tab <- data.frame(
  variable = vars_order,
  VECM  = pick(vecm_lr),
  ARDL  = pick(ardl_lr),
  NARDL = pick(nardl_lr)
)

# adjustment speed as a footer row
lr_tab <- rbind(lr_tab, data.frame(
  variable = "alpha (adj. speed)",
  VECM  = unname(vecm_alpha["alpha"]),
  ARDL  = unname(ardl_alpha["alpha"]),
  NARDL = unname(nardl_alpha["alpha"])
))

lr_tab[ , -1] <- round(lr_tab[ , -1], 3)
lr_tab