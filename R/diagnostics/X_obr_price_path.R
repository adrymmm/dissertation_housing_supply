# Reconciles the OBR real house price path (lrprc) used for R/09_foreARDL.R
# against the VECM-endogenous lrprc path from R/10_foreVECM.R, to check the
# "2.89% / 2.85% / 3.72% / 6.72%" figures used in the exec summary draft.
# Standalone - reads committed CSVs directly, writes nothing.

obr       <- read.csv("data/python_master/OBR/obr_scenario.csv")
vecm_cov  <- read.csv("data/outputs/forecasts/vecm_implied_covariates.csv")
master    <- read.csv("data/python_master/england_master.csv")
colnames(master)[1] <- "period"

# ---- Confirm lrprc is in log form, not levels -----------------------------
# lrprc is built as log(hprice / gdp_def) in R/01_data_loading.R and mirrored
# in python/core/07_obr_data.ipynb. Sanity check: exp(lrprc) at the anchor
# should reproduce the actual hprice/gdp_def ratio from the master data.
anchor_row   <- master[master$period == "2025Q4", ]
anchor_ratio <- anchor_row$hprice / anchor_row$gdp_def
anchor_lrprc <- obr$lrprc[obr$period == "2025Q4"]
cat(sprintf(
  "lrprc form check: exp(lrprc_2025Q4)=%.4f vs hprice/gdp_def=%.4f (match confirms log form)\n\n",
  exp(anchor_lrprc), anchor_ratio
))
stopifnot(isTRUE(all.equal(exp(anchor_lrprc), anchor_ratio)))
stopifnot(isTRUE(all.equal(anchor_lrprc, log(anchor_ratio))))

# ---- Raw lrprc values used -------------------------------------------------
lrprc_2025q4      <- anchor_lrprc
lrprc_obr_2029q1  <- obr$lrprc[obr$period == "2029Q1"]
lrprc_obr_2031q1  <- obr$lrprc[obr$period == "2031Q1"]
lrprc_vecm_2026q1 <- vecm_cov$lrprc[vecm_cov$period == "2026Q1"]
lrprc_vecm_2031q1 <- vecm_cov$lrprc[vecm_cov$period == "2031Q1"]

cat("Raw lrprc values (log real house price):\n")
cat(sprintf("  2025Q4 anchor            : %.15f\n", lrprc_2025q4))
cat(sprintf("  2029Q1 OBR               : %.15f\n", lrprc_obr_2029q1))
cat(sprintf("  2031Q1 OBR               : %.15f\n", lrprc_obr_2031q1))
cat(sprintf("  2026Q1 VECM-implied      : %.15f\n", lrprc_vecm_2026q1))
cat(sprintf("  2031Q1 VECM-implied      : %.15f\n\n", lrprc_vecm_2031q1))

pct_change   <- function(end, start) (exp(end - start) - 1) * 100
logpt_change <- function(end, start) (end - start) * 100
annualized   <- function(end, start, n_q) (exp((end - start) / (n_q / 4)) - 1) * 100

# ---- OBR path, anchored at 2025Q4 -----------------------------------------
obr_2029_true  <- pct_change(lrprc_obr_2029q1, lrprc_2025q4)
obr_2029_logpt <- logpt_change(lrprc_obr_2029q1, lrprc_2025q4)
obr_2029_ann   <- annualized(lrprc_obr_2029q1, lrprc_2025q4, 13)

obr_2031_true  <- pct_change(lrprc_obr_2031q1, lrprc_2025q4)
obr_2031_logpt <- logpt_change(lrprc_obr_2031q1, lrprc_2025q4)
obr_2031_ann   <- annualized(lrprc_obr_2031q1, lrprc_2025q4, 21)

# ---- VECM-implied path, SAME anchor (2025Q4) for a like-for-like compare --
vecm_2031_true  <- pct_change(lrprc_vecm_2031q1, lrprc_2025q4)
vecm_2031_logpt <- logpt_change(lrprc_vecm_2031q1, lrprc_2025q4)
vecm_2031_ann   <- annualized(lrprc_vecm_2031q1, lrprc_2025q4, 21)

# ---- Diagnostic only: VECM path anchored at its OWN first forecast quarter
# (2026Q1) rather than the actual 2025Q4 anchor - included because it explains
# where "6.72%" in the draft notes actually came from (see printed note below).
vecm_2031_true_from2026 <- pct_change(lrprc_vecm_2031q1, lrprc_vecm_2026q1)

cat("=== OBR real house price path (2025Q4 anchor) ===\n")
cat(sprintf("2025Q4 -> 2029Q1 (FY2028/29, 13q): true %% = %.4f%%   log-pt %% = %.4f%%   annualized = %.4f%%/yr\n",
            obr_2029_true, obr_2029_logpt, obr_2029_ann))
cat(sprintf("2025Q4 -> 2031Q1          (21q): true %% = %.4f%%   log-pt %% = %.4f%%   annualized = %.4f%%/yr\n\n",
            obr_2031_true, obr_2031_logpt, obr_2031_ann))

cat("=== VECM-endogenous real house price path (2025Q4 anchor, R/10_foreVECM.R) ===\n")
cat(sprintf("2025Q4 -> 2031Q1          (21q): true %% = %.4f%%   log-pt %% = %.4f%%   annualized = %.4f%%/yr\n\n",
            vecm_2031_true, vecm_2031_logpt, vecm_2031_ann))

cat("=== Diagnostic: where does 6.72% in the draft notes come from? ===\n")
cat(sprintf("VECM 2026Q1 (first forecast qtr) -> 2031Q1 (20q), true %% = %.4f%%\n",
            vecm_2031_true_from2026))
cat("This uses the VECM's own first forecast quarter as the base, one quarter\n")
cat("later than the 2025Q4 actual anchor used for the OBR figures above - not\n")
cat("the same basis as the 2025Q4-anchored comparison the limitations section needs.\n\n")

cat("=== Replacement for the exec-summary draft figures ===\n")
cat(sprintf("\"2.89%%\" -> OBR  2025Q4->2029Q1 true %%   = %.2f%%  (MATCHES, correct as-is)\n", obr_2029_true))
cat(sprintf("\"2.85%%\" -> OBR  2025Q4->2029Q1 log-pt %% = %.2f%%  (MATCHES, correct as-is)\n", obr_2029_logpt))
cat(sprintf("\"3.72%%\" -> OBR  2025Q4->2031Q1 true %%   = %.2f%%  (MATCHES, correct as-is)\n", obr_2031_true))
cat(sprintf("\"6.72%%\" -> VECM 2025Q4->2031Q1 true %%   = %.2f%%  (DOES NOT MATCH - wrong anchor;\n", vecm_2031_true))
cat(sprintf("           6.72%% is actually VECM 2026Q1->2031Q1 = %.2f%%, a different base quarter)\n",
            vecm_2031_true_from2026))
