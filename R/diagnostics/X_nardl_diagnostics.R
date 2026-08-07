source("R/functions/nardl_functions.R")

eng_zoo  <- readRDS("R/models/nardl_eng_zoo.rds")
dum_full <- readRDS("R/models/nardl_dum_full.rds")
out_full <- readRDS("R/models/nardl_fits_full.rds")

nl_vars <- c("lrprc", "lrcc", "r3", "lvol")

# Dummy-set Sensitivity
dummy_variants <- list(
  gfc_only = c("d08Q3"),
  none     = character(0)
)

out_variants <- setNames(
  lapply(names(dummy_variants), function(spec_name) {
    dum <- dummy_variants[[spec_name]]
    setNames(lapply(nl_vars, run_nardl, dum = dum, eng_zoo = eng_zoo), nl_vars)
  }),
  names(dummy_variants)
)

out_all <- c(list(full = out_full), out_variants)

summary_tbl <- summarize_specs(out_all)
cat("== dummy-set sensitivity ==\n")
print_short_sum(summary_tbl)

# Residual diagnostics
diag_tbl <- residual_diag_table(out_all)
cat("\n== residual diagnostics per NARDL screen (all dummy specs) ==\n")
print(diag_tbl, row.names = FALSE, digits = 3)

# Lag order sensitivity
cat("\n== lag-order sensitivity (full spec) ==\n")
for (nm in nl_vars) {
  cat("\n==", nm, "==\n")
  for (ml in c(5, 6, 7, 8, 10)) {
    z <- run_nardl(nm, dum = dum_full, eng_zoo = eng_zoo, max_lag = ml)
    cat("max_lag=", ml, " order=", paste(z$order, collapse = ","),
        " LR_p=", round(z$LR_p, 3),
        " SRadd_p=", round(z$SRadd_p, 3), "\n", sep = "")
  }
}

saveRDS(out_all, "R/models/nardl_fits_dummy_sensitivity.rds")
write.csv(summary_tbl, "R/models/nardl_summary_dummy_sensitivity.csv", row.names = FALSE)
write.csv(diag_tbl, "R/models/nardl_residual_diagnostics.csv", row.names = FALSE)