library(nardl)
library(car)

# eng_tf must be a data.frame (not matrix) for nardl
eng_df <- as.data.frame(eng_tf)
eng_df <- na.omit(eng_df)
eng_df <- eng_df[is.finite(rowSums(eng_df)), ]

# ── 1. NARDL: decompose lrprc only, symmetric controls via pipe ───────────────
# Formula: depvar ~ decomposed_var | control1 + control2 + ...
start_time <- Sys.time()
nardl_fit <- nardl(
  lhstarts ~ lrprc | lvol + r3 + lstock + lrcc,
  data   = eng_df,
  ic     = "aic",
  maxlag = 4,       
  graph  = TRUE,
  case   = 3
)

end_time <- Sys.time()
end_time - start_time

summary(nardl_fit)

# Key scalars
# F-bounds statistic (joint null: rho = theta+ = theta- = 0)
nardl_fit$fstat

# ECT t-statistic (null: rho = 0, alt: rho < 0)
nardl_fit$tstat

# Diagnostics
# Normality - Shapiro Test
shapiro.test(nardl_fit$selresidu)

# ARCH test for heteroscedasticity (Using internal selected lags 'nl')
ArchTest(nardl_fit$selresidu, lags = nardl_fit$nl)

# Breusch-Godfrey LM test for serial correlation (Runs on the internal OLS fit '$fits')
bp2(nardl_fit$fits, nlags = nardl_fit$nl, fill = 0, type = "F")
