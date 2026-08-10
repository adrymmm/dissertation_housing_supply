# Diagnostic Block -Divergence between ARDL and NARDL
out <- read.csv("data/outputs/forecasts/obr_scenario_forecasts.csv")

out$diff_pct <- 100 * (out$nardl_starts - out$ardl_starts) / out$ardl_starts

# quarter-by-quarter comparison
print(out[, c("period", "ardl_starts", "nardl_starts", "diff_pct")])

# summary stats
cat("Mean abs % diff:", mean(abs(out$diff_pct)), "\n")
cat("Max abs % diff:", max(abs(out$diff_pct)), "at", out$period[which.max(abs(out$diff_pct))], "\n")

# does the gap grow over the horizon, or stay flat?
plot(seq_along(out$period), out$diff_pct, type = "b",
     xlab = "quarters ahead", ylab = "% diff (NARDL - ARDL)")