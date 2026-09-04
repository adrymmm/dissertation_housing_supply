# Sequential Johansen trace-test rank decision at 10%/5%/1%, printed alongside
# the raw trace table so the r=1 call doesn't have to be worked out by hand
# each time (see: gdp_def vs p_def rank stability check).
trace_rank_decision <- function(jo) {
  ts   <- jo@teststat
  cval <- jo@cval
  n    <- length(ts)

  cat("\n--- Sequential rank decision ---\n")
  for (lev in c("10pct", "5pct", "1pct")) {
    idx <- n
    while (idx >= 1 && ts[idx] > cval[idx, lev]) idx <- idx - 1
    r <- if (idx < 1) n else n - idx
    if (idx >= 1) {
      cat(sprintf("  %-5s: rank = %d  (stopped at r <= %d, stat %.2f vs cval %.2f, margin %.2f)\n",
                  lev, r, r, ts[idx], cval[idx, lev], cval[idx, lev] - ts[idx]))
    } else {
      cat(sprintf("  %-5s: rank = %d  (rejected every r <= %d test - full rank, unusual)\n",
                  lev, r, n - 1))
    }
  }
}
