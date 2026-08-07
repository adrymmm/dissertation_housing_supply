source("R/functions/nardl_functions.R")   # gts, run_nardl, summarize_specs, etc.

eng_tf <- readRDS("R/models/eng_tf.rds")
eng_ts <- ts(eng_tf, start = c(1975, 1), frequency = 4)

ta <- time(eng_ts); dd <- function(yr) which.min(abs(ta - yr))


D <- matrix(0, nrow(eng_ts), 5,
            dimnames = list(NULL, c("d08Q3", "d20Q2", "d20Q3", "d23Q2", "d23Q3")))
D[dd(2008.50), "d08Q3"] <- 1
D[dd(2020.25), "d20Q2"] <- 1
D[dd(2020.50), "d20Q3"] <- 1
D[dd(2023.25), "d23Q2"] <- 1
D[dd(2023.50), "d23Q3"] <- 1

# Seasonals
S <- outer(as.numeric(cycle(eng_ts)), 1:3, "==") - 1/4
colnames(S) <- c("sd1", "sd2", "sd3")

eng_df <- as.data.frame(cbind(as.matrix(eng_tf), D, S))
nl_vars <- c("lrprc", "lrcc", "r3", "lvol")

# Partial sums
for (v in nl_vars) {
  d <- c(0, diff(eng_df[[v]]))
  eng_df[[paste0(v, "_pos")]] <- cumsum(pmax(d, 0))
  eng_df[[paste0(v, "_neg")]] <- cumsum(pmin(d, 0))
}
stopifnot(all(is.finite(as.matrix(eng_df))))

eng_zoo <- as.zooreg(ts(eng_df, start = c(1975, 1), frequency = 4))

# ---- Headline spec: all five dummies -----------------------------------
dum_full <- c("d08Q3", "d20Q2", "d20Q3", "d23Q2", "d23Q3")

out <- list(full = setNames(
  lapply(nl_vars, run_nardl, dum = dum_full, eng_zoo = eng_zoo),
  nl_vars
))

summary_tbl <- summarize_specs(out)
print_short_sum(summary_tbl)

print_full_sum(summary_tbl)

saveRDS(eng_zoo,  "R/models/nardl_eng_zoo.rds")
saveRDS(dum_full, "R/models/nardl_dum_full.rds")
saveRDS(out$full, "R/models/nardl_fits_full.rds")
write.csv(summary_tbl, "R/models/nardl_summary.csv", row.names = FALSE)