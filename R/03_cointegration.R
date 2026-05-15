library(tsDyn)

run_rank_test <- function(tf, lag, label) {
  seas <- make_centered_seas(tf)
  vc <- VECM(tf, lag = lag, include = "const", LRinclude = "none",
             estim = "ML", exogen = seas)
  cat("\n===", label, "| lag =", lag, "===\n")
  print(rank.test(vc, type = "eigen"))
  print(rank.test(vc, type = "trace"))
}

# Centered seasonals (sum to zero), since tsDyn has no `season` arg
make_centered_seas <- function(tf) {
  qq <- cycle(tf)
  cbind(sd1 = (qq == 1) - 1/4,
        sd2 = (qq == 2) - 1/4,
        sd3 = (qq == 3) - 1/4)
}

run_rank_test(uk_tf,  lag = 1, "UK")
run_rank_test(eng_tf, lag = 1, "England")
