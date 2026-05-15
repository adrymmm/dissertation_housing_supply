library(vars)

make_seas_dummies <- function(tf) {
  qq <- cycle(tf)
  cbind(d1 = as.numeric(qq == 1),
        d2 = as.numeric(qq == 2),
        d3 = as.numeric(qq == 3))
}

# Step 1 — lag selection
for (nm in c("uk", "eng")) {
  tf    <- if (nm == "uk") uk_tf else eng_tf
  seas  <- make_seas_dummies(tf)
  vs    <- VARselect(tf, lag.max = 8, type = "const", exogen = seas)
  cat(nm, "VARselect:\n"); print(vs$selection); cat("\n")
}

# Step 2 — Johansen max-eigenvalue
run_johansen <- function(tf, K, label) {
  seas  <- make_seas_dummies(tf)
  jtest <- ca.jo(tf, type = "eigen", ecdet = "const", K = K, dumvar = seas)
  cat("\n===", label, "| K =", K, "===\n")
  print(summary(jtest))
}

# Run with BIC-selected K — also try K=2 to match Michalis if BIC differs
run_johansen(uk_tf,  K = 2, "UK")
run_johansen(eng_tf, K = 2, "England")
