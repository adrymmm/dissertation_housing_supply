library(ARDL)

# Convert ts to data frames and add seasonal dummies
to_ardl_df <- function(tf) {
  df <- as.data.frame(tf)
  qq <- cycle(tf)
  df$d1 <- as.numeric(qq == 1)
  df$d2 <- as.numeric(qq == 2)
  df$d3 <- as.numeric(qq == 3)
  df
}

uk_df  <- to_ardl_df(uk_tf)
eng_df <- to_ardl_df(eng_tf)

run_ardl <- function(df, label) {
  cat("\n===", label, "===\n")
  m <- auto_ardl(
    lhstarts ~ lrprc + lvol + r3 + lstock + lrcc | d1 + d2 + d3,
    data      = df,
    max_order = 8,       # applies only to the 6 non-fixed variables
    selection = "AIC"
  )
  cat("Best order:", m$best_order, "\n")
  
  cat("\nBounds F-test:\n")
  print(bounds_f_test(m$best_model, case = 3))
  cat("\nBounds t-test:\n")
  print(bounds_t_test(m$best_model, case = 3))
  
  invisible(m)
}

ardl_uk  <- run_ardl(uk_df,  "UK ARDL")
ardl_eng <- run_ardl(eng_df, "England ARDL")

cat("=== UK long-run coefficients ===\n")
print(multipliers(ardl_uk$best_model))

cat("\n=== England long-run coefficients ===\n")
print(multipliers(ardl_eng$best_model))
