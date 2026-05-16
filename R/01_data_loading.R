library(tidyr)
library(readr)

col_names <- c("NA", "year", "quarter", "starts", "nom_price", "nom_cc",
              "transactions", "stock", "r3", "deflator_inc", "deflator_cc")

uk <- read_csv("data/processed/UK_master.csv", skip=7, col_names=col_names)
eng <- read_csv("data/processed/england_master.csv", skip=7, col_names=col_names)

uk$`NA` <- NULL
eng$`NA` <- NULL

uk <- fill(uk, year)
eng <- fill(eng, year)

uk <- drop_na(uk)
eng <- drop_na(eng)

uk$quarter <- as.numeric(sub("Q", "", uk$quarter))
eng$quarter <- as.numeric(sub("Q", "", eng$quarter))


uk_ts <- ts(uk[, c("starts", "nom_price", "nom_cc", "transactions",
                   "stock", "r3", "deflator_inc", "deflator_cc")],
            start = c(uk$year[1], uk$quarter[1]), frequency = 4)

eng_ts <- ts(eng[, c("starts", "nom_price", "nom_cc", "transactions",
                     "stock", "r3", "deflator_inc", "deflator_cc")],
             start = c(eng$year[1], eng$quarter[1]), frequency = 4)

# Deflate and log-transform — r3 enters in levels, no log
transform_vecm <- function(x) {
  lhstarts <- log(x[, "starts"] / 1000)
  lrprc    <- log(x[, "nom_price"]   / x[, "deflator_inc"])
  lvol     <- log(x[, "transactions"] / 1000)
  lstock   <- log(x[, "stock"])
  lrcc     <- log(x[, "nom_cc"]      / x[, "deflator_cc"])
  r3       <- x[, "r3"]
  
  ts(cbind(lhstarts, lrprc, lvol, r3, lstock, lrcc),
     start = start(x), frequency = frequency(x))
}

uk_tf  <- transform_vecm(uk_ts)
eng_tf <- transform_vecm(eng_ts)

plot_vecm_vars <- function(tf, main_prefix = "") {
  labels <- c("Housing Starts (log)", "Real House Price Index (log)",
              "Transactions (log)", "Short-term Interest Rate",
              "Housing Stock (log)", "Construction Costs (log)")
  par(mfrow = c(3, 2), mar = c(3, 3, 2, 1))
  for (i in 1:6) {
    plot(tf[, i], main = paste0(main_prefix, labels[i]),
         ylab = "", xlab = "", col = "steelblue", lwd = 1.2)
  }
  par(mfrow = c(1, 1))
}

plot_vecm_vars(uk_tf,  main_prefix = "UK - ")
plot_vecm_vars(eng_tf, main_prefix = "England - ")
