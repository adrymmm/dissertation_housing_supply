library(ARDL)
library(zoo)

master <- read.csv("data/python_master/OBR/obr_scenario.csv")
master$period <- as.yearqtr(master$period, format = "%YQ%q")
master$X <- NULL
names(master)[names(master) == "lhprice"] <- "lrprc"

eng_df    <- readRDS("R/models/eng_df.rds")
ardl_best <- readRDS("R/models/ardl_best.rds")

hist   <- data.frame(period = as.yearqtr(eng_df$Date), lhstarts = eng_df$lhstarts)
master <- merge(master, hist, by = "period", all.x = TRUE)
master <- master[order(master$period), ]
rownames(master) <- NULL

# dummies enter contemporaneously only; zero in the forecast period
fc_idx <- which(is.na(master$lhstarts))
master[fc_idx, c("d08Q3","d20Q2","d20Q3","d23Q2")] <- 0

# parse each coef name into (var, lag): "(Intercept)", bare = lag 0, L(var, k)
co <- coef(ardl_best)
trm <- lapply(names(co), function(nm) {
  if (nm == "(Intercept)") return(list(var = "(Intercept)", lag = 0L))
  m <- regmatches(nm, regexec("^L\\(\\s*([^,]+),\\s*([0-9]+)\\s*\\)$", nm))[[1]]
  if (length(m) == 3) list(var = trimws(m[2]), lag = as.integer(m[3]))
  else                list(var = nm, lag = 0L)
})

# recursive forecast: write each yhat back so later lags pick it up
for (t in fc_idx) {
  yhat <- 0
  for (k in seq_along(co)) {
    v <- trm[[k]]$var; lg <- trm[[k]]$lag
    if (v == "(Intercept)") { yhat <- yhat + co[k]; next }
    yhat <- yhat + co[k] * master[[v]][t - lg]
  }
  master$lhstarts[t] <- yhat
}

fc_ardl <- master$lhstarts[fc_idx]

# the forecast values, with dates attached
master[fc_idx, c("period", "lhstarts")]

# back out of logs to actual starts (per quarter)
data.frame(period = master$period[fc_idx],
           starts = exp(master$lhstarts[fc_idx]))

# annual totals (sum of quarterly levels)
aggregate(exp(master$lhstarts[fc_idx]),
          by = list(year = floor(as.numeric(master$period[fc_idx]))),
          FUN = sum)


## NEED TO DROP LSTOCK OR IDK 