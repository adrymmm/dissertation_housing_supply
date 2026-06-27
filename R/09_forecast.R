library(ARDL)
library(zoo)

model <- readRDS("R/models/ardl_best.rds")
obr <- read.csv("data/python_master/OBR/obr_scenario.csv")

# Dropping overlap row
obr <- obr[obr$period > "2025Q4", ]
obr$d08Q3 <- 0; obr$d20Q2 <- 0; obr$d20Q3 <- 0; obr$d23Q2 <- 0   # no future shocks

# Last in-sample lrcc level
eng <- as.data.frame(readRDS("R/models/eng_tf.rds"))
lrcc_last <- as.numeric(tail(eng$lrcc, 1))

# ARDL forecast
ardl_model <- readRDS("R/models/ardl_best.rds")
ardl_nd <- obr[, c("lrprc","lvol","r3","lrcc","d08Q3","d20Q2","d20Q3","d23Q2")]
ardl_nd_zoo <- zooreg(ardl_nd, start = as.yearqtr("2026 Q1"), frequency = 4)
ardl_fc <- predict(ardl_model, ardl_nd_zoo)

# NARDL forecast
nardl_model <- readRDS("R/models/nardl_lrcc.rds")$fit
d_in <- nardl_model$data
pos_last <- as.numeric(tail(d_in[, "lrcc_pos"], 1))
neg_last <- as.numeric(tail(d_in[, "lrcc_neg"], 1))

# OBR lrcc growth into partial sums
dlrcc <- diff(c(lrcc_last, obr$lrcc))
obr$lrcc_pos <- pos_last + cumsum(pmax(dlrcc, 0))
obr$lrcc_neg <- neg_last + cumsum(pmin(dlrcc, 0))


need <- setdiff(colnames(nardl_model$data), "lhstarts")
nardl_nd <- obr[, need]
nardl_nd_zoo <- zooreg(nardl_nd, start = as.yearqtr("2026 Q1"), frequency = 4)
nardl_fc <- predict(nardl_model, nardl_nd_zoo)

out <- data.frame(
  period      = obr$period,
  ardl_log    = ardl_fc,
  ardl_starts = exp(ardl_fc),
  nardl_log   = nardl_fc,
  nardl_starts = exp(nardl_fc)
)

write.csv(out, "data/outputs/forecasts/obr_scenario_forecasts.csv", row.names = FALSE)