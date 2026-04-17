library(readr)
library(dplyr)

df <- read_csv("data/processed/quarterly_master.csv") |>
  mutate(date = as.Date(date))

par(mfrow = c(3, 2), mar = c(2, 4, 2, 1))

plot(df$date, df$lhstarts,  type = "l", main = "lhstarts",  ylab = "")
plot(df$date, df$lrprc,     type = "l", main = "lrprc",     ylab = "")
plot(df$date, df$lvol,      type = "l", main = "lvol",      ylab = "")
plot(df$date, df$lrcc,      type = "l", main = "lrcc",      ylab = "")
plot(df$date, df$lstock,    type = "l", main = "lstock",    ylab = "")
plot(df$date, df$base_rate, type = "l", main = "base_rate", ylab = "")
