library(readr)
library(dplyr)

vars <- c("lhstarts", "lrprc", "lvol", "base_rate", "lstock", "lrcc")

df <- read_csv("data/processed/quarterly_master.csv") |>
  mutate(date = as.Date(date)) |>
  filter(date >= as.Date("1987-01-01")) # Starting at earliest date

par(mfrow = c(3, 2), mar = c(2, 4, 2, 1))
for (v in vars) plot(df$date, df[[v]], type = "l", main = v, ylab = "")