library(readr)
library(dplyr)
library(urca)
library(tseries)

vars <- c("lhstarts", "lrprc", "lvol", "base_rate", "lstock", "lrcc")

level_type <- c(lhstarts = "trend", lrprc = "trend", lvol = "trend",
                base_rate = "drift", lstock = "trend", lrcc = "trend")

df_v <- read_csv("data/processed/quarterly_master.csv") |>
  mutate(date = as.Date(date)) |>
  filter(complete.cases(across(all_of(vars))))

pmax <- floor(12 * (nrow(df_v) / 100)^0.25)

# === LEVELS ===
for (v in vars) {
  cat("\n---", v, "---\n")
  summary(ur.df(df_v[[v]], type = level_type[[v]], lags = pmax, selectlags = "AIC")) |> print()
  kpss.test(df_v[[v]], null = "Trend") |> print()
  summary(ur.za(df_v[[v]], model = "both")) |> print()
}

# === FIRST DIFFERENCES ===
for (v in vars) {
  cat("\n--- D.", v, "---\n")
  summary(ur.df(diff(df_v[[v]]), type = "drift", lags = pmax, selectlags = "AIC")) |> print()
  kpss.test(diff(df_v[[v]]), null = "Level") |> print()
}