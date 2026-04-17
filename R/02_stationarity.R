library(readr)
library(dplyr)
library(urca)
library(tseries)

df <- read_csv("data/processed/quarterly_master.csv") |>
  mutate(date = as.Date(date))

vars <- c("lhstarts", "lrprc", "lvol", "base_rate", "lstock", "lrcc")
df_v <- df |> filter(complete.cases(across(all_of(vars))))

level_type <- c(
  lhstarts  = "trend",
  lrprc     = "trend",
  lvol      = "trend",
  base_rate = "drift",
  lstock    = "trend",
  lrcc      = "trend"
)

# === LEVELS ===
for (v in vars) {
  x <- df_v[[v]]
  cat("\n---", v, "---\n")
  summary(ur.df(x, type = level_type[[v]], selectlags = "BIC")) |> print()
  kpss.test(x, null = ifelse(level_type[[v]] == "trend", "Trend", "Level")) |> print()
  summary(ur.za(x, model = "both")) |> print()
}

# === FIRST DIFFERENCES ===
for (v in vars) {
  x <- diff(df_v[[v]])
  cat("\n--- D.", v, "---\n")
  summary(ur.df(x, type = "drift", selectlags = "BIC")) |> print()
  kpss.test(x, null = "Level") |> print()
}