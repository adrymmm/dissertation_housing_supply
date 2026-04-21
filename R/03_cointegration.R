library(readr)
library(dplyr)
library(urca)
library(vars)

vars <- c("lhstarts", "lrprc", "lvol", "base_rate", "lstock", "lrcc")

df_v <- read_csv("data/processed/quarterly_master.csv") |>
  mutate(date = as.Date(date)) |>
  filter(complete.cases(across(all_of(vars)))) |>
  dplyr::select(all_of(vars)) |>
  as.matrix()

pmax <- floor(12 * (nrow(df_v) / 100)^0.25)

varsel <- VARselect(df_v, lag.max = pmax, type = "const")
varsel$selection

K <- varsel$selection["AIC(n)"]

summary(ca.jo(df_v, type = "trace", ecdet = "const", K = K, spec = "longrun")) |> print()
summary(ca.jo(df_v, type = "eigen", ecdet = "const", K = K, spec = "longrun")) |> print()