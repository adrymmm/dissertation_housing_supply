library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(zoo)

df <- read_csv("data/python_master/england_master.csv")
colnames(df)[1] <- "Date"
df <- na.omit(df)

# Parsing to date
df$Date <- as.Date(zoo::as.yearqtr(df$Date, format = "%YQ%q"))

df <- df %>%
  mutate(
    lrprc  = log(hprice / cc_def),   # real house prices
    lrcc   = log(cc / cc_def),       # real construction costs
    lhstarts = log(starts),          # starts (no deflation needed)
    lvol   = log(vol),               # transactions volume
    lstock = log(hstock),            # housing stock
    r3     = rate                    # interest rate
  )

# Long format for plot
plot_df <- df[, c("Date", "lhstarts", "lrprc", "lvol", "r3", "lstock", "lrcc")]

# Labels
var_labels <- c(
  lhstarts = "log Starts",
  lrprc    = "log Real House Price",
  lvol     = "log Transactions",
  r3       = "Interest Rate",
  lstock   = "log Housing Stock",
  lrcc     = "log Real Construction Cost"
)

eng_tf <- df %>%
  dplyr::select(lhstarts, lrprc, lvol, r3, lstock, lrcc) %>%
  as.matrix()

plot_df <- df %>%
  dplyr::select(Date, lhstarts, lrprc, lvol, r3, lstock, lrcc) %>%
  pivot_longer(-Date, names_to = "variable", values_to = "value")

# Plotting
ggplot(plot_df, aes(x = Date, y = value)) +
  geom_line(colour = "#2c7bb6", linewidth = 0.5) +
  facet_wrap(~ variable, scales = "free_y", ncol = 2) +
  labs(title = "VECM Variables - England 1975Q1-2025Q4",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(strip.text = element_text(face = "bold"))

# Saving tf and df
saveRDS(eng_tf, "R/models/eng_tf.rds")
saveRDS(df, "R/models/eng_df.rds")
