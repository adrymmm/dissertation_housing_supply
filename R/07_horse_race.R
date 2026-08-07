library(urca)
library(vars)
library(tsibble)
library(fable)
library(dplyr)
library(tidyr)
library(ARDL)

eng_df   <- readRDS("R/models/eng_df.rds")
ardl_best <- readRDS("R/models/ardl_best.rds")

# Core vars to be used in johansen loop
eng_core <- as.matrix(eng_df[, c("lhstarts","lrprc","lvol","r3","lstock","lrcc")])
  
# CONFIG
# K, R_RANK, ord_* fixed from full sample; coefficients re-estimated per origin
y_name    <- "lhstarts"
K         <- 5  # VECM Lag order
R_RANK    <- 1  # VECM cointegration rank
nardl_var <- "lrcc"

eval0 <- which(yearquarter(eng_df$Date) == yearquarter("2010 Q1"))
stopifnot(length(eval0) == 1)

n      <- nrow(eng_df)
y      <- eng_df[[y_name]]

# Tsibble for fable  
eng_tsbl <- eng_df %>%
  transmute(Date = yearquarter(Date), lhstarts) %>%
  as_tsibble(index = Date)

# Range over scored forecasts - origin is cutoff target = origin + 1 i.e. target-1
origin_set <- (eval0 - 1):(n - 1)

cat(sprintf("Origins %d..%d  ->  targets %s .. %s\n",
            min(origin_set), max(origin_set),
            as.character(yearquarter(eng_df$Date[min(origin_set) + 1])),
            as.character(yearquarter(eng_df$Date[max(origin_set) + 1]))))

# ARDL best lag orders
ord <- ardl_best$order
py  <- ord[[1]]    # lag order of lhstarts
qx  <- ord[-1]     # lag orders of x vector

# Form ARDL formula lhstarts ~ lrprc + lvol + r3 + lstock + lrcc
ardl_frm <- reformulate(names(qx), response = y_name)
# Returns full numeric vector of lag order c(4,4,4,4,4,4)
ord_ardl <- c(py, unname(qx[names(qx)]))

# NARDL SECTION

# NARDL: decompose nardl_var into partial sums of positive / negative changes.
# cumsum to row o depends only on rows 1:o, so computing on the full series
# before the loop leaks nothing.
d    <- c(NA, diff(eng_df[[nardl_var]]))
pos  <- cumsum(ifelse(is.na(d), 0, pmax(d, 0)))
neg  <- cumsum(ifelse(is.na(d), 0, pmin(d, 0)))
rest <- setdiff(names(qx), nardl_var)
pos_name <- paste0(nardl_var, "_pos")
neg_name <- paste0(nardl_var, "_neg")
qx_n <- setNames(c(qx[[nardl_var]], qx[[nardl_var]], qx[rest]),
                 c(pos_name, neg_name, rest))
nardl_df <- eng_df
nardl_df[[pos_name]] <- pos
nardl_df[[neg_name]] <- neg

# Creates formula lh_starts ~ nardl_pos + nardl_neg + ...
nardl_frm <- reformulate(names(qx_n), response = y_name)
ord_nardl <- c(py, unname(qx_n[names(qx_n)]))

# Unconditional Forecasts
uncond_fc <- eng_tsbl %>%
  filter(row_number() <= max(origin_set)) %>%
  stretch_tsibble(.init = min(origin_set), .step = 1, .id = "origin_id") %>%
  model(rw     = RW(lhstarts),
        snaive = SNAIVE(lhstarts ~ lag("year")),
        ar     = AR(lhstarts ~ order(p = 0:8), ic = "aic"),
        tslm   = TSLM(lhstarts ~ trend()),
        tslm_s = TSLM(lhstarts ~ trend() + season())) %>%
  fabletools::forecast(h = 1) %>%
  as_tibble() %>%
  mutate(origin = min(origin_set) - 1L + origin_id) %>%
  dplyr::select(origin, model = .model, fc = .mean) %>%
  pivot_wider(names_from = model, values_from = fc)
 

# VECM, ARDL, NARDL forecast
# newdata = row o+1 => ARDL/NARDL conditional on realised covariates; VECM/naives are not
cond_fc_tab <- do.call(rbind, lapply(origin_set, function(o) {
  m_ardl  <- ardl(ardl_frm,  data = eng_df[1:o, ],   order = ord_ardl)
  m_nardl <- ardl(nardl_frm, data = nardl_df[1:o, ], order = ord_nardl)
  jo <- ca.jo(eng_core[1:o, ], type = "trace", ecdet = "none", K = K,
              spec = "transitory", season = 4)
  data.frame(
    origin = o,
    ardl   = unname(predict(m_ardl,
               newdata = eng_df[o + 1, names(qx), drop = FALSE])),
    nardl  = unname(predict(m_nardl,
               newdata = nardl_df[o + 1, names(qx_n), drop = FALSE])),
    vecm   = predict(vec2var(jo, r = R_RANK),
                     n.ahead = 1)$fcst[[y_name]][1, "fcst"]
  )
}))

 
fc_all <- full_join(uncond_fc, cond_fc_tab, by = "origin")
mcols  <- c("rw", "snaive", "ar", "tslm", "tslm_s", "vecm", "ardl", "nardl")
 
res <- fc_all %>%
  mutate(target = origin + 1, actual = y[target], date = eng_df$Date[target]) %>%
  arrange(target) %>%
  dplyr::select(target, date, actual, all_of(mcols))
 
write.csv(res, "data/outputs/forecasts/h1_forecasts.csv", row.names = FALSE)