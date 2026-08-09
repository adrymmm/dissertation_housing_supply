library(zoo)

START <- c(1975, 1)
FC_START <- c(2026, 1)
NL_VARS <- c("lrprc", "lrcc", "r3", "lvol")
DUMMY_NAMES <- c("d08Q3", "d20Q2", "d20Q3", "d23Q2", "d23Q3")

# Build the combined history + OBR path

eng_tf <- as.data.frame(readRDS("R/models/eng_tf.rds"))
obr <- read.csv("data/python_master/OBR/obr_scenario.csv")
obr <- obr[obr$period > "2025Q4", ]

cov_vars <- c("lrprc", "lvol", "r3", "lrcc")
stopifnot(all(cov_vars %in% names(obr)))

n_in <- nrow(eng_tf)
n_fc <- nrow(obr)

# lhstarts known in sample, NA over the forecast horizon
hist <- data.frame(
  lhstarts = c(eng_tf$lhstarts, rep(NA_real_, n_fc))
)
for (v in cov_vars) hist[[v]] <- c(eng_tf[[v]], obr[[v]])

# Deterministics over the whole extended index
idx <- ts(rep(0, n_in + n_fc), start = START, frequency = 4)
ta <- time(idx)
dd <- function(yr) which.min(abs(ta - yr))

for (nm in DUMMY_NAMES) hist[[nm]] <- 0
hist$d08Q3[dd(2008.50)] <- 1
hist$d20Q2[dd(2020.25)] <- 1
hist$d20Q3[dd(2020.50)] <- 1
hist$d23Q2[dd(2023.25)] <- 1
hist$d23Q3[dd(2023.50)] <- 1

S <- outer(as.numeric(cycle(idx)), 1:3, "==") - 1 / 4
hist$sd1 <- S[, 1]; hist$sd2 <- S[, 2]; hist$sd3 <- S[, 3]

# Partial sums over the full extended series
for (v in NL_VARS) {
  d <- c(0, diff(hist[[v]]))
  hist[[paste0(v, "_pos")]] <- cumsum(pmax(d, 0))
  hist[[paste0(v, "_neg")]] <- cumsum(pmin(d, 0))
}

stopifnot(all(is.finite(as.matrix(hist[, setdiff(names(hist), "lhstarts")]))))

# Coefficient-driven recursion 

# Parse "L(lrprc, 2)" -> list(var = "lrprc", lag = 2); bare "lrprc" -> lag 0.
parse_term <- function(nm) {
  m <- regmatches(nm, regexec("^L\\(\\s*([^,]+?)\\s*,\\s*(\\d+)\\s*\\)$", nm))[[1]]
  if (length(m) == 3) list(var = m[2], lag = as.integer(m[3]))
  else list(var = nm, lag = 0L)
}

# Recursively project lhstarts
recurse <- function(model, data, first_fc) {
  b <- coef(model)
  terms <- lapply(names(b), parse_term)
  
  needed <- unique(vapply(terms, `[[`, "", "var"))
  needed <- setdiff(needed, c("(Intercept)", "lhstarts"))
  missing <- setdiff(needed, names(data))
  if (length(missing)) {
    stop("newdata is missing model terms: ", paste(missing, collapse = ", "))
  }
  
  y <- data$lhstarts
  for (t in seq(first_fc, nrow(data))) {
    val <- 0
    for (i in seq_along(b)) {
      tm <- terms[[i]]
      x <- if (tm$var == "(Intercept)") 1
      else if (tm$var == "lhstarts") y[t - tm$lag]
      else data[[tm$var]][t - tm$lag]
      val <- val + b[[i]] * x
    }
    y[t] <- val
  }
  y
}

# Backstep one-step check
backtest <- function(model, data, n = 8) {
  fit <- as.numeric(fitted(model))
  last <- n_in
  b <- coef(model); terms <- lapply(names(b), parse_term)
  errs <- numeric(n)
  for (k in seq_len(n)) {
    t <- last - n + k
    val <- 0
    for (i in seq_along(b)) {
      tm <- terms[[i]]
      x <- if (tm$var == "(Intercept)") 1 else data[[tm$var]][t - tm$lag]
      val <- val + b[[i]] * x
    }
    errs[k] <- val - fit[length(fit) - n + k]
  }
  errs
}

first_fc <- n_in + 1

# ARDL
ardl_best <- readRDS("R/models/ardl_best.rds")
cat("ARDL order:", paste(ardl_best$order, collapse = ","), "\n")

ardl_data <- hist
ardl_data$lhstarts <- hist$lhstarts
cat("ARDL backtest errors:", round(backtest(ardl_best, ardl_data), 5), "\n")

ardl_y <- recurse(ardl_best, ardl_data, first_fc)
ardl_fc <- ardl_y[first_fc:length(ardl_y)]

# NARDL
NARDL_VAR <- "lrcc"
nardl_fits <- readRDS("R/models/nardl_fits_full.rds")
stopifnot(NARDL_VAR %in% names(nardl_fits))
nardl_model <- nardl_fits[[NARDL_VAR]]$fit

cat("NARDL backtest errors:", round(backtest(nardl_model, hist), 5), "\n")

nardl_y <- recurse(nardl_model, hist, first_fc)
nardl_fc <- nardl_y[first_fc:length(nardl_y)]

# Output
out <- data.frame(
  period       = obr$period,
  ardl_log     = ardl_fc,
  ardl_starts  = exp(ardl_fc),
  nardl_log    = nardl_fc,
  nardl_starts = exp(nardl_fc)
)

dir.create("data/outputs/forecasts", recursive = TRUE, showWarnings = FALSE)
write.csv(out, "data/outputs/forecasts/obr_scenario_forecasts.csv", row.names = FALSE)
print(out)