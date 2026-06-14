library(ARDL)
library(car)
library(sandwich)

eng_df <- as.data.frame(eng_tf)

# Creating impulse dummies
ta <- time(ts(eng_df, start = c(1975,1), frequency = 4))
dd <- function(yr) which.min(abs(ta - yr))
eng_df$d08Q3 <- 0; eng_df$d08Q3[dd(2008.50)] <- 1
eng_df$d20Q2 <- 0; eng_df$d20Q2[dd(2020.25)] <- 1
eng_df$d20Q3 <- 0; eng_df$d20Q3[dd(2020.50)] <- 1
eng_df$d23Q2 <- 0; eng_df$d23Q2[dd(2023.25)] <- 1

eng_df <- na.omit(eng_df)
eng_df <- eng_df[is.finite(rowSums(eng_df)), ]

# Partial sum decomposition
nl_vars <- c("lrprc", "lrcc", "r3")

for (var in nl_vars) {
  d <- c(0, diff(eng_df[[var]]))
  eng_df[[paste0(var, "_pos")]] <- cumsum(pmax(d, 0))
  eng_df[[paste0(var, "_neg")]] <- cumsum(pmin(d, 0))
}


run_nardl <- function(x) {
  base <- c("lrprc", "lvol", "r3", "lstock", "lrcc")
  rhs  <- c(paste0(x, c("_pos", "_neg")), setdiff(base, x))
  dum  <- c("d08Q3","d20Q2","d20Q3","d23Q2")
  f    <- as.formula(paste("lhstarts ~", paste(rhs, collapse = " + "),
                           "|", paste(dum, collapse = " + ")))
  
  nardl_fit <- auto_ardl(f, data = eng_df, max_order = 4)$best_model
  uecm_fit  <- uecm(nardl_fit)
  V         <- vcovHC(uecm_fit, type = "HC3")
  
  cf <- coef(uecm_fit); k <- length(cf)
  contrast <- function(pos, neg) {
    v <- setNames(numeric(k), names(cf))
    v[pos] <-  1
    v[neg] <- -1
    matrix(v, nrow = 1)
  }
  sym_test <- function(pos, neg) {
    if (!length(pos) || !length(neg) || anyNA(cf[c(pos, neg)])) return(NA)
    linearHypothesis(uecm_fit, contrast(pos, neg), rhs = 0, vcov. = V)
  }
  
  xpos <- paste0(x, "_pos"); xneg <- paste0(x, "_neg")
  allp <- grep(xpos, names(cf), value = TRUE, fixed = TRUE)
  alln <- grep(xneg, names(cf), value = TRUE, fixed = TRUE)
  lr_p <- allp[!grepl("^d\\(", allp)]
  lr_n <- alln[!grepl("^d\\(", alln)]
  sr_p <- grep("^d\\(", allp, value = TRUE)
  sr_n <- grep("^d\\(", alln, value = TRUE)
  
  res <- residuals(nardl_fit)
  list(
    fit      = nardl_fit,
    bounds_f = bounds_f_test(nardl_fit, case = 3),
    bounds_t = bounds_t_test(nardl_fit, case = 3),
    recm     = recm(nardl_fit, case = 3),
    mult     = multipliers(nardl_fit),
    lr_sym   = sym_test(lr_p, lr_n),
    sr_sym   = sym_test(sr_p, sr_n),
    aliased  = names(which(is.na(cf))),
    shapiro  = shapiro.test(res),
    arch     = FinTS::ArchTest(res, lags = 4),
    bg       = lmtest::bgtest(nardl_fit, order = 4)
  )
}

out <- lapply(c("lrprc", "lrcc", "r3"), run_nardl)
names(out) <- c("lrprc", "lrcc", "r3")

# Summary table
cell <- function(z, col) if (inherits(z, "anova")) z[2, col] else NA

sym_tbl <- data.frame(
  var  = names(out),
  lr_F = sapply(out, function(z) cell(z$lr_sym, "F")),
  lr_p = sapply(out, function(z) cell(z$lr_sym, "Pr(>F)")),
  sr_F = sapply(out, function(z) cell(z$sr_sym, "F")),
  sr_p = sapply(out, function(z) cell(z$sr_sym, "Pr(>F)")),
  row.names = NULL
)
sym_tbl
