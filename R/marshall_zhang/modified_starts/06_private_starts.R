library(dplyr); library(purrr); library(ggplot2); library(tibble)
library(rddtools); library(AER)

load("../original/working/rdata/dat_1920.Rdata")
load("../original/working/rdata/sr_df.Rdata")
source("../original/03_rdd_functions.R")

ps_dat <- rdd_data(
  y = per_1000_private_starts,
  x = afford_gap_median,
  z = dat_1920$funded_binary,
  data = dat_1920,
  cutpoint = 50
)

cov_df <- dat_1920 %>%
  select(afford_gap_median, per_1000_private_starts, funded_binary,
         earnings_01, households_01, household_change_01,
         per_1000_sales_01, social_rent_pct_01,
         pro_fin_pct_01, over_65_pct_01) %>%
  na.omit()

# private starts dropped from covariates (it is the outcome), following their
# earnings placebo convention in 05
covariates <- cov_df %>% select(earnings_01:over_65_pct_01)

ps_itt <- itt_mod(ps_dat)
ps_mod <- my_fuzzy_rd(ps_dat)
ps_cov <- my_fuzzy_cov(cov_df, cov_df$per_1000_private_starts,
                       cov_df$afford_gap_median, 50,
                       cov_df$funded_binary, covariates)

summary_robust(ps_mod)
summary_robust(ps_cov$mod)

# bandwidths differ by spec: the no-covariate specs select IK on ps_dat (all
# non-missing rows), the covariate specs select it on the na.omit'd frame built
# inside my_fuzzy_cov, so they are not the same number
bw_nocov <- rdd_bw_ik(ps_dat, kernel = "Uniform")
cat(sprintf("IK bandwidth, no-covariate specs : %.4f\n", bw_nocov))
cat(sprintf("IK bandwidth, covariate specs    : %.4f\n", ps_cov$bw))

# dropping sales: plausibly downstream of private starts once it is the outcome
covariates_nosales <- covariates %>% select(-per_1000_sales_01)
ps_cov2 <- my_fuzzy_cov(cov_df, cov_df$per_1000_private_starts,
                        cov_df$afford_gap_median, 50,
                        cov_df$funded_binary, covariates_nosales)
summary_robust(ps_cov2$mod)

# first stage / ITT: the fuzzy LATE is the reduced form divided by this jump,
# so a weak first stage inflates both the point estimate and its standard error
ps_itt_ct <- coeftest(ps_itt, vcov. = sandwich::vcovHC, type = "HC0")
cat(sprintf("\nfirst stage (ITT on D), coef on ins: %.4f (se %.4f, F %.2f, p %.4f)\n",
            ps_itt_ct["ins","Estimate"], ps_itt_ct["ins","Std. Error"],
            (ps_itt_ct["ins","Estimate"] / ps_itt_ct["ins","Std. Error"])^2,
            ps_itt_ct["ins","Pr(>|t|)"]))

# functional forms
ps_lin <- fuzzy_rd_fn(ps_dat, form = "linear")
ps_int <- fuzzy_rd_fn(ps_dat, form = "interaction")
summary_robust(ps_lin)
summary_robust(ps_int)

mods <- list(no_covs = ps_mod, covs = ps_cov$mod, covs_nosales = ps_cov2$mod,
             linear = ps_lin, interaction = ps_int)

# per-spec metadata: bandwidth actually used, and whether the contemporaneous
# sales covariate is in the covariate set (labelled here so it is visible in the
# output rather than only by reading the covariate list above)
spec_meta <- tibble(
  spec  = names(mods),
  form  = c("quadratic", "quadratic", "quadratic", "linear", "interaction"),
  sales_cov = c("-- (no covariates)", "included", "excluded",
                "-- (no covariates)", "-- (no covariates)"),
  bw = c(bw_nocov, ps_cov$bw, ps_cov2$bw, bw_nocov, bw_nocov)
)

# first stage matched to each spec's own functional form, bandwidth weights and
# covariate set, rather than one shared number
first_stage <- function(m) {
  mf <- model.frame(m)
  w  <- model.weights(mf)
  mf <- mf[, setdiff(names(mf), "(weights)"), drop = FALSE]
  f  <- reformulate(sprintf("`%s`", setdiff(names(mf), c("y", "D"))), response = "D")
  ct <- coeftest(lm(f, data = mf, weights = w),
                 vcov. = sandwich::vcovHC, type = "HC0")
  c(fs_coef = ct["ins","Estimate"], fs_se = ct["ins","Std. Error"],
    fs_F = (ct["ins","Estimate"] / ct["ins","Std. Error"])^2)
}

# CIs use the same t(df) reference distribution as the p-values reported by
# coeftest, rather than mixing a normal 1.96 critical value with t p-values
ps_table <- bind_cols(
  spec_meta,
  bind_rows(lapply(mods, function(m) {
    ct   <- coeftest(m, vcov. = sandwich::vcovHC, type = "HC0")
    dfr  <- df.residual(m)
    tcrit <- qt(0.975, dfr)
    fs   <- first_stage(m)
    tibble(
      D  = ct["D","Estimate"], se = ct["D","Std. Error"],
      df = dfr, p = ct["D","Pr(>|t|)"],
      lo = ct["D","Estimate"] - tcrit * ct["D","Std. Error"],
      hi = ct["D","Estimate"] + tcrit * ct["D","Std. Error"],
      fs_coef = fs["fs_coef"], fs_se = fs["fs_se"], fs_F = fs["fs_F"]
    )
  }))
) %>%
  mutate(across(c(bw, D, se, p, lo, hi, fs_coef, fs_se, fs_F), ~ round(.x, 3)))

print(as.data.frame(ps_table), row.names = FALSE)
