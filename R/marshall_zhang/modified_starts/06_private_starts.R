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
rdd_bw_ik(ps_dat, kernel = "Uniform")

# dropping sales: plausibly downstream of private starts once it is the outcome
covariates_nosales <- covariates %>% select(-per_1000_sales_01)
ps_cov2 <- my_fuzzy_cov(cov_df, cov_df$per_1000_private_starts,
                        cov_df$afford_gap_median, 50,
                        cov_df$funded_binary, covariates_nosales)
summary_robust(ps_cov2$mod)

# functional forms
ps_lin <- fuzzy_rd_fn(ps_dat, form = "linear")
ps_int <- fuzzy_rd_fn(ps_dat, form = "interaction")
summary_robust(ps_lin)
summary_robust(ps_int)

mods <- list(no_covs = ps_mod, covs = ps_cov$mod, covs_nosales = ps_cov2$mod,
             linear = ps_lin, interaction = ps_int)

sapply(mods, function(m) {
  ct <- coeftest(m, vcov. = sandwich::vcovHC, type = "HC0")
  c(D = ct["D","Estimate"], se = ct["D","Std. Error"], p = ct["D","Pr(>|t|)"],
    lo = ct["D","Estimate"] - 1.96*ct["D","Std. Error"],
    hi = ct["D","Estimate"] + 1.96*ct["D","Std. Error"])
}) %>% round(3) %>% t()