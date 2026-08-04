library(ARDL)

# IMPORTANT HOUSING STOCK IS DROPPED
eng_tf <- readRDS("R/models/eng_tf.rds")
eng_ts <- ts(eng_tf, start = c(1975, 1), frequency = 4)

# Impulse dummies
ta <- time(eng_ts); dd <- function(yr) which.min(abs(ta - yr))
D <- matrix(0, nrow(eng_ts), 5,
            dimnames = list(NULL, c("d08Q3","d20Q2","d20Q3","d23Q2","d23Q3")))
D[dd(2008.50),1] <- 1
D[dd(2020.25),2] <- 1
D[dd(2020.50),3] <- 1
D[dd(2023.25),4] <- 1
D[dd(2023.50),5] <- 1

# Centred seasonals (matches ca.jo's season=4)
q <- cycle(eng_ts)
S <- outer(as.numeric(q), 1:3, "==") - 1/4
colnames(S) <- c("sd1","sd2","sd3")

eng_zoo <- as.zooreg(ts(cbind(as.matrix(eng_tf), D, S), start = c(1975, 1), frequency = 4))

mod <- auto_ardl(lhstarts ~ lrprc + lvol + r3 + lrcc |
                   d08Q3 + d20Q2 + d20Q3 + d23Q2 + d23Q3 + sd1 + sd2 + sd3,
                 data = eng_zoo, max_order = 6)
ardl_best <- mod$best_model
cat("Selected order:", paste(ardl_best$order, collapse = ","), "\n")
summary(ardl_best)

ardl_ecm <- recm(ardl_best, case = 3)
summary(ardl_ecm)
multipliers(ardl_best)

saveRDS(mod, "R/models/ardl_mod.rds")  # keeps top_orders for the diagnostics script
saveRDS(ardl_best, "R/models/ardl_best.rds")
saveRDS(ardl_ecm, "R/models/ardl_ecm.rds")

# PYTHON EXPORT SECTION
ecm_coefs <- data.frame(
  type = "ecm",
  term = names(coef(ardl_ecm)),
  estimate = coef(ardl_ecm)
)
lr_coefs <- multipliers(ardl_best)
print(lr_coefs$Term)  # confirm the intercept label before trusting the filter below
lr_coefs <- lr_coefs[!grepl("Intercept|Constant", lr_coefs$Term, ignore.case = TRUE), ]
lr_coefs <- data.frame(
  type = "longrun",
  term = lr_coefs$Term,
  estimate = lr_coefs$Estimate
)
all_coefs <- rbind(ecm_coefs, lr_coefs)
write.csv(all_coefs, "R/models/ardl_coefs_full.csv", row.names = FALSE)