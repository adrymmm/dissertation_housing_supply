library(tsDyn)

vecm_uk  <- VECM(uk_tf,  lag = 1, r = 1, include = "const",
                 LRinclude = "none", estim = "ML",
                 exogen = make_centered_seas(uk_tf))
vecm_eng <- VECM(eng_tf, lag = 1, r = 1, include = "const",
                 LRinclude = "none", estim = "ML",
                 exogen = make_centered_seas(eng_tf))

# Cointegrating vectors — tsDyn stores β with first variable normalised to 1
cat("=== UK cointegrating vector ===\n");      print(coefB(vecm_uk))
cat("\n=== England cointegrating vector ===\n"); print(coefB(vecm_eng))

# α (loadings)
cat("\n=== UK α ===\n");      print(coefA(vecm_uk))
cat("\n=== England α ===\n"); print(coefA(vecm_eng))

# Full coefficient table (per-equation, like cajorls$rlm)
summary(vecm_uk)
summary(vecm_eng)

# Diagnostics
# Convert ca.jo objects to VAR representation for diagnostic tools
# 1. Residuals are available directly
res_uk <- residuals(vecm_uk)

# 2. For Portmanteau / JB, run them on the residual matrix manually
Box.test(res_uk[, "lhstarts"], lag = 16, type = "Ljung-Box")
shapiro.test(res_uk[, "lhstarts"])
