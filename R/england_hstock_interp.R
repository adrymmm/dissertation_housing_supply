# disagg_stock.R
# Fernandez (1981) quarterly disaggregation of annual England private stock
# Indicator: quarterly England private enterprise starts (ONS Table 1b + Meen GB×0.846 pre-1978)

library(tempdisagg)

# ============================================================================
# 1. ANNUAL STOCK (raw dwellings, Dec-stamped per LT104+ONS2023 splice)
# ============================================================================
stock_annual_raw <- "12051000
12138000
11998000
12120000
12242000
12344000
12530000
12825000
13089000
13328000
13552000
13786000
14163000
14496000
14820000
15077000
14997309
15194359
15359543
15545489
15738503
15915360
16104104
16307777
16493966
16688629
16868000
17043264
17318326
17564920
17820181
18037267
18274518
18510062
18672898
18807238
18932749
19045770
19167254
19313881
19481091
19662210
19883579
20125826
20353524
20588296
20788901
21011385
21222365
21412048"

annual <- data.frame(
  year  = 1975:2024,
  stock = as.numeric(strsplit(gsub("[ ,]", "", stock_annual_raw), "\n")[[1]]) / 1000
)
stopifnot(nrow(annual) == 50)

# ============================================================================
# 2. QUARTERLY STARTS (England, private enterprise; Meen×0.846 pre-1978)
# ============================================================================
starts_raw <- "24365
35194
35024
31640
31471
39339
36209
23773
22165
32233
33079
26564
25880
39760
34490
33450
19060
33300
34260
34510
20470
25210
20120
17500
22790
27720
27830
21270
27730
33340
33340
28050
34450
41890
38600
32900
31910
37980
34830
28690
28470
39720
39110
33810
30860
44470
43910
34750
39030
44760
47370
40040
45040
54280
49640
44520
40220
41780
33390
26070
29470
30440
28310
24510
25700
32500
31460
24650
26940
28910
24510
19230
29110
31560
30130
25660
32590
36980
33470
28360
29220
33040
27280
20870
26960
31590
33100
29940
35320
35360
34340
31060
35530
34770
34240
27270
32760
34240
33540
29750
35210
34970
32830
25460
33020
35720
35390
29190
35070
34840
36650
29410
37790
37580
36540
33480
38210
43020
41310
34610
35380
40820
39940
36660
42900
38900
36420
30990
43040
41470
40440
34950
29290
25400
15670
12010
12490
15360
20160
16990
21260
24930
22400
16270
21120
23390
23660
19620
20630
19820
22230
17580
22350
26420
28370
21680
30870
30170
29220
21530
33630
30450
30810
25700
30500
33490
35870
28300
34940
35730
34020
30610
34990
34720
39380
29100
31900
31830
35350
23980
28050
13560
31460
29380
37160
37720
38530
28770
35670
46910
35310
24880
28840
53690
17530
12800
17100
18690"

starts_vec <- as.numeric(strsplit(gsub("[ ,]", "", starts_raw), "\n")[[1]])
cat("Starts values loaded:", length(starts_vec), "\n")

# Build quarterly data.frame assuming sequential from 1975 Q1
n <- length(starts_vec)
starts <- data.frame(
  year    = rep(1975:(1975 + ceiling(n/4) - 1), each = 4)[1:n],
  quarter = rep(1:4, length.out = n),
  starts  = starts_vec
)
cat("Starts range:", starts$year[1], "Q", starts$quarter[1],
    "to", starts$year[n], "Q", starts$quarter[n], "\n")

# ============================================================================
# 3. ALIGN: keep only years with complete 4 quarters of indicator
#    AND a matching annual stock observation
# ============================================================================
complete_yrs <- as.integer(names(table(starts$year)[table(starts$year) == 4]))
yrs <- intersect(annual$year, complete_yrs)
y1  <- min(yrs); yN <- max(yrs)
cat("Disagg sample:", y1, "to", yN, "(", length(yrs), "years )\n")

annual_ts <- ts(annual$stock[annual$year %in% yrs],
                start = y1, frequency = 1)

starts_sub <- starts[starts$year %in% yrs, ]
starts_sub <- starts_sub[order(starts_sub$year, starts_sub$quarter), ]
qind_ts <- ts(starts_sub$starts, start = c(y1, 1), frequency = 4)

cat("annual_ts: length", length(annual_ts), "\n")
cat("qind_ts:   length", length(qind_ts), "( expected", 4*length(yrs), ")\n")
stopifnot(length(qind_ts) == 4 * length(annual_ts))

# ============================================================================
# 4. FERNANDEZ DISAGGREGATION
#    conversion = "last": annual obs = Q4 of that year (Dec stamp)
#    method = "fernandez": Chow-Lin variant w/ RW residuals, for I(1) targets
# ============================================================================
fit <- td(annual_ts ~ qind_ts,
          conversion = "last",
          method     = "fernandez")
print(summary(fit))

stock_q <- predict(fit)

# ============================================================================
# 5. OUTPUT
# ============================================================================
out <- data.frame(
  year    = floor(time(stock_q) + 1e-8),
  quarter = cycle(stock_q),
  stock   = round(as.numeric(stock_q), 3)
)
write.csv(out, "england_private_stock_quarterly.csv", row.names = FALSE)
cat("\nWritten:", nrow(out), "rows to england_private_stock_quarterly.csv\n")

cat("\n=== COPY-PASTE INTO SPREADSHEET (col S) ===\n")
for (i in seq_len(nrow(out))) cat(out$stock[i], "\n", sep="")

# ============================================================================
# 6. SANITY CHECK — Q4 of disagg should EXACTLY match annual anchor
# ============================================================================
cat("\n=== Q4 of disagg vs annual anchor (diff should be ~0) ===\n")
q4 <- out[out$quarter == 4, ]
q4$annual_orig <- annual$stock[match(q4$year, annual$year)]
q4$diff <- q4$stock - q4$annual_orig
print(q4)
cat("\nMax abs diff:", max(abs(q4$diff)), "thousand dwellings\n")

# ============================================================================
# 7. DIAGNOSTIC: variance of diff(log(stock)) — should be MUCH bigger than PCHIP
# ============================================================================
cat("\n=== Variance of diff(log(stock_q)) ===\n")
dl <- diff(log(out$stock))
cat("variance:", var(dl), "\n")
cat("PCHIP comparison: previous variance was ~2.16e-6\n")
cat("If Fernandez >> PCHIP variance, the smoothness problem is fixed.\n")