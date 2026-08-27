library(zoo)
library(ggplot2)

g <- readRDS("R/models/mrf_gtvp.rds")
eng_zoo <- readRDS("R/models/nardl_eng_zoo.rds")
d <- as.Date(tail(as.yearqtr(as.numeric(zoo::index(eng_zoo))), nrow(g$betas)))
FIG <- "data/outputs/figures"
j <- match("margin", colnames(g$betas))

theme_set(theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold")))

# the elasticity path
band <- apply(g$betas.draws[, j, ], 1, quantile, probs = c(0.05, 0.95), na.rm = TRUE)
B <- data.frame(date = d, beta = unname(g$betas[, j]),
                lo = band[1, ], hi = band[2, ])
p1 <- ggplot(B, aes(date, beta)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = "grey80") +
  geom_hline(yintercept = 0, linetype = 2, linewidth = 0.3) +
  geom_line(linewidth = 0.7) +
  labs(title = "Elasticity of housing starts to the price-cost margin",
       subtitle = "MRF posterior mean, 90% credible band", x = NULL, y = NULL)
ggsave(file.path(FIG, "gtvp_elasticities.png"), p1, width = 8, height = 4.5, dpi = 200)

# held-out vs raw betas -- the look-ahead a full-sample two-step cannot avoid
R <- rbind(data.frame(date = d, beta = unname(g$betas[, j]), src = "Held out"),
           data.frame(date = d, beta = unname(g$betas.raw[, j]), src = "Uses own observation"))
p2 <- ggplot(R, aes(date, beta, colour = src)) +
  geom_hline(yintercept = 0, linetype = 2, linewidth = 0.3) +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(values = c("black", "firebrick")) +
  labs(title = "Margin elasticity: held-out vs contaminated estimate",
       x = NULL, y = NULL, colour = NULL) +
  theme(legend.position = "bottom")
ggsave(file.path(FIG, "gtvp_trespassing.png"), p2, width = 8, height = 4, dpi = 200)

cat("Wrote gtvp_elasticities.png, gtvp_trespassing.png\n")
