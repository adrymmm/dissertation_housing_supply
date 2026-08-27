# MacroRF (vendored)

Macroeconomic Random Forest -- Philippe Goulet Coulombe.
RF-generated generalized time-varying parameters for a linear equation.

- Upstream: https://github.com/philgoucou/macrorf (branch `main`)
- Package version: 0.1.1, source file `MRF_v210403.R` (2021-04-03)
- Retrieved: 2026-08-26
- sha256(MRF_v210403.R): `e052c0f12544a2bfb79974c07174b08116b7c80298de5905bdb3753732a7b631`
- Paper: Goulet Coulombe, "The Macroeconomy as a Random Forest", https://arxiv.org/abs/2006.12724
- License: GPL-3 (see DESCRIPTION)

Vendored rather than installed because upstream is a prototype that is not on
CRAN; pinning the file keeps the thesis reproducible. Its only dependency is
`pracma` (for `repmat`), so `R/07_MacroRF.R` sources the file directly
instead of building the package.
