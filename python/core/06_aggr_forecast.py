import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
from python.functions.forecast import quarter_str_from_dates, plot_rmse_bar, report, run_ljungbox, run_mcs, run_spa


# IMPORTANT: RUN 08_horse_race.R first for h1_forecasts.csv + h1_model_roles.csv

# --- CONFIG ---
FORECASTS_DIR = ROOT / "data" / "outputs" / "forecasts"
FIGURES_DIR = ROOT / "data" / "outputs" / "figures"

# Column roles, matching the R script:
#   *_rw    RW projections -- unconditional
#   *_dir   h=1 reparameterisation -- unconditional
#   *_rt    lag orders / K / rank re-selected at each origin -- real time
#   *_cond  actual covariates at t+1 -- ex post, conditional
BENCH = ["RW", "SNAIVE", "AR", "TSLM", "TSLM_s"]
ML_TESTED = ["ARRF", "LSTM", "Chronos"]
ML_ALL = ML_TESTED
# Chronos was previously reported but untested, because the t5 forward path
# was unusable (it collapsed onto a token-bin lattice). Chronos-Bolt produces a
# clean forward path, so it is now a full member of the race.
#
# Blend membership is kept separate from test membership so the two decisions
# are independent: Chronos joins the ensembles as well, since its h=1 forecast
# is unconditional and real-time-feasible on the same information set as the
# other members. Set to ["ARRF", "LSTM"] to test Chronos without blending it.
ENSEMBLE_ML = ML_TESTED
ENSEMBLE = ["Ensemble_avg", "Ensemble_invmse"]
COND_EXPOST = ["ARDL_cond", "NARDL_cond"]

# The *_rt columns re-select the ARDL/NARDL orders, the NARDL's decomposed
# variable, and the VECM's K and rank using data up to each origin only. The
# frozen variant keeps the full-sample choices and is reported for disclosure,
# not tested: putting it in the model set would compare specifications that saw
# the evaluation window against benchmarks that didn't.
VARIANTS = {
    "rt":     ("VECM_rt", "ARDL_rt",     "NARDL_rt"),
    "rt_dir": ("VECM_rt", "ARDL_dir_rt", "NARDL_dir_rt"),
    "frozen": ("VECM",    "ARDL_rw",     "NARDL_rw"),
}
TESTED_VARIANTS = ["rt", "rt_dir"]
HEADLINE_VARIANT = "rt"
FROZEN_DISCLOSURE = list(VARIANTS["frozen"]) + ["ARDL_dir", "NARDL_dir"]

SPA_BENCHMARKS = ["RW"]
WARMUP = 8
EXCLUDE_QUARTERS = ["2020Q2", "2020Q3"]

# (display_name, filename, quarter_col, pred_col, actual_col, date_to_quarter)
SOURCES = [
    ("RW_py", "chronos_forecasts.csv", "Quarter", "rw", "actual", False),
    ("Chronos", "chronos_forecasts.csv", "Quarter", "chronos", "actual", False),
    ("ARRF", "rf_forecasts.csv", "Quarter", "rf", "actual", False),
    ("LSTM", "lstm_headline_forecast.csv", "date", "lstm", "actual", True),
    ("RW", "h1_forecasts.csv", "date", "rw", "actual", True),
    ("SNAIVE", "h1_forecasts.csv", "date", "snaive", "actual", True),
    ("AR", "h1_forecasts.csv", "date", "ar", "actual", True),
    ("TSLM", "h1_forecasts.csv", "date", "tslm", "actual", True),
    ("TSLM_s", "h1_forecasts.csv", "date", "tslm_s", "actual", True),
    ("VECM", "h1_forecasts.csv", "date", "vecm", "actual", True),
    ("ARDL_rw", "h1_forecasts.csv", "date", "ardl_rw", "actual", True),
    ("NARDL_rw", "h1_forecasts.csv", "date", "nardl_rw", "actual", True),
    ("ARDL_dir", "h1_forecasts.csv", "date", "ardl_dir", "actual", True),
    ("NARDL_dir", "h1_forecasts.csv", "date", "nardl_dir", "actual", True),
    ("VECM_rt", "h1_forecasts.csv", "date", "vecm_rt", "actual", True),
    ("ARDL_rt", "h1_forecasts.csv", "date", "ardl_rt", "actual", True),
    ("NARDL_rt", "h1_forecasts.csv", "date", "nardl_rt", "actual", True),
    ("ARDL_dir_rt", "h1_forecasts.csv", "date", "ardl_dir_rt", "actual", True),
    ("NARDL_dir_rt", "h1_forecasts.csv", "date", "nardl_dir_rt", "actual", True),
    ("ARDL_cond", "h1_forecasts.csv", "date", "ardl_cond", "actual", True),
    ("NARDL_cond", "h1_forecasts.csv", "date", "nardl_cond", "actual", True),
]


roles = pd.read_csv(FORECASTS_DIR / "h1_model_roles.csv").set_index("model")["role"]
role_of = {name: roles.get(pcol)
           for (name, fname, _, pcol, _, _) in SOURCES
           if fname == "h1_forecasts.csv"}
assert all(role_of.get(m) == "conditional" for m in COND_EXPOST), \
    f"expected conditional role for {COND_EXPOST}, got {role_of}"
assert all(role_of.get(m) in ("model_set", "robust_swap")
           for k in TESTED_VARIANTS for m in VARIANTS[k]), \
    f"a tested structural column is not unconditional: {role_of}"
# The frozen-spec columns must never end up in a tested panel: R marks them
# `frozen_spec` precisely so this assertion can catch it if VARIANTS is edited.
assert all(role_of.get(m) == "frozen_spec" for m in FROZEN_DISCLOSURE), \
    f"frozen-spec columns are not tagged frozen_spec: {role_of}"

# Load, merge and chronologize
merged = None
for name, fname, qcol, pcol, acol, dtc in SOURCES:
    df = pd.read_csv(FORECASTS_DIR / fname)[[qcol, pcol, acol]]
    df["Quarter"] = quarter_str_from_dates(df[qcol]) if dtc else df[qcol]
    df = df[["Quarter", pcol, acol]].rename(columns={pcol: name, acol: f"actual_{name}"})
    merged = df if merged is None else merged.merge(df, on="Quarter", how="inner")

merged = merged.assign(_q=pd.PeriodIndex(merged["Quarter"], freq="Q")).sort_values("_q").reset_index(drop=True)

actual_cols = [c for c in merged.columns if c.startswith("actual_")]
actual = merged[actual_cols[0]].to_numpy(dtype=float)

# Test actuals across files for vintages
for c in actual_cols[1:]:
    assert np.allclose(merged[c], actual, atol=1e-6, equal_nan=True), \
        f"Actuals disagree between {actual_cols[0]} and {c}"

# Check if RW in python aligns with RW in R
assert np.allclose(merged["RW_py"], merged["RW"], atol=1e-6), "Python and R RW series misaligned!"

# RMSE gap between conditional and RW projected
for u, c in [("ARDL_rw", "ARDL_cond"), ("NARDL_rw", "NARDL_cond")]:
    r_u = np.sqrt(np.nanmean((actual - merged[u].to_numpy(float)) ** 2))
    r_c = np.sqrt(np.nanmean((actual - merged[c].to_numpy(float)) ** 2))
    print(f"lookahead value  {u:10s} {r_u:.4f}  vs  {c:11s} {r_c:.4f}"
          f"   ({100 * (r_u - r_c) / r_u:+.1f}% RMSE)")


def invmse_ensemble(preds, actual, warmup, weight_mask):
    """Expanding-window inverse-MSE weights. nanmean, so a model with a NaN
    in its history is skipped at that t rather than poisoning every later
    weight. weight_mask restricts the weighting history to the evaluation
    subsample, so the ex-COVID panel isn't weighted on COVID quarters."""
    out = np.nanmean(preds, axis=1)
    for t in range(warmup, len(preds)):
        hist = weight_mask[:t]
        mses = np.nanmean((actual[:t, None] - preds[:t, :])[hist] ** 2, axis=0)
        ok = np.isfinite(mses) & (mses > 0) & np.isfinite(preds[t])
        if ok.sum():
            w = (1 / mses[ok]) / np.sum(1 / mses[ok])
            out[t] = np.sum(w * preds[t, ok])
    return out


full_mask = np.ones(len(merged), dtype=bool)
excov_mask = ~merged["Quarter"].isin(EXCLUDE_QUARTERS).to_numpy()

PLOT_NOTE = ("Structural models use lag orders / K / rank re-selected at each origin. "
             "Ex post columns (ARDL_cond, NARDL_cond) omitted: conditioned on "
             "realised covariates at t+1, not comparable on the same "
             "information set.")

results = {}

print("\nSpecification leakage: full-sample vs real-time selection (RMSE, full sample)")
for frozen, rt in [("ARDL_rw", "ARDL_rt"), ("NARDL_rw", "NARDL_rt"),
                   ("VECM", "VECM_rt"), ("ARDL_dir", "ARDL_dir_rt"),
                   ("NARDL_dir", "NARDL_dir_rt")]:
    r_f = np.sqrt(np.nanmean((actual - merged[frozen].to_numpy(float)) ** 2))
    r_r = np.sqrt(np.nanmean((actual - merged[rt].to_numpy(float)) ** 2))
    r_b = np.sqrt(np.nanmean((actual - merged["RW"].to_numpy(float)) ** 2))
    print(f"  {frozen:10s} {r_f:.4f}  ->  {rt:13s} {r_r:.4f}"
          f"   ({100 * (r_r - r_f) / r_f:+.1f}% RMSE, {100 * (r_r - r_b) / r_b:+.1f}% vs RW)")


for vkey, struct in VARIANTS.items():
    struct = list(struct)
    tested = vkey in TESTED_VARIANTS
    blend = struct + ENSEMBLE_ML
    blend_preds = merged[blend].to_numpy(dtype=float)

    mcs_models = BENCH + struct + ML_TESTED
    spa_models = mcs_models + ENSEMBLE

    panels = [
        ("Unconditional benchmarks", BENCH),
        ("Structural (spec selected per origin)" if tested
         else "Structural (full-sample spec -- MCS/SPA below inherit its "
              "specification leakage, report for disclosure only)", struct),
        ("ML", ML_ALL),
        ("Ensembles", ENSEMBLE),
        ("Ex post (conditional, not tested)", COND_EXPOST),
    ]

    for label, mask, fig_title in [
        ("Full sample", full_mask, "Forecast RMSE - Full Sample"),
        ("Excluding COVID (2020Q2-Q3)", excov_mask, "Forecast RMSE - Excluding COVID"),
    ]:
        m = merged.copy()
        m["Ensemble_avg"] = np.nanmean(blend_preds, axis=1)
        m["Ensemble_invmse"] = invmse_ensemble(blend_preds, actual, WARMUP, mask)

        res = report(actual, m, mask, f"{label} [variant={vkey}]",
                     panels=panels, mcs_models=mcs_models,
                     spa_benchmarks=SPA_BENCHMARKS, spa_models=spa_models)
        results[(vkey, label)] = res

        if vkey == HEADLINE_VARIANT:    
            lb = run_ljungbox(res["errs"], mcs_models + ENSEMBLE)
            print(f"\n  [Ljung-Box, {label}]")
            print(lb.to_string(index=False))
            plot_rmse_bar(res["errs"], FIGURES_DIR, title=fig_title,
                          exclude=COND_EXPOST, note=PLOT_NOTE)

            # Testing block 4
            for loss in ("sq", "abs"):
                spa4 = run_spa(res["errs"], "RW", spa_models, loss=loss, block_size=4)
                mcs4 = run_mcs(res["errs"], mcs_models, loss=loss, block_size=4)
                name = "MSE" if loss == "sq" else "MAE"
                print(f"\n  [block_size=4, {name}] p(consistent)={spa4['pvalues']['consistent']:.3f}"
                    f"  better than RW: {', '.join(spa4['better']) or 'none'}")
                print(mcs4.to_string(index=False))