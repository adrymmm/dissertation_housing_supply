import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from arch.bootstrap import MCS, SPA
from statsmodels.stats.diagnostic import acorr_ljungbox


def quarter_str_from_dates(date_series):
    dt = pd.to_datetime(date_series)
    return dt.dt.year.astype(str) + "Q" + dt.dt.quarter.astype(str)


def _loss_frame(errs_dict, cols, loss):
    df = pd.DataFrame({c: np.asarray(errs_dict[c], float) for c in cols}).dropna()
    return df ** 2 if loss == "sq" else df.abs()


def run_mcs(errs_dict, models, alpha=0.1, reps=10_000, block_size=1,
            loss="sq", seed=42):
    L = _loss_frame(errs_dict, models, loss)
    mcs = MCS(L, size=alpha, reps=reps, block_size=block_size, seed=seed)
    mcs.compute()

    pvals = mcs.pvalues.iloc[:, 0]
    inc = set(mcs.included)
    return pd.DataFrame({
        "Model": list(pvals.index),
        "MCS_p": pvals.to_numpy(float),
        "In_MCS": [m in inc for m in pvals.index],
        "n": len(L),
    }).sort_values("MCS_p", ascending=False).reset_index(drop=True)


def run_spa(errs_dict, benchmark, models, alpha=0.1, reps=10_000, block_size=1,
            loss="sq", seed=42):
    alts = [m for m in models if m != benchmark]
    L = _loss_frame(errs_dict, [benchmark] + alts, loss)

    spa = SPA(L[benchmark], L[alts], block_size=block_size, reps=reps,
              studentize=True, nested=False, seed=seed)
    spa.compute()

    idx = np.asarray(spa.better_models(pvalue=alpha, pvalue_type="consistent"))
    better = [alts[i] for i in idx]
    return {"benchmark": benchmark, "alternatives": alts,
            "pvalues": spa.pvalues, "better": better, "n": len(L)}


def report(actual, merged, mask, label, panels, mcs_models, spa_benchmarks,
           spa_models, mcs_alpha=0.1, mcs_reps=10_000):
    all_models = sorted({m for _, members in panels for m in members},
                        key=lambda m: m)
    errs = {m: (actual - merged[m].to_numpy(dtype=float))[mask]
            for m in all_models}

    print(f"\n=== {label} ({mask.sum()} quarters) ===")

    for panel, members in panels:
        print(f"\n  [{panel}]\n  {'Model':18s} {'RMSE':>8s} {'MAE':>8s} "
              f"{'ME':>9s} {'n':>5s}")
        for m in members:
            e = errs[m][np.isfinite(errs[m])]
            print(f"  {m:18s} {np.sqrt(np.mean(e**2)):8.4f} "
                  f"{np.mean(np.abs(e)):8.4f} {np.mean(e):+9.4f} {len(e):5d}")

    spa = {}
    for b in spa_benchmarks:
        for loss in ("sq", "abs"):
            res = run_spa(errs, b, spa_models, alpha=mcs_alpha, reps=mcs_reps,
                          loss=loss)
            spa[(b, loss)] = res
            name = "MSE" if loss == "sq" else "MAE"
            pv = res["pvalues"]
            print(f"\n  [SPA vs {b}, {name} loss, n={res['n']}] "
                  f"H0: {b} not inferior to any of "
                  f"{len(res['alternatives'])} alternatives")
            print(f"    p(lower) = {pv['lower']:.3f}   "
                  f"p(consistent) = {pv['consistent']:.3f}   "
                  f"p(upper) = {pv['upper']:.3f}")
            print(f"    superior to {b} at {mcs_alpha:g}: "
                  f"{', '.join(res['better']) if res['better'] else 'none'}")

    mcs = {}
    for loss in ("sq", "abs"):
        res = run_mcs(errs, mcs_models, alpha=mcs_alpha, reps=mcs_reps,
                      loss=loss)
        mcs[loss] = res
        name = "MSE" if loss == "sq" else "MAE"
        print(f"\n  [MCS, {name} loss, alpha={mcs_alpha}, "
              f"n={res['n'].iloc[0]}] panel: {', '.join(mcs_models)}")
        for _, r in res.iterrows():
            print(f"    {r.Model:18s} p = {r.MCS_p:.3f}"
                  + ("  *(superior set)" if r.In_MCS else ""))

    return {"errs": errs, "spa": spa, "mcs": mcs}


def plot_rmse_bar(errs, outdir, title="", exclude=(), note=""):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rmse = {n: np.sqrt(np.nanmean(np.asarray(e, float) ** 2))
            for n, e in errs.items() if n not in exclude}
    order = sorted(rmse, key=rmse.get)
    colors = ['#2E7D32' if n.startswith('Ensemble') else '#9E9E9E'
              for n in order]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    bars = ax.bar(order, [rmse[n] for n in order], color=colors)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.4f}",
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha='right')
    if note:
        fig.text(0.01, 0.01, note, fontsize=7.5, color='#555555', ha='left')
        fig.tight_layout(rect=(0, 0.045, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(outdir / f"{'_'.join(title.lower().replace('-', ' ').split())}.png",
                dpi=200)
    plt.show()

def run_ljungbox(errs_dict, models, lags=4):
    """Ljung-Box on h=1 forecast errors"""
    rows = []
    for m in models:
        e = np.asarray(errs_dict[m], float)
        e = e[np.isfinite(e)]
        lag = min(lags, len(e) - 1)
        lb = acorr_ljungbox(e, lags=[lag], return_df=True)
        rows.append({
            "Model": m,
            "n": len(e),
            "lag": lag,
            "LB_stat": round(lb["lb_stat"].iloc[0], 3),
            "p_value": round(lb["lb_pvalue"].iloc[0], 4),
        })
    return pd.DataFrame(rows).sort_values("p_value")