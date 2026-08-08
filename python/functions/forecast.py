import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import t as tdist
from arch.bootstrap import MCS


def metrics(a, p):
    e = a - p
    return np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e))


def quarter_str_from_dates(date_series):
    dt = pd.to_datetime(date_series)
    return dt.dt.year.astype(str) + "Q" + dt.dt.quarter.astype(str)


def dm_test(e1, e2, h=1, power=2):
    """H0: equal accuracy. Loss = |e|^power. stat>0 => model-1 worse.
    HLN small-sample correction; compared to t(n-1). Non-finite pairs dropped."""
    d = np.abs(np.asarray(e1, float)) ** power - np.abs(np.asarray(e2, float)) ** power
    d = d[np.isfinite(d)]
    nn = len(d)
    if nn < 3:
        return np.nan, np.nan
    dbar = d.mean()
    var = np.sum((d - dbar) ** 2) / nn                  # gamma0
    for lag in range(1, h):                             # HAC up to h-1 (none for h=1)
        var += 2 * np.sum((d[lag:] - dbar) * (d[:-lag] - dbar)) / nn
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var / nn)
    stat *= np.sqrt((nn + 1 - 2 * h + h * (h - 1) / nn) / nn)
    return stat, 2 * tdist.cdf(-abs(stat), df=nn - 1)


def run_mcs(errs_dict, alpha=0.1, reps=10_000, block_size=1, loss="sq", seed=42):
    """Hansen, Lunde & Nason (2011) MCS, T_R statistic (arch default).

    Rows with any NaN are dropped: arch fails on NaN input with an opaque
    IndexError. block_size=1 because h=1 loss differentials are serially
    uncorrelated under correct specification, matching dm_test's no-HAC-at-h=1.
    """
    df = pd.DataFrame(errs_dict).astype(float).dropna()
    df = df ** 2 if loss == "sq" else df.abs()

    mcs = MCS(df, size=alpha, reps=reps, block_size=block_size, seed=seed)
    mcs.compute()

    pvals = mcs.pvalues.iloc[:, 0]
    inc = set(mcs.included)
    return pd.DataFrame({
        "Model": list(pvals.index),
        "MCS_p": pvals.to_numpy(float),
        "In_MCS": [m in inc for m in pvals.index],
        "n": len(df),
    }).sort_values("MCS_p", ascending=False).reset_index(drop=True)


def report(actual, merged, mask, label, uncond=None, cond=None, ml=None,
           benchmarks=None, mcs_models=None, mcs_alpha=0.1, mcs_reps=10_000):
    uncond = uncond or ["RW", "SNAIVE", "AR", "TSLM", "TSLM_s", "VECM"]
    cond = cond or ["ARDL", "NARDL", "Ensemble_avg", "Ensemble_invmse"]
    ml = ml or ["ARRF", "LSTM", "Chronos"]
    benchmarks = benchmarks or ["RW"]
    all_models = uncond + cond + ml
    mcs_models = mcs_models or [m for m in uncond + ml if m != "Chronos"]

    print(f"\n=== {label} ({mask.sum()} quarters) ===")
    errs = {m: (actual - merged[m].to_numpy(dtype=float))[mask] for m in all_models}

    for panel, members in [("Unconditional", uncond), ("Conditional", cond), ("ML", ml)]:
        print(f"\n  [{panel}]\n  {'Model':18s} {'RMSE':>8s} {'MAE':>8s} {'ME':>9s} {'n':>5s}")
        for m in members:
            e = errs[m][np.isfinite(errs[m])]
            print(f"  {m:18s} {np.sqrt(np.mean(e**2)):8.4f} {np.mean(np.abs(e)):8.4f} "
                  f"{np.mean(e):+9.4f} {len(e):5d}")

    rows = []
    for b in benchmarks:
        for m in all_models:
            if m == b:
                continue
            stat, pv = dm_test(errs[m], errs[b])
            rows.append({"Benchmark": b, "Model": m, "DM_stat": stat, "DM_p": pv})
    dm = pd.DataFrame(rows)
    print("\n  [Diebold-Mariano vs benchmark, h=1, MSE loss] stat>0 => model worse")
    for b in benchmarks:
        print(f"\n  vs {b}:\n    {'Model':18s} {'DM':>8s} {'p':>8s}")
        for _, r in dm[dm.Benchmark == b].sort_values("DM_stat").iterrows():
            print(f"    {r.Model:18s} {r.DM_stat:8.3f} {r.DM_p:8.3f}")

    mcs = {}
    for loss in ("sq", "abs"):
        res = run_mcs({m: errs[m] for m in mcs_models}, alpha=mcs_alpha,
                      reps=mcs_reps, loss=loss)
        mcs[loss] = res
        name = "MSE" if loss == "sq" else "MAE"
        print(f"\n  [MCS, {name} loss, alpha={mcs_alpha}, n={res['n'].iloc[0]}] "
              f"panel: {', '.join(mcs_models)}")
        for _, r in res.iterrows():
            print(f"    {r.Model:18s} p = {r.MCS_p:.3f}"
                  + ("  *(superior set)" if r.In_MCS else ""))

    return {"errs": errs, "dm": dm, "mcs": mcs}


def plot_rmse_bar(errs, outdir, title=""):
    """RMSE bars, best first. No significance markers: DM and MCS results are
    in the printed tables, so the bars don't carry two meanings of '*'."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rmse = {n: np.sqrt(np.nanmean(np.asarray(e, float) ** 2)) for n, e in errs.items()}
    order = sorted(rmse, key=rmse.get)
    colors = ['#2E7D32' if n.startswith('Ensemble') else '#9E9E9E' for n in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(order, [rmse[n] for n in order], color=colors)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.4f}",
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha='right')
    fig.tight_layout()
    fig.savefig(outdir / f"{'_'.join(title.lower().replace('-', ' ').split())}.png", dpi=200)
    plt.show()