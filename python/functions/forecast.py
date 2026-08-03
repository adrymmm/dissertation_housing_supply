import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import t as tdist
import matplotlib.pyplot as plt


def metrics(a, p):
    e = a - p
    return np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e))

def dm_test(e1, e2, h=1, power=2):
    """H0: equal accuracy. Loss = |e|^power. stat>0 => model-1 worse.
    HLN small-sample correction; compared to t(n-1)."""
    d = np.abs(e1) ** power - np.abs(e2) ** power
    nn = len(d)
    dbar = d.mean()
    var = np.sum((d - dbar) ** 2) / nn                  # gamma0
    for lag in range(1, h):                             # HAC up to h-1 (none for h=1)
        var += 2 * np.sum((d[lag:] - dbar) * (d[:-lag] - dbar)) / nn
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var / nn)
    stat *= np.sqrt((nn + 1 - 2 * h + h * (h - 1) / nn) / nn)
    return stat, 2 * tdist.cdf(-abs(stat), df=nn - 1)

# Convert R-style date column to "YYYYQ#"
def quarter_str_from_dates(date_series):
    dt = pd.to_datetime(date_series)
    return dt.dt.year.astype(str) + "Q" + dt.dt.quarter.astype(str)

def report(mask, label):
    print(f"\n=== {label} ({mask.sum()} quarters) ===")
    print(f"{'Model':18s} {'RMSE':>8s} {'MAE':>8s}")
    errs = {}
    for name in model_names:
        e = (actual - merged[name].to_numpy())[mask]
        errs[name] = e
        rmse = np.sqrt(np.mean(e ** 2))
        mae = np.mean(np.abs(e))
        print(f"{name:18s} {rmse:8.4f} {mae:8.4f}")

    if RW_BENCHMARK in errs:
        print(f"\nDM vs {RW_BENCHMARK} (stat>0 => model worse than RW):")
        for name in model_names:
            if name == RW_BENCHMARK:
                continue
            s, pv = dm_test(errs[name], errs[RW_BENCHMARK])
            flag = "  *significant at 5%" if pv < 0.05 else ""
            print(f"  {name:18s} stat={s:+.3f}  p={pv:.3f}{flag}")

    if STRUCTURAL_BEST in errs and ML_BEST in errs:
        s, pv = dm_test(errs[ML_BEST], errs[STRUCTURAL_BEST])
        flag = "  *significant at 5%" if pv < 0.05 else ""
        print(f"\nDM {ML_BEST} vs {STRUCTURAL_BEST}: stat={s:+.3f}  p={pv:.3f}"
              f"  (stat>0 => {ML_BEST} worse){flag}")

    if "Ensemble_avg" in errs and STRUCTURAL_BEST in errs:
        s, pv = dm_test(errs["Ensemble_avg"], errs[STRUCTURAL_BEST])
        flag = "  *significant at 5%" if pv < 0.05 else ""
        print(f"\nDM Ensemble_avg vs {STRUCTURAL_BEST}: stat={s:+.3f}  p={pv:.3f}"
              f"  (stat>0 => ensemble worse){flag}")

    return errs

def plot_rmse_bar(errs, benchmark="RW", title=""):
    rmse = {n: np.sqrt(np.mean(e ** 2)) for n, e in errs.items()}
    order = sorted(rmse, key=rmse.get)  # ascending, best first

    pvals = {}
    for n in order:
        if n == benchmark:
            continue
        _, pv = dm_test(errs[n], errs[benchmark])
        pvals[n] = pv

    colors = ['#2E7D32' if n.startswith('Ensemble') else '#C62828' if n == "TSLM" else '#9E9E9E' for n in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(order, [rmse[n] for n in order], color=colors)

    for bar, n in zip(bars, order):
        h = bar.get_height()
        star = "*" if pvals.get(n, 1) < 0.05 else ""
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.4f}{star}",
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.set_xticklabels(order, rotation=30, ha='right')
    fig.tight_layout()
    fig.savefig(f"data/outputs/figures/{title.replace(' ', '_').lower()}.png", dpi=200)
    plt.show()

