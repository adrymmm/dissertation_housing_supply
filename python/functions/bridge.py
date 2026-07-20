
import pandas as pd
import numpy as np

# Convert the period text into a quarterly time index
def parse_quarter(x):
    x = str(x)

    if len(x) < 4:
        return pd.NaT

    try:
        year = int(x[-4:])
    except:
        return pd.NaT

    if x.startswith("Jan - Mar"):
        q = 1
    elif x.startswith("Apr - Jun"):
        q = 2
    elif x.startswith("Jul - Sep"):
        q = 3
    elif x.startswith("Oct - Dec"):
        q = 4
    else:
        return pd.NaT

    return pd.Period(year=year, quarter=q, freq="Q")


def run_bridge(path, log_col, seed, p, seasonal, fy_map, net_add, actual_back, strip_space=False):
    fc = pd.read_csv(path)
    if strip_space:
        fc["period"] = fc["period"].str.replace(" ", "", regex=False)
    if log_col == "ensemble_log" and "ensemble_log" not in fc:
        fc["ensemble_log"] = (fc["ardl_log"] + fc["nardl_log"]) / 2
    fc["ensemble_log"] = fc[log_col]

    ln_S_full = np.concatenate([seed, fc["ensemble_log"].values])
    quarters = pd.PeriodIndex(fc["period"], freq="Q")

    ln_C, lag_C = [], p["last_ln_C"]
    for t in range(len(fc)):
        lag_C = p["intercept"] + p["rho"]*lag_C + p["beta"]*ln_S_full[t] + seasonal[quarters[t].quarter]
        ln_C.append(lag_C)
    fc["completions"] = np.exp(ln_C) * p["smearing_factor"]
    fc["fy"] = quarters.map(fy_map)
    annual = fc.groupby("fy")["completions"].sum().rename("private_completions").to_frame().iloc[1:]
    annual["net_additions"] = net_add(annual["private_completions"])

    q1 = quarters[quarters.year == 2026][0]
    priv_2025_26 = actual_back + fc.loc[quarters == q1, "completions"].iloc[0]

    return {"2024-25": 208600, "2025-26": net_add(priv_2025_26),
            "2026-27": annual.loc[2026, "net_additions"],
            "2027-28": annual.loc[2027, "net_additions"],
            "2028-29": annual.loc[2028, "net_additions"]}

