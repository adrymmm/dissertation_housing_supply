
import pandas as pd
import numpy as np
import json
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

def build_bridge_inputs(base="../.."):
    fy_map = lambda q: q.year - 1 if q.quarter == 1 else q.year
    p = json.load(open(f"{base}/data/processed/bridge_params.json"))
    seasonal = {1: 0, 2: p["q2"], 3: p["q3"], 4: p["q4"]}

    eng = pd.read_csv(f"{base}/data/python_master/england_master.csv")
    seed = np.log(eng["starts"].dropna().iloc[-4:].values)

    ons = pd.read_excel(f"{base}/data/raw/starts/indicatorsofukhousebuilding.xlsx", sheet_name="1b", skiprows=5)
    ons["quarter"] = ons["Period"].apply(parse_quarter)
    ons = ons.dropna(subset=["quarter"]).set_index("quarter").sort_index()
    ons_fy = ons.index.map(fy_map)

    components = ["New build completions", "Net conversions", "Net change of use",
                 "Net other gains", "Demolitions", "Total net additional dwellings"]
    lt120 = pd.read_excel(f"{base}/data/raw/net_additions/Live_Table_120.ods", sheet_name="LT120_unrounded", skiprows=4)
    lt120 = lt120.set_index(lt120.columns[0]).T[components].iloc[:-2].apply(pd.to_numeric, errors="coerce")
    avg = lt120.iloc[-4:-1].mean()

    priv_actual = ons.groupby(ons_fy)["Completed - Private Enterprise"].sum().loc[2021:2023].mean()
    non_private = avg["New build completions"] - priv_actual
    net_add = lambda priv: (priv + non_private + avg["Net conversions"] + avg["Net change of use"]
                            - avg["Demolitions"] + avg["Net other gains"])
    actual_back = ons.loc[pd.PeriodIndex(["2025Q2","2025Q3","2025Q4"], freq="Q"),
                          "Completed - Private Enterprise"].sum()

    return dict(p=p, seasonal=seasonal, seed=seed, fy_map=fy_map,
                net_add=net_add, actual_back=actual_back, lt120=lt120)

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

