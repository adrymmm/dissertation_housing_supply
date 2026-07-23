
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

def compute_d_lhstarts(window, new_exog, theta_price_mult, ecm, lr, lrprc_t0=None):
    w = window  # w[0]=t-4 ... w[3]=t-1

    d_lrprc = {
        0: new_exog["lrprc"] - w[3]["lrprc"],
        1: w[3]["lrprc"] - w[2]["lrprc"],
        2: w[2]["lrprc"] - w[1]["lrprc"],
        3: w[1]["lrprc"] - w[0]["lrprc"],
    }
    d_lvol = {
        0: new_exog["lvol"] - w[3]["lvol"],
        1: w[3]["lvol"] - w[2]["lvol"],
        2: w[2]["lvol"] - w[1]["lvol"],
        3: w[1]["lvol"] - w[0]["lvol"],
    }
    d_r3 = {
        0: new_exog["r3"] - w[3]["r3"],
        1: w[3]["r3"] - w[2]["r3"],
        2: w[2]["r3"] - w[1]["r3"],
        3: w[1]["r3"] - w[0]["r3"],
    }
    d_lrcc = {
        0: new_exog["lrcc"] - w[3]["lrcc"],
        1: w[3]["lrcc"] - w[2]["lrcc"],
        2: w[2]["lrcc"] - w[1]["lrcc"],
        3: w[1]["lrcc"] - w[0]["lrcc"],
    }

    d_lhstarts_lag1 = w[3]["lhstarts"] - w[2]["lhstarts"]
    d_lhstarts_lag2 = w[2]["lhstarts"] - w[1]["lhstarts"]
    d_lhstarts_lag3 = w[1]["lhstarts"] - w[0]["lhstarts"]

    # ect at t-1, no long-run intercept (case 3). price splits into a shared t0-level
    # anchor plus the uplifted response to price movement since t0, so elasticity only
    # bites on forward price change, not the raw level
    if lrprc_t0 is None:
        lrprc_t0 = w[3]["lrprc"]
    price_term = lr["lrprc"] * lrprc_t0 + lr["lrprc"] * theta_price_mult * (w[3]["lrprc"] - lrprc_t0)
    ect_lag1 = w[3]["lhstarts"] - (
        price_term
        + lr["lvol"] * w[3]["lvol"]
        + lr["r3"] * w[3]["r3"]
        + lr["lrcc"] * w[3]["lrcc"]
    )

    d_lhstarts_t = (
        ecm["(Intercept)"]
        + ecm["d(L(lhstarts, 1))"] * d_lhstarts_lag1
        + ecm["d(L(lhstarts, 2))"] * d_lhstarts_lag2
        + ecm["d(L(lhstarts, 3))"] * d_lhstarts_lag3
        + ecm["d(lrprc)"] * d_lrprc[0]
        + ecm["d(L(lrprc, 1))"] * d_lrprc[1]
        + ecm["d(L(lrprc, 2))"] * d_lrprc[2]
        + ecm["d(L(lrprc, 3))"] * d_lrprc[3]
        + ecm["d(lvol)"] * d_lvol[0]
        + ecm["d(L(lvol, 1))"] * d_lvol[1]
        + ecm["d(L(lvol, 2))"] * d_lvol[2]
        + ecm["d(L(lvol, 3))"] * d_lvol[3]
        + ecm["d(r3)"] * d_r3[0]
        + ecm["d(L(r3, 1))"] * d_r3[1]
        + ecm["d(L(r3, 2))"] * d_r3[2]
        + ecm["d(L(r3, 3))"] * d_r3[3]
        + ecm["d(lrcc)"] * d_lrcc[0]
        + ecm["d(L(lrcc, 1))"] * d_lrcc[1]
        + ecm["d(L(lrcc, 2))"] * d_lrcc[2]
        + ecm["d(L(lrcc, 3))"] * d_lrcc[3]
        + ecm["ect"] * ect_lag1
    )

    return d_lhstarts_t


def run_scenario(theta_price_mult, init_df, obr_df, ecm, lr):
    window = init_df.to_dict("records")
    lrprc_t0 = window[-1]["lrprc"]  # 2025Q4 anchor price, shared across scenarios

    results = []
    for t in range(len(obr_df)):
        new_exog = obr_df.iloc[t][["lrprc", "lvol", "r3", "lrcc"]].to_dict()

        d_lhstarts = compute_d_lhstarts(window, new_exog, theta_price_mult, ecm, lr, lrprc_t0)
        lhstarts_t = window[-1]["lhstarts"] + d_lhstarts

        window = window[1:] + [{**new_exog, "lhstarts": lhstarts_t}]
        results.append({"period": obr_df.iloc[t]["period"], "lhstarts": lhstarts_t})

    return pd.DataFrame(results)


def backtest_one_step(master, target_period, ecm, lr):
    idx = master.index[master["period"] == target_period][0]
    window = master.iloc[idx-4:idx][["lhstarts", "lrprc", "lvol", "r3", "lrcc"]].to_dict("records")

    actual_row = master.iloc[idx]
    new_exog = actual_row[["lrprc", "lvol", "r3", "lrcc"]].to_dict()

    predicted_d = compute_d_lhstarts(window, new_exog, 1.00, ecm, lr)
    actual_d = actual_row["lhstarts"] - window[-1]["lhstarts"]

    print(f"Target: {target_period}")
    print(f"  Predicted Δlhstarts: {predicted_d:.4f}")
    print(f"  Actual Δlhstarts:    {actual_d:.4f}")
    print(f"  Predicted lhstarts:  {window[-1]['lhstarts'] + predicted_d:.4f}")
    print(f"  Actual lhstarts:     {actual_row['lhstarts']:.4f}")

    return predicted_d, actual_d