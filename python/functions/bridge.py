import pandas as pd
import numpy as np
import re
import joblib

AR_ORDER = 4
STARTS_LAG = 4
DUMMY_COLS = ["gfc_2008_2009", "d20Q2", "d20Q3", "d20Q4"]

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



def seasonal_factors(series, since=None, freq=4):
    """Log-additive quarterly seasonal factors, classical decomposition
    (centred 2x4 moving average), normalised to sum to zero.

    Needed because OBR publishes some paths seasonally adjusted (transactions
    are flagged as such in EFO table 1.16) while the models are estimated on
    the raw NSA master series. Adding these back puts a projected SA path onto
    the basis the coefficients were fitted on. `since` restricts to a recent
    window so a changed seasonal regime isn't averaged with the 1970s.
    """
    ln = np.log(series.dropna())
    trend = ln.rolling(freq, center=True).mean().rolling(2, center=True).mean()
    dev = (ln - trend).dropna()
    if since is not None:
        dev = dev[dev.index.year >= since]
    s = dev.groupby(dev.index.quarter).mean()
    return s - s.mean()


def build_bridge_inputs(base="../.."):
    fy_map = lambda q: q.year - 1 if q.quarter == 1 else q.year
    bridge = joblib.load(f"{base}/data/processed/completions_bridge.pkl")
 
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
    
    return dict(model=bridge["model"], smearing=bridge["smearing"],
                hist_ln_C=bridge["hist_ln_C"], hist_ln_S=bridge["hist_ln_S"],
                fy_map=fy_map, net_add=net_add, actual_back=actual_back, lt120=lt120)


def run_bridge(path, log_col, model, smearing, hist_ln_C, hist_ln_S, fy_map, net_add,
                actual_back, strip_space=False):
    fc = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    if strip_space:
        fc["period"] = fc["period"].str.replace(" ", "", regex=False)
    if log_col == "ensemble_log" and "ensemble_log" not in fc:
        fc["ensemble_log"] = (fc["ardl_log"] + fc["nardl_log"]) / 2
    fc["ensemble_log"] = fc[log_col]
 
    quarters = pd.PeriodIndex(fc["period"], freq="Q")
    future_ln_S = pd.concat([hist_ln_S, pd.Series(fc["ensemble_log"].values, index=quarters)])
 
    completions_fc = recursive_completions_forecast(
        model, hist_ln_C, future_ln_S, horizon=len(fc), smearing=smearing
    )
    assert (completions_fc.index == quarters).all(), "forecast horizon doesn't align with fc periods"
    fc["completions"] = completions_fc["C_hat"].values
 
    fc["fy"] = quarters.map(fy_map)
    annual = fc.groupby("fy")["completions"].sum().rename("private_completions").to_frame().iloc[1:]
    annual["net_additions"] = net_add(annual["private_completions"])
 
    q1 = quarters[quarters.year == 2026][0]
    priv_2025_26 = actual_back + fc.loc[quarters == q1, "completions"].iloc[0]
 
    return {"2024-25": 208600, "2025-26": net_add(priv_2025_26),
            "2026-27": annual.loc[2026, "net_additions"],
            "2027-28": annual.loc[2027, "net_additions"],
            "2028-29": annual.loc[2028, "net_additions"]}
 


def _short_run_terms(ecm, var):
    terms = {}
    key0 = f"d({var})"
    if key0 in ecm:
        terms[0] = ecm[key0]
    pattern = re.compile(rf"^d\(L\({re.escape(var)},\s*(\d+)\)\)$")
    for k in ecm.keys():
        m = pattern.match(k)
        if m:
            terms[int(m.group(1))] = ecm[k]
    return terms


def _seasonal_term(ecm, quarter):
    """Centred seasonals"""
    return sum(ecm[f"sd{j}"] * ((quarter == j) - 0.25) for j in (1, 2, 3))


_DUMMY_RE = re.compile(r"^d(\d{2})Q(\d)$")


def _dummy_period(name):
    """'d08Q3' -> Period('2008Q3'); None if the name isn't an impulse dummy.
    Two-digit years map to 1950-2049, which covers the sample."""
    m = _DUMMY_RE.match(name)
    if not m:
        return None
    yy, q = int(m.group(1)), int(m.group(2))
    return pd.Period(year=2000 + yy if yy < 50 else 1900 + yy, quarter=q, freq="Q")


def _impulse_term(ecm, period):
    """Impulse dummies. Zero over any forecast horizon, so scenarios are
    unaffected -- but non-zero on the event quarters, which is what makes a
    backtest over 2008Q3 / 2020Q2-Q3 / 2023Q2-Q3 reproduce the R ECM."""
    return sum(coef for name, coef in ecm.items()
               if _dummy_period(name) == period)


def compute_d_lhstarts(window, new_exog, theta_mult, ecm, lr, t0=None, period=None):
    theta_mult = theta_mult or {}
    t0 = t0 or {}
    period = pd.Period(period, freq="Q")
    quarter = period.quarter
    w = window  # w[0]=t-4 ... w[-1]=t-1
    max_lag = len(w) - 1
 
    diffs = {}
    for var in ["lrprc", "lvol", "r3", "lrcc"]:
        d = {0: new_exog[var] - w[-1][var]}
        for k in range(1, len(w)):
            d[k] = w[-k][var] - w[-k - 1][var]
        diffs[var] = d
 
    d_lhstarts_lags = {k: w[-k]["lhstarts"] - w[-k - 1]["lhstarts"]
                        for k in range(1, len(w))}
    long_run_terms = {}
    for var in ["lrprc", "lvol", "r3", "lrcc"]:
        mult = theta_mult.get(var, 1.0)
        var_t0 = t0.get(var, w[-1][var])
        long_run_terms[var] = lr[var] * (var_t0 + mult * (w[-1][var] - var_t0))
 
    ect_lag1 = w[-1]["lhstarts"] - sum(long_run_terms.values())
 
    d_lhstarts_t = ecm["(Intercept)"]
 
    for lag, coef in _short_run_terms(ecm, "lhstarts").items():
        if lag not in d_lhstarts_lags:
            raise ValueError(f"ECM needs d(L(lhstarts,{lag})) but window only "
                              f"covers lags up to {max_lag} - extend the window.")
        d_lhstarts_t += coef * d_lhstarts_lags[lag]
 
    for var in ["lrprc", "lvol", "r3", "lrcc"]:
        for lag, coef in _short_run_terms(ecm, var).items():
            if lag not in diffs[var]:
                raise ValueError(f"ECM needs d(L({var},{lag})) but window only "
                                  f"covers lags up to {max_lag} - extend the window.")
            d_lhstarts_t += coef * diffs[var][lag]
 
    d_lhstarts_t += _seasonal_term(ecm, quarter)
    d_lhstarts_t += _impulse_term(ecm, period)

    d_lhstarts_t += ecm["ect"] * ect_lag1
    return d_lhstarts_t
 

def run_scenario(theta_mult, init_df, obr_df, ecm, lr):
    theta_mult = theta_mult or {}
    # Impulse dummies are handled by _impulse_term, but only if their names
    # parse -- anything else is a term nothing in here applies.
    unused = [k for k in ecm.index
              if not k.startswith("d(")
              and k not in {"(Intercept)", "ect", "sd1", "sd2", "sd3"}
              and _dummy_period(k) is None]
    assert not unused, f"unhandled ECM terms: {unused}"
 
    window = init_df.to_dict("records")
    t0 = {var: window[-1][var] for var in theta_mult}
 
    results = []
    for t in range(len(obr_df)):
        row = obr_df.iloc[t]
        new_exog = row[["lrprc", "lvol", "r3", "lrcc"]].to_dict()

        d_lhstarts = compute_d_lhstarts(window, new_exog, theta_mult, ecm, lr,
                                        t0, period=row["period"])
        lhstarts_t = window[-1]["lhstarts"] + d_lhstarts
 
        window = window[1:] + [{**new_exog, "lhstarts": lhstarts_t}]
        results.append({"period": row["period"], "lhstarts": lhstarts_t})
 
    return pd.DataFrame(results)


def backtest_one_step(master, target_period, ecm, lr):
    idx = master.index[master["period"] == target_period][0]
    window = master.iloc[idx-4:idx][["lhstarts", "lrprc", "lvol", "r3", "lrcc"]].to_dict("records")

    actual_row = master.iloc[idx]
    new_exog = actual_row[["lrprc", "lvol", "r3", "lrcc"]].to_dict()

    predicted_d = compute_d_lhstarts(window, new_exog, {}, ecm, lr,
                                     period=target_period)
    actual_d = actual_row["lhstarts"] - window[-1]["lhstarts"]

    print(f"Target: {target_period}")
    print(f"  Predicted Δlhstarts: {predicted_d:.4f}")
    print(f"  Actual Δlhstarts:    {actual_d:.4f}")
    print(f"  Predicted lhstarts:  {window[-1]['lhstarts'] + predicted_d:.4f}")
    print(f"  Actual lhstarts:     {actual_row['lhstarts']:.4f}")

    return predicted_d, actual_d

 
def _seasonal_dummies(quarter):
    """Calendar-quarter dummies, Q1 as dropped baseline."""
    return {f"q{q}": int(quarter == q) for q in [2, 3, 4]}
 
 
def recursive_completions_forecast(model, hist_ln_C, future_ln_S,
    horizon, smearing=1.0, dummy_cols=None, ar_order=AR_ORDER, starts_lag=STARTS_LAG):

    dummy_cols = DUMMY_COLS if dummy_cols is None else dummy_cols
    future_index = pd.period_range(hist_ln_C.index.max() + 1, periods=horizon, freq="Q")
 
    needed = pd.period_range(
        future_index.min() - starts_lag, future_index.max() - starts_lag, freq="Q"
    ).difference(future_ln_S.index)
    if len(needed) > 0:
        raise ValueError(f"future_ln_S missing required quarters: {list(needed)}")
 
    ln_C = hist_ln_C.copy()
    records = []
    for q in future_index:
        row = {f"ln_C_lag{L}": ln_C.loc[q - L] for L in range(1, ar_order + 1)}
        row[f"ln_S_lag{starts_lag}"] = future_ln_S.loc[q - starts_lag]
        row.update(_seasonal_dummies(q.quarter))
        row.update({d: 0 for d in dummy_cols})
 
        ln_C_hat = float(model.predict(pd.DataFrame([row])).iloc[0])
        ln_C.loc[q] = ln_C_hat
        records.append(
            {"quarter": q, "ln_C_hat": ln_C_hat, "C_hat": smearing * np.exp(ln_C_hat)}
        )
 
    return pd.DataFrame(records).set_index("quarter")