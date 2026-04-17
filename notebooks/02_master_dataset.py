"""
02_master_dataset.py
--------------------
Reads raw files, cleans each source, merges into quarterly and annual panels.

Outputs
-------
data/processed/quarterly_master.csv
data/processed/annual_net_additions.csv

Geographic coverage notes
-------------------------
starts / completions  : England (ONS Table 253)
net additions         : England (MHCLG LT120)
epc_lodgements        : England (MHCLG EPC NB1)
housing_stock_eng     : England (DLUHC LT104)
lrprc (HPI)           : UK-wide (Nationwide) — England-specific mix-adjusted
                        series either too short (2011+) or simple average only;
                        Nationwide justified as England ≈85% of UK transactions
transactions_england  : England post-2005 (HMRC); pre-2005 scaled from UK-wide
                        BoE approvals using 2005–2007 linking factor
lrcc (construction)   : GB-level (BIS/ONS OPI splice) — no England equivalent
                        with sufficient length; justified as construction inputs
                        are traded in national markets
r3 (base rate)        : UK-wide by definition
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("../data/raw")
OUT = Path("../data/processed")
OUT.mkdir(parents=True, exist_ok=True)


# ── 1. Starts & Completions (quarterly, England) ──────────────────────────────
df_sc = pd.read_excel(
    RAW / "ons_starts_completions_england.xlsx",
    sheet_name="1b",
    skiprows=5,
)
df_sc = df_sc.dropna(subset=["Period", "Started - All Dwellings"])

df_sc["year"] = df_sc["Period"].str.extract(r"(\d{4})").astype(int)
df_sc["quarter"] = df_sc["Period"].apply(
    lambda x: 1 if "Jan" in str(x)
    else 2 if "Apr" in str(x)
    else 3 if "Jul" in str(x)
    else 4
)
df_sc["date"] = pd.to_datetime(
    df_sc["year"].astype(str) + "-"
    + (df_sc["quarter"] * 3 - 2).astype(str) + "-01"
)
df_sc = df_sc[
    ["date", "year", "quarter",
     "Started - All Dwellings", "Started - Private Enterprise",
     "Completed - All Dwellings", "Completed - Private Enterprise"]
].copy()
df_sc.columns = [
    "date", "year", "quarter",
    "starts_all", "starts_private", "comp_all", "comp_private",
]
print(f"[1] Starts/completions: {df_sc.shape[0]} quarters, "
      f"{df_sc.date.min().date()} to {df_sc.date.max().date()}")


# ── 2. Net Additions (annual, England) ────────────────────────────────────────
df_na = pd.read_excel(
    RAW / "mhclg_net_additions_england.ods",
    sheet_name="LT120_unrounded",
    engine="odf",
)

years_raw = df_na.iloc[3, 1:-2].tolist()
years = [str(y).split(" ")[0] for y in years_raw]
labels = df_na.iloc[:, 0].tolist()

rows_we_want = {
    "new_build_comp":      "New build completions",
    "net_conversions":     "Net conversions",
    "net_change_of_use":   "Net change of use",
    "net_other_gains":     "Net other gains",
    "demolitions":         "Demolitions",
    "census_adjustments":  "Census adjustments",
    "total_net_additions": "Total net additional dwellings",
}

data = {"fiscal_year": years}
for col_name, label in rows_we_want.items():
    row_idx = labels.index(label)
    values = df_na.iloc[row_idx, 1:-2].tolist()
    data[col_name] = pd.to_numeric(values, errors="coerce")

df_net = pd.DataFrame(data)
df_net["year"] = df_net["fiscal_year"].str[:4].astype(int)
print(f"[2] Net additions: {df_net.shape[0]} years, "
      f"{df_net.year.min()} to {df_net.year.max()}")


# ── 3. EPC New Builds (quarterly, England) ────────────────────────────────────
df_epc = pd.read_excel(
    RAW / "epc_new_builds.ods",
    sheet_name="NB1_England_Only",
    engine="odf",
)
df_epc.columns = df_epc.iloc[2].tolist()
df_epc = df_epc.iloc[3:].copy()
df_epc = df_epc.dropna(subset=["Quarter"]).copy()

df_epc["year"] = df_epc["Year"].astype(int)
df_epc["quarter"] = df_epc["Quarter"].str.split("/").str[1].astype(int)
df_epc["date"] = pd.to_datetime(
    df_epc["year"].astype(str) + "-"
    + (df_epc["quarter"] * 3 - 2).astype(str) + "-01"
)
df_epc = df_epc[["date", "year", "quarter", "Number Lodgements"]].copy()
df_epc.columns = ["date", "year", "quarter", "epc_lodgements"]
df_epc["epc_lodgements"] = pd.to_numeric(df_epc["epc_lodgements"], errors="coerce")
df_epc = df_epc.reset_index(drop=True)
print(f"[3] EPC lodgements: {df_epc.shape[0]} quarters, "
      f"{df_epc.date.min().date()} to {df_epc.date.max().date()}")


# ── 4. Nationwide HPI (quarterly, UK, 1952–present) ───────────────────────────
# UK-wide; justified as England ≈85% of UK transactions and no suitable
# England-specific mix-adjusted series with sufficient length exists.
df_hpi = pd.read_excel(
    RAW / "nationwide_housing_prices_1952.xlsx",
    sheet_name="UK HP Since 1952",
    skiprows=4,
    header=None,
)
df_hpi = df_hpi.iloc[:, :3].copy()
df_hpi.columns = ["quarter_label", "hpi_index", "hpi_price"]
df_hpi = df_hpi.dropna(subset=["quarter_label"])
df_hpi = df_hpi[
    df_hpi["quarter_label"].astype(str).str.match(r"^Q\d \d{4}$")
].copy()

df_hpi["quarter"] = df_hpi["quarter_label"].str[1].astype(int)
df_hpi["year"] = df_hpi["quarter_label"].str[-4:].astype(int)
df_hpi["date"] = pd.to_datetime(
    df_hpi["year"].astype(str) + "-"
    + (df_hpi["quarter"] * 3 - 2).astype(str) + "-01"
)
df_hpi["hpi_index"] = pd.to_numeric(df_hpi["hpi_index"], errors="coerce")
df_hpi["hpi_price"] = pd.to_numeric(df_hpi["hpi_price"], errors="coerce")
df_hpi = df_hpi[
    ["date", "year", "quarter", "hpi_index", "hpi_price"]
].reset_index(drop=True)

base_nationwide = df_hpi[df_hpi["year"] == 2015]["hpi_index"].mean()
df_hpi["hpi_index_rebased"] = df_hpi["hpi_index"] / base_nationwide * 100

print(f"[4] Nationwide HPI: {df_hpi.shape[0]} quarters, "
      f"{df_hpi.date.min().date()} to {df_hpi.date.max().date()}")


# ── 5. Bank of England Base Rate (monthly → quarterly) ────────────────────────
df_rate = pd.read_csv(RAW / "boe_base_rate_1975.csv")
df_rate.columns = ["date_raw", "base_rate"]
df_rate["date_raw"] = pd.to_datetime(df_rate["date_raw"], format="%d %b %y")
df_rate["base_rate"] = pd.to_numeric(df_rate["base_rate"], errors="coerce")
df_rate["year"] = df_rate["date_raw"].dt.year
df_rate["quarter"] = df_rate["date_raw"].dt.quarter
df_rate["date"] = pd.to_datetime(
    df_rate["year"].astype(str) + "-"
    + (df_rate["quarter"] * 3 - 2).astype(str) + "-01"
)
df_rate = (
    df_rate.groupby(["date", "year", "quarter"])["base_rate"]
    .mean()
    .reset_index()
)
print(f"[5] Base rate: {df_rate.shape[0]} quarters, "
      f"{df_rate.date.min().date()} to {df_rate.date.max().date()}")


# ── 6. Transactions splice ─────────────────────────────────────────────────────
# Post-2005: HMRC England (non-SA); pre-2005: BoE UK approvals scaled to
# England-equivalent using 2005–2007 linking factor (pre-GFC window only).
# The splice dummy flags the pre-2005 approvals-based observations.

# C1. HMRC England residential transactions (non-SA, 2005 Q2 onwards)
df_hmrc = pd.read_excel(
    RAW / "transaction_costs_splice/govuk_transactions_2005.ods",
    sheet_name="Residential_quarterly",
    engine="odf",
    header=None,
)
df_hmrc.columns = df_hmrc.iloc[5].tolist()
df_hmrc = df_hmrc.iloc[6:].copy()
df_hmrc = df_hmrc.dropna(subset=["Calendar Quarter"])
df_hmrc = df_hmrc[
    df_hmrc["Calendar Quarter"].astype(str).str.match(r"^\d{4} Quarter \d$")
].copy()
df_hmrc["year"] = df_hmrc["Calendar Quarter"].str[:4].astype(int)
df_hmrc["quarter"] = df_hmrc["Calendar Quarter"].str[-1].astype(int)
df_hmrc["date"] = pd.to_datetime(
    df_hmrc["year"].astype(str) + "-"
    + (df_hmrc["quarter"] * 3 - 2).astype(str) + "-01"
)
df_hmrc["transactions_england"] = pd.to_numeric(
    df_hmrc["England"], errors="coerce"
)
df_hmrc = df_hmrc[
    ["date", "year", "quarter", "transactions_england"]
].reset_index(drop=True)

# C2. BoE mortgage approvals (UK-wide, 1987–present)
df_approvals = pd.read_csv(
    RAW / "transaction_costs_splice/boe_mortgage_approvals_1987.csv"
)
df_approvals.columns = ["date_raw", "approvals"]
df_approvals["date_raw"] = pd.to_datetime(
    df_approvals["date_raw"], format="%d %b %y"
)
df_approvals["approvals"] = pd.to_numeric(
    df_approvals["approvals"], errors="coerce"
)
df_approvals["year"] = df_approvals["date_raw"].dt.year
df_approvals["quarter"] = df_approvals["date_raw"].dt.quarter
df_approvals["date"] = pd.to_datetime(
    df_approvals["year"].astype(str) + "-"
    + (df_approvals["quarter"] * 3 - 2).astype(str) + "-01"
)
df_approvals = df_approvals[
    ["date", "year", "quarter", "approvals"]
].reset_index(drop=True)

# C3. Linking factor: BoE approvals → England HMRC transactions
# 2005–2007 window only; avoids GFC period where approvals/completions diverge
overlap = df_hmrc[
    (df_hmrc["year"] >= 2005) & (df_hmrc["year"] <= 2007)
].merge(df_approvals, on=["date", "year", "quarter"])
scale_factor = overlap["transactions_england"].mean() / overlap["approvals"].mean()
print(f"[6] Transactions scale factor (approvals → England): {scale_factor:.4f}")

df_approvals["transactions_england"] = df_approvals["approvals"] * scale_factor

# C4. Combine
df_pre2005 = df_approvals[df_approvals["year"] < 2005][
    ["date", "year", "quarter", "transactions_england"]
].copy()
df_transactions = pd.concat(
    [df_pre2005, df_hmrc], ignore_index=True
).sort_values("date")
df_transactions["transactions_england"] = (
    df_transactions["transactions_england"].interpolate(method="linear")
)
# Flag: 1 = pre-2005 approvals-based (also UK-wide geographically)
#        0 = post-2005 HMRC England
#      NaN = pre-1987, no transactions data at all
df_transactions["transactions_spliced_flag"] = (
    df_transactions["year"] < 2005
).astype("Int64")

print(f"    Spliced transactions: {df_transactions.shape[0]} quarters, "
      f"{df_transactions.date.min().date()} to {df_transactions.date.max().date()}")
print(f"    Pre-2005 (approvals): {df_transactions['transactions_spliced_flag'].sum()}, "
      f"Post-2005 (HMRC England): {(df_transactions['transactions_spliced_flag']==0).sum()}")


# ── 7. Housing Stock (annual → quarterly, England) ────────────────────────────
df_stock = pd.read_excel(
    RAW / "govuk_housingstock_annual.ods",
    sheet_name="LT_104",
    engine="odf",
    header=None,
)
df_stock = df_stock.iloc[4:].copy()
df_stock.columns = [
    "date_type", "year", "owner_occ", "private_rent",
    "social_rent", "LA_rent", "other", "total_stock", "notes",
]
df_stock = df_stock[
    pd.to_numeric(df_stock["year"], errors="coerce").notna()
].copy()
df_stock["year"] = df_stock["year"].astype(int)
df_stock["total_stock"] = pd.to_numeric(df_stock["total_stock"], errors="coerce")
df_stock = df_stock[df_stock["year"] >= 1975][["year", "total_stock"]].dropna()
df_stock = df_stock.groupby("year")["total_stock"].last().reset_index()
df_stock["date"] = pd.to_datetime(df_stock["year"].astype(str) + "-01-01")
df_stock = df_stock.set_index("date")[["total_stock"]]

full_q_index = pd.date_range(start="1975-01-01", end="2025-07-01", freq="QS")
df_stock_q = df_stock.reindex(df_stock.index.union(full_q_index))
df_stock_q = df_stock_q.interpolate(method="cubic")

# Forward-extrapolate final quarters where annual data has run out.
# Uses the average quarterly increment from the last 4 observed annual steps.
last_valid = df_stock_q.loc[df_stock_q["total_stock"].notna()].index[-1]
if df_stock_q["total_stock"].isna().any():
    last_4_annual = (
        df_stock.iloc[-4:]["total_stock"].diff().dropna().mean() / 4
    )
    null_idx = df_stock_q.index[df_stock_q["total_stock"].isna()]
    for i, idx in enumerate(null_idx, start=1):
        df_stock_q.loc[idx, "total_stock"] = (
            df_stock_q.loc[last_valid, "total_stock"] + i * last_4_annual
        )

df_stock_q = df_stock_q.loc[full_q_index].reset_index()
df_stock_q.columns = ["date", "housing_stock_eng"]
df_stock_q["year"] = df_stock_q["date"].dt.year
df_stock_q["quarter"] = df_stock_q["date"].dt.quarter

print(f"[7] Housing stock: {df_stock_q.shape[0]} quarters, "
      f"{df_stock_q.date.min().date()} to {df_stock_q.date.max().date()}, "
      f"nulls: {df_stock_q['housing_stock_eng'].isna().sum()}")


# ── 8. Construction Costs splice (GB-level) ───────────────────────────────────
# BIS Private Housing OPI pre-2014, ONS Housing OPI from 2014.
# GB-level throughout; no England-specific series with sufficient length exists.

BASE_CC = RAW / "construction_costs_splice"

bis = pd.read_excel(
    BASE_CC / "bis_const_cost_1990.xls", sheet_name="Table 1", header=None
)
df_bis = bis.iloc[69:307, [1, 2, 5]].copy()
df_bis.columns = ["year", "quarter", "cc_bis"]
df_bis["year"] = pd.to_numeric(df_bis["year"], errors="coerce").ffill()
df_bis["quarter"] = df_bis["quarter"].astype(str).str.extract(r"(\d)").astype(int)
df_bis["cc_bis"] = pd.to_numeric(df_bis["cc_bis"], errors="coerce")
df_bis = df_bis.dropna().astype({"year": int, "quarter": int})
df_bis["date"] = pd.to_datetime(
    df_bis["year"].astype(str) + "-"
    + (df_bis["quarter"] * 3 - 2).astype(str) + "-01"
)
print(f"[8] BIS construction costs: {len(df_bis)} quarters, "
      f"{df_bis.date.min().date()} to {df_bis.date.max().date()}")

ons_cc = pd.read_excel(
    BASE_CC / "ons_const_costs_2014.xlsx", sheet_name="New work", header=None
)
df_ons_cc = ons_cc.iloc[5:, [0, 1]].copy()
df_ons_cc.columns = ["period", "cc_ons"]
df_ons_cc["cc_ons"] = pd.to_numeric(df_ons_cc["cc_ons"], errors="coerce")
df_ons_cc = df_ons_cc.dropna()
df_ons_cc["year"] = (
    df_ons_cc["period"].astype(str).str.extract(r"(\d{4})").ffill().astype(int)
)
df_ons_cc["month"] = pd.to_datetime(
    df_ons_cc["period"].astype(str).str.extract(r"([A-Za-z]{3})")[0], format="%b"
).dt.month
df_ons_cc["quarter"] = ((df_ons_cc["month"] - 1) // 3) + 1
df_ons_cc = df_ons_cc.groupby(["year", "quarter"])["cc_ons"].mean().reset_index()
df_ons_cc["date"] = pd.to_datetime(
    df_ons_cc["year"].astype(str) + "-"
    + (df_ons_cc["quarter"] * 3 - 2).astype(str) + "-01"
)

overlap_cc = df_bis.merge(df_ons_cc, on=["year", "quarter"])
linking_factor = (
    overlap_cc[overlap_cc["year"] == 2014]["cc_bis"].mean()
    / overlap_cc[overlap_cc["year"] == 2014]["cc_ons"].mean()
)
print(f"    Linking factor BIS/ONS (2014): {linking_factor:.4f}")
df_ons_cc["cc_linked"] = df_ons_cc["cc_ons"] * linking_factor

pre_cc = (
    df_bis[df_bis["year"] < 2014][["date", "year", "quarter", "cc_bis"]]
    .rename(columns={"cc_bis": "cc_nominal"})
)
post_cc = (
    df_ons_cc[df_ons_cc["year"] >= 2014][["date", "year", "quarter", "cc_linked"]]
    .rename(columns={"cc_linked": "cc_nominal"})
)
df_cc = pd.concat([pre_cc, post_cc], ignore_index=True).sort_values("date")
df_cc["cc_spliced_flag"] = (df_cc["year"] < 2014).astype(int)
print(f"    Spliced CC: {len(df_cc)} quarters, "
      f"{df_cc.date.min().date()} to {df_cc.date.max().date()}")


# ── 9. Price Deflator — CPIH spliced with historical modelled CPIH pre-1988 ──
#
# Source: ONS Consumer Price Indices (MM23) extended dataset
#   ons_cpih_extended.csv  (place in data/raw/)
#
# Series extracted from the MM23 CSV (identified by CDID):
#   L522  — CPIH INDEX 00: ALL ITEMS 2015=100        (1988 JAN – present)
#   JF4D  — CPIH HISTORICAL MODELLED INDEX 00:       (1949 JAN – 1988 DEC)
#             ALL ITEMS 1965=100
#
# The historical modelled series is ONS's own back-modelled estimate and is
# labelled "indicative purposes only".  It is methodologically preferable to
# RPI for this purpose because it shares the CPIH basket and OOH treatment.
#
# Splice methodology:
#   1. Extract monthly values for both series from the MM23 CSV.
#   2. Compute a linking factor = mean(L522 1988Q1) / mean(JF4D 1988Q1).
#   3. Scale JF4D by the linking factor to align with the 2015=100 base.
#   4. Concatenate: pre-1988 uses scaled JF4D, 1988+ uses L522.
#   5. Aggregate to quarterly means.
# The result is a continuous deflator from 1949 Q1 (in practice we only
# need from 1975 Q1 for the VECM sample).

import csv as _csv
import re as _re

_CPIH_FILE = RAW / "ons_cpih_extended.csv"

# Column positions in the MM23 CSV (0-indexed after the first column):
#   JF4D is at position 2316, L522 at position 2406.
# These are validated against the CDID row — re-check if ONS updates the file.
_COL_HIST = 2316   # JF4D  CPIH historical modelled 1965=100
_COL_L522 = 2406   # L522  CPIH all items 2015=100

def _parse_mm23_monthly(filepath, col_hist, col_l522):
    """
    Parse the ONS MM23 CSV, returning a DataFrame with columns:
      period, date, cpih_hist (JF4D), cpih_l522 (L522)
    Skips the 8 header rows and only returns rows matching YYYY MON format.
    """
    rows = []
    with open(filepath, "r", encoding="utf-8-sig") as fh:
        reader = _csv.reader(fh)
        for row_num, row in enumerate(reader):
            if row_num < 8:
                continue
            if not row or not row[0].strip():
                continue
            period = row[0].strip()
            if not _re.match(r"^\d{4}\s[A-Z]{3}$", period):
                continue
            hist = row[col_hist] if col_hist < len(row) else ""
            l522 = row[col_l522] if col_l522 < len(row) else ""
            rows.append([period, hist, l522])
    df = pd.DataFrame(rows, columns=["period", "cpih_hist", "cpih_l522"])
    df["cpih_hist"] = pd.to_numeric(df["cpih_hist"], errors="coerce")
    df["cpih_l522"] = pd.to_numeric(df["cpih_l522"], errors="coerce")
    df["date"]    = pd.to_datetime(df["period"], format="%Y %b")
    df["year"]    = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    return df.sort_values("date").reset_index(drop=True)

df_mm23 = _parse_mm23_monthly(_CPIH_FILE, _COL_HIST, _COL_L522)

# Linking factor: scale JF4D so it matches L522 at the splice point (1988 Q1)
_splice = df_mm23[
    (df_mm23["date"] >= "1988-01-01") & (df_mm23["date"] <= "1988-03-31")
]
_hist_scale = _splice["cpih_l522"].mean() / _splice["cpih_hist"].mean()
print(f"[9] CPIH splice: linking factor (L522/JF4D at 1988 Q1) = {_hist_scale:.6f}")

df_mm23["cpih_scaled_hist"] = df_mm23["cpih_hist"] * _hist_scale
df_mm23["cpih_spliced"] = np.where(
    df_mm23["date"] >= "1988-01-01",
    df_mm23["cpih_l522"],
    df_mm23["cpih_scaled_hist"],
)

# Quarterly means
df_cpi_q = (
    df_mm23
    .groupby(["year", "quarter"])["cpih_spliced"]
    .mean()
    .reset_index()
    .rename(columns={"cpih_spliced": "cpih"})
)

# Flag pre-1988 observations (scaled historical modelled, indicative only)
df_cpi_q["cpih_hist_flag"] = (df_cpi_q["year"] < 1988).astype(int)

base_cpih = df_cpi_q[df_cpi_q["year"] == 2015]["cpih"].mean()
_valid = df_cpi_q["cpih"].notna()
print(f"    CPIH deflator: {_valid.sum()} quarters, "
      f"{df_cpi_q.loc[_valid,'year'].min()} Q{df_cpi_q.loc[_valid & (df_cpi_q.year==df_cpi_q.loc[_valid,'year'].min()), 'quarter'].min()} "
      f"to {df_cpi_q.loc[_valid,'year'].max()} Q{df_cpi_q.loc[_valid & (df_cpi_q.year==df_cpi_q.loc[_valid,'year'].max()), 'quarter'].max()}")
print(f"    Pre-1988 (hist modelled, indicative): "
      f"{df_cpi_q['cpih_hist_flag'].sum()} quarters; "
      f"Post-1988 (official CPIH L522): {(df_cpi_q['cpih_hist_flag']==0).sum()} quarters")


# ── 10. Merge all series ──────────────────────────────────────────────────────
df_master = df_sc.copy()
df_master = df_master.merge(df_hpi,          on=["date", "year", "quarter"], how="left")
df_master = df_master.merge(df_rate,         on=["date", "year", "quarter"], how="left")
df_master = df_master.merge(df_transactions, on=["date", "year", "quarter"], how="left")
df_master = df_master.merge(df_stock_q,      on=["date", "year", "quarter"], how="left")
df_master = df_master.merge(df_epc,          on=["date", "year", "quarter"], how="left")
df_master = df_master.merge(df_cpi_q,        on=["year", "quarter"],         how="left")
# cpih_hist_flag=1 flags quarters where the deflator uses the ONS historical
# modelled CPIH (pre-1988, indicative only rather than National Statistics)

# Real HPI
df_master["hpi_real"] = (
    df_master["hpi_index_rebased"] / (df_master["cpih"] / base_cpih)
)
df_master["lrprc"] = np.log(df_master["hpi_real"])

# Real construction costs
df_cc_defl = df_cc.merge(df_cpi_q, on=["year", "quarter"], how="left")
base_cc = df_cc_defl[df_cc_defl["year"] == 2015]["cc_nominal"].mean()
df_cc_defl["cc_real"] = (
    (df_cc_defl["cc_nominal"] / base_cc)
    / (df_cc_defl["cpih"] / base_cpih)
    * 100
)
df_cc_defl["lrcc"] = np.log(df_cc_defl["cc_real"])

df_master = df_master.merge(
    df_cc_defl[["date", "year", "quarter", "cc_real", "lrcc", "cc_spliced_flag"]],
    on=["date", "year", "quarter"],
    how="left",
)
df_master = df_master.drop(columns=["cpih"])

# Log transforms for VECM variables
df_master["lhstarts"]  = np.log(df_master["starts_private"])
df_master["lvol"]      = np.log(df_master["transactions_england"])
df_master["lstock"]    = np.log(df_master["housing_stock_eng"])


# ── 11. Save ──────────────────────────────────────────────────────────────────
df_master.to_csv(OUT / "quarterly_master.csv", index=False)
df_net.to_csv(OUT / "annual_net_additions.csv", index=False)

print(f"\n{'='*55}")
print(f"Saved quarterly_master.csv:    {df_master.shape}")
print(f"Saved annual_net_additions.csv: {df_net.shape}")

# Summary of VECM-ready window
vecm_cols = ["lhstarts", "lrprc", "lvol", "base_rate", "lstock", "lrcc"]
complete = df_master[vecm_cols].notna().all(axis=1)
print(f"\nVECM-ready observations (all core vars non-null):")
print(f"  {complete.sum()} quarters — "
      f"{df_master.loc[complete, 'date'].min().strftime('%Y-%m-%d')} to "
      f"{df_master.loc[complete, 'date'].max().strftime('%Y-%m-%d')}")

_pre88 = df_master.loc[complete, "cpih_hist_flag"].sum() if "cpih_hist_flag" in df_master.columns else 0
if _pre88:
    print(f"  Of which pre-1988 (hist modelled deflator, indicative): {_pre88} quarters")

print(f"\nNon-null counts (key vars):")
for col in vecm_cols + ["epc_lodgements", "transactions_spliced_flag", "cc_spliced_flag", "cpih_hist_flag"]:
    n = df_master[col].notna().sum() if col in df_master.columns else "N/A"
    print(f"  {col:<30} {n}")
