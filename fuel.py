from __future__ import annotations

import numpy as np
import pandas as pd


GRADE_MAP = {
    "REGUNL": 87,
    "PLSUNL": 89,
    "SUPUNL": 93,
}

CREDIT_TENDERS = {"creditCards", "generic"}
CASH_TENDERS = {"cash", "debitCards"}


def require_columns(df: pd.DataFrame, needed: list[str]) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    needed = ["Date Sold", "Grade Name", "Gallons Sold", "Final Amount", "Primary Tender Code"]
    require_columns(df, needed)

    df["Date"] = pd.to_datetime(df["Date Sold"], errors="coerce").dt.date
    df["Grade"] = df["Grade Name"].map(GRADE_MAP)

    tender = df["Primary Tender Code"].astype(str).str.strip()
    df["TenderType"] = np.where(
        tender.eq("cash"),
        "CASH",
        np.where(
            tender.eq("debitCards"),
            "DEBIT",
            np.where(tender.isin(CREDIT_TENDERS) | tender.eq("creditCards"), "CREDIT", "OTHER"),
        ),
    )
    df["PriceTier"] = np.where(df["TenderType"].eq("CREDIT"), "CREDIT", "CASH")

    df["Gallons"] = pd.to_numeric(df["Gallons Sold"], errors="coerce").fillna(0.0)
    df["FinalAmount"] = pd.to_numeric(df["Final Amount"], errors="coerce").fillna(0.0)

    df = df[df["Date"].notna()]
    df = df[df["Grade"].isin([87, 89, 93])]

    return df[["Date", "Grade", "Gallons", "FinalAmount", "TenderType", "PriceTier"]]


def summarize_by_day(df_clean: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df_clean.groupby(["Date", "Grade", "PriceTier"], as_index=False).agg(
            Gallons=("Gallons", "sum"), POSAmount=("FinalAmount", "sum")
        )
    )

    out = pivot.pivot_table(
        index=["Date", "Grade"],
        columns="PriceTier",
        values=["Gallons", "POSAmount"],
        aggfunc="sum",
        fill_value=0.0,
    )

    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()

    for col in ["Gallons_CASH", "Gallons_CREDIT", "POSAmount_CASH", "POSAmount_CREDIT"]:
        if col not in out.columns:
            out[col] = 0.0

    out["TotalGallons"] = out["Gallons_CASH"] + out["Gallons_CREDIT"]
    out["POSRevenue"] = out["POSAmount_CASH"] + out["POSAmount_CREDIT"]

    return out.sort_values(["Date", "Grade"])


def normalize_date_column(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def build_profit_table(summary: pd.DataFrame, prices: pd.DataFrame, costs: pd.DataFrame, credit_fee_rate: float) -> pd.DataFrame:
    if summary is None or summary.empty:
        return summary

    profit = summary.copy()
    profit = normalize_date_column(profit, "Date")
    prices_n = normalize_date_column(prices.copy() if prices is not None else pd.DataFrame(), "Date")
    costs_n = normalize_date_column(costs.copy() if costs is not None else pd.DataFrame(), "Date")

    for df in (profit, prices_n, costs_n):
        if df is not None and not df.empty and "Grade" in df.columns:
            df["Grade"] = pd.to_numeric(df["Grade"], errors="coerce").astype("Int64")

    if prices_n is None or prices_n.empty:
        prices_n = pd.DataFrame(columns=["Date", "Grade", "CashPrice", "CreditPrice"])
    if "PricePerGallon" in prices_n.columns and not {"CashPrice", "CreditPrice"}.issubset(prices_n.columns):
        prices_n["CashPrice"] = prices_n["PricePerGallon"]
        prices_n["CreditPrice"] = prices_n.get("CreditPrice", prices_n["PricePerGallon"])

    if costs_n is None or costs_n.empty:
        costs_n = pd.DataFrame(columns=["Date", "Grade", "CostPerGallon"])

    def _asof_by_grade(base: pd.DataFrame, postings: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
        out_parts = []
        for g, s in base.groupby("Grade", dropna=False):
            s2 = s.sort_values("Date")
            p2 = postings[postings["Grade"] == g].sort_values("Date") if (postings is not None and not postings.empty) else postings
            if p2 is None or p2.empty:
                merged = s2.copy()
                for c in value_cols:
                    merged[c] = pd.NA
            else:
                merged = pd.merge_asof(
                    s2,
                    p2[["Date"] + value_cols],
                    on="Date",
                    direction="backward",
                    allow_exact_matches=True,
                )
            out_parts.append(merged)
        out = pd.concat(out_parts, ignore_index=True) if out_parts else base.copy()
        return out

    profit = _asof_by_grade(
        profit,
        prices_n[["Date", "Grade", "CashPrice", "CreditPrice"]].dropna(subset=["Grade"]),
        ["CashPrice", "CreditPrice"],
    )
    profit = _asof_by_grade(
        profit,
        costs_n[["Date", "Grade", "CostPerGallon"]].dropna(subset=["Grade"]),
        ["CostPerGallon"],
    )

    if not profit.empty and "Grade" in profit.columns:
        mask_89 = profit["Grade"] == 89
        if mask_89.any():
            base_89 = profit.loc[mask_89, ["Date"]].copy()
            base_89["__idx"] = base_89.index
            base_89 = base_89.sort_values("Date")
            costs_87 = costs_n[costs_n["Grade"] == 87].sort_values("Date")
            costs_93 = costs_n[costs_n["Grade"] == 93].sort_values("Date")

            if not costs_87.empty and not costs_93.empty:
                c87 = pd.merge_asof(
                    base_89,
                    costs_87[["Date", "CostPerGallon"]],
                    on="Date",
                    direction="backward",
                    allow_exact_matches=True,
                ).rename(columns={"CostPerGallon": "Cost87"})
                c93 = pd.merge_asof(
                    base_89,
                    costs_93[["Date", "CostPerGallon"]],
                    on="Date",
                    direction="backward",
                    allow_exact_matches=True,
                ).rename(columns={"CostPerGallon": "Cost93"})
                blended = (c87["Cost87"] + c93["Cost93"]) / 2
                blended.index = base_89["__idx"].values

                fill_mask = profit.loc[mask_89, "CostPerGallon"].isna() | (profit.loc[mask_89, "CostPerGallon"] <= 0)
                profit.loc[mask_89 & fill_mask, "CostPerGallon"] = blended.loc[profit.loc[mask_89 & fill_mask].index].values

    profit["_MissingPrice"] = profit[["CashPrice", "CreditPrice"]].isna().any(axis=1)
    profit["_MissingCost"] = profit["CostPerGallon"].isna()

    for c in ["CashPrice", "CreditPrice", "CostPerGallon"]:
        if c not in profit.columns:
            profit[c] = 0.0
        profit[c] = pd.to_numeric(profit[c], errors="coerce").fillna(0.0)

    profit["ExpectedRevenue"] = profit["Gallons_CASH"] * profit["CashPrice"] + profit["Gallons_CREDIT"] * profit["CreditPrice"]
    profit["COGS"] = profit["TotalGallons"] * profit["CostPerGallon"]
    profit["GrossProfit"] = profit["POSRevenue"] - profit["COGS"]
    profit["CreditCardFees"] = (profit["Gallons_CREDIT"] * profit["CreditPrice"]) * float(credit_fee_rate or 0.0)
    profit["NetFuelProfit"] = profit["GrossProfit"] - profit["CreditCardFees"]
    profit["MarginPerGallon"] = profit["NetFuelProfit"] / profit["TotalGallons"].replace({0: np.nan})
    profit["MarginPerGallon"] = profit["MarginPerGallon"].fillna(0.0)
    profit["RevenueDiff_POS_minus_Expected"] = profit["POSRevenue"] - profit["ExpectedRevenue"]

    return profit
