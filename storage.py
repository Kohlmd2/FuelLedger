from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


DATA_DIR = Path(".fuel_profit_data")
DATA_DIR.mkdir(exist_ok=True)

LEGACY_FILES = [
    "daily_totals_history.csv",
    "fixed_costs.csv",
    "store_daily.csv",
    "tank_deliveries.csv",
    "tank_baseline.csv",
    "pricebook_current.csv",
    "inside_daily_totals_history.csv",
    "invoices.csv",
    "invoice_vendors.csv",
    "inventory.csv",
    "inventory_deliveries.csv",
]


def get_user_data_dir() -> Path:
    user_id = st.session_state.get("user_id")
    if not user_id:
        return DATA_DIR
    user_dir = DATA_DIR / "users" / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def user_data_file(filename: str) -> Path:
    return get_user_data_dir() / filename


def migrate_legacy_data_if_present() -> None:
    if st.session_state.get("legacy_migrated"):
        return
    user_dir = get_user_data_dir()
    moved = False
    for name in LEGACY_FILES:
        src = DATA_DIR / name
        dst = user_dir / name
        if src.exists() and not dst.exists():
            src.rename(dst)
            moved = True
    if moved:
        st.sidebar.info("Moved existing data into your account.")
    st.session_state["legacy_migrated"] = True


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_history() -> pd.DataFrame:
    df = _load_csv(user_data_file("daily_totals_history.csv"))
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    return df


def save_history(df: pd.DataFrame) -> None:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date
    _save_csv(user_data_file("daily_totals_history.csv"), out)


def load_last_posted_prices() -> dict:
    df = _load_csv(user_data_file("last_posted_prices.csv"))
    if df.empty:
        return {}
    df["Grade"] = pd.to_numeric(df["Grade"], errors="coerce")
    df["CashPrice"] = pd.to_numeric(df["CashPrice"], errors="coerce")
    df["CreditPrice"] = pd.to_numeric(df["CreditPrice"], errors="coerce")
    result = {}
    for _, row in df.dropna(subset=["Grade"]).iterrows():
        grade = int(row["Grade"])
        result[grade] = (row.get("CashPrice"), row.get("CreditPrice"))
    return result


def save_last_posted_prices(prices_by_grade: dict) -> None:
    rows = []
    for grade, pair in prices_by_grade.items():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            continue
        rows.append({"Grade": grade, "CashPrice": pair[0], "CreditPrice": pair[1]})
    df = pd.DataFrame(rows, columns=["Grade", "CashPrice", "CreditPrice"])
    _save_csv(user_data_file("last_posted_prices.csv"), df)


def load_fixed_costs() -> pd.DataFrame:
    df = _load_csv(user_data_file("fixed_costs.csv"))
    if df.empty:
        return pd.DataFrame(columns=["Month", "Category", "Amount"])
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    return df


def save_fixed_costs(df: pd.DataFrame) -> None:
    out = df.copy()
    out["Month"] = out["Month"].astype(str)
    out["Category"] = out["Category"].astype(str)
    out["Amount"] = pd.to_numeric(out["Amount"], errors="coerce").fillna(0.0)
    _save_csv(user_data_file("fixed_costs.csv"), out)


def load_tank_baseline() -> pd.DataFrame:
    df = _load_csv(user_data_file("tank_baseline.csv"))
    if df.empty:
        df = pd.DataFrame(
            {
                "Grade": [87, 93],
                "StartingGallons": [0.0, 0.0],
                "StartingAvgCost": [0.0, 0.0],
                "BaselineDate": [pd.Timestamp.today().date().isoformat()] * 2,
            }
        )
        _save_csv(user_data_file("tank_baseline.csv"), df)
        return df

    if "Grade" not in df.columns:
        df["Grade"] = [87, 93][: len(df)]

    for col, default in [
        ("StartingGallons", 0.0),
        ("StartingAvgCost", 0.0),
        ("BaselineDate", pd.Timestamp.today().date().isoformat()),
    ]:
        if col not in df.columns:
            df[col] = default

    df["Grade"] = pd.to_numeric(df["Grade"], errors="coerce")
    df = df[df["Grade"].isin([87, 93])].copy()
    if df.empty:
        df = pd.DataFrame(
            {
                "Grade": [87, 93],
                "StartingGallons": [0.0, 0.0],
                "StartingAvgCost": [0.0, 0.0],
                "BaselineDate": [pd.Timestamp.today().date().isoformat()] * 2,
            }
        )

    df = df.sort_values(["Grade"]).drop_duplicates(subset=["Grade"], keep="last")
    df["StartingGallons"] = pd.to_numeric(df["StartingGallons"], errors="coerce").fillna(0.0)
    df["StartingAvgCost"] = pd.to_numeric(df["StartingAvgCost"], errors="coerce").fillna(0.0)
    df["BaselineDate"] = df["BaselineDate"].astype(str)

    for g in [87, 93]:
        if g not in set(df["Grade"].tolist()):
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "Grade": [g],
                            "StartingGallons": [0.0],
                            "StartingAvgCost": [0.0],
                            "BaselineDate": [pd.Timestamp.today().date().isoformat()],
                        }
                    ),
                ],
                ignore_index=True,
            )

    df = df.sort_values("Grade").reset_index(drop=True)
    _save_csv(user_data_file("tank_baseline.csv"), df)
    return df


def save_tank_baseline(df: pd.DataFrame) -> None:
    out = df.copy()
    out = out[["Grade", "StartingGallons", "StartingAvgCost", "BaselineDate"]].copy()
    out["Grade"] = pd.to_numeric(out["Grade"], errors="coerce")
    out = out[out["Grade"].isin([87, 93])].copy()
    out["StartingGallons"] = pd.to_numeric(out["StartingGallons"], errors="coerce").fillna(0.0)
    out["StartingAvgCost"] = pd.to_numeric(out["StartingAvgCost"], errors="coerce").fillna(0.0)
    out["BaselineDate"] = out["BaselineDate"].astype(str)
    out = out.sort_values("Grade").reset_index(drop=True)
    _save_csv(user_data_file("tank_baseline.csv"), out)


def load_store_daily() -> pd.DataFrame:
    df = _load_csv(user_data_file("store_daily.csv"))
    if df.empty:
        return pd.DataFrame(columns=["Date", "InsideSales", "InsideCOGS", "OtherVariableCosts", "Notes"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["InsideSales", "InsideCOGS", "OtherVariableCosts"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "Notes" not in df.columns:
        df["Notes"] = ""
    return df


def save_store_daily(df: pd.DataFrame) -> None:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for c in ["InsideSales", "InsideCOGS", "OtherVariableCosts"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    if "Notes" not in out.columns:
        out["Notes"] = ""
    _save_csv(user_data_file("store_daily.csv"), out)


def load_invoices() -> pd.DataFrame:
    df = _load_csv(user_data_file("invoices.csv"))
    if df.empty:
        return pd.DataFrame(columns=["Date", "Vendor", "Amount", "PaymentType", "InvoiceNumber", "Notes"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Vendor"] = df.get("Vendor", "").astype(str)
    df["Amount"] = pd.to_numeric(df.get("Amount", 0.0), errors="coerce").fillna(0.0)
    if "PaymentType" not in df.columns:
        df["PaymentType"] = ""
    df["PaymentType"] = df["PaymentType"].astype(str)
    if "InvoiceNumber" not in df.columns:
        df["InvoiceNumber"] = ""
    df["InvoiceNumber"] = df["InvoiceNumber"].astype(str)
    if "Notes" not in df.columns:
        df["Notes"] = ""
    return df


def save_invoices(df: pd.DataFrame) -> None:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Vendor"] = out.get("Vendor", "").astype(str)
    amount_raw = out.get("Amount", 0.0)
    amount_clean = amount_raw.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
    out["Amount"] = pd.to_numeric(amount_clean, errors="coerce").fillna(0.0)
    if "PaymentType" not in out.columns:
        out["PaymentType"] = ""
    out["PaymentType"] = out["PaymentType"].astype(str)
    if "InvoiceNumber" not in out.columns:
        out["InvoiceNumber"] = ""
    out["InvoiceNumber"] = out["InvoiceNumber"].astype(str)
    if "Notes" not in out.columns:
        out["Notes"] = ""
    out = out[["Date", "Vendor", "Amount", "PaymentType", "InvoiceNumber", "Notes"]].copy()
    _save_csv(user_data_file("invoices.csv"), out)


def load_invoice_vendors() -> pd.DataFrame:
    df = _load_csv(user_data_file("invoice_vendors.csv"))
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Vendor",
                "ContactPerson",
                "ContactPhone",
                "ContactEmail",
                "Order",
                "OrderDay",
                "DeliveryDay",
                "Notes",
            ]
        )
    for col in ["Vendor", "ContactPerson", "ContactPhone", "ContactEmail", "OrderDay", "DeliveryDay", "Notes"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).replace({"nan": ""})
    if "Order" not in df.columns:
        df["Order"] = ""
    df["Order"] = df["Order"].astype(str).replace({"nan": ""})
    df = df.fillna("")
    return df


def save_invoice_vendors(df: pd.DataFrame) -> None:
    out = df.copy()
    for col in ["Vendor", "ContactPerson", "ContactPhone", "ContactEmail", "Order", "OrderDay", "DeliveryDay", "Notes"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str).replace({"nan": ""})
    out = out.fillna("")
    out = out[
        [
            "Vendor",
            "ContactPerson",
            "ContactPhone",
            "ContactEmail",
            "Order",
            "OrderDay",
            "DeliveryDay",
            "Notes",
        ]
    ]
    _save_csv(user_data_file("invoice_vendors.csv"), out)


def load_inventory() -> pd.DataFrame:
    df = _load_csv(user_data_file("inventory.csv"))
    if df.empty:
        return pd.DataFrame(columns=["SKU", "Name", "Quantity", "UnitCost", "LastUpdated"])
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["Name"] = df.get("Name", "").astype(str)
    df["Quantity"] = pd.to_numeric(df.get("Quantity", 0), errors="coerce").fillna(0.0)
    df["UnitCost"] = pd.to_numeric(df.get("UnitCost", 0), errors="coerce").fillna(0.0)
    df["LastUpdated"] = pd.to_datetime(df.get("LastUpdated", ""), errors="coerce")
    return df


def save_inventory(df: pd.DataFrame) -> None:
    out = df.copy()
    out["SKU"] = out["SKU"].astype(str).str.strip()
    out["Name"] = out.get("Name", "").astype(str)
    out["Quantity"] = pd.to_numeric(out.get("Quantity", 0), errors="coerce").fillna(0.0)
    out["UnitCost"] = pd.to_numeric(out.get("UnitCost", 0), errors="coerce").fillna(0.0)
    out["LastUpdated"] = pd.to_datetime(out.get("LastUpdated", datetime.now()), errors="coerce")
    out = out[["SKU", "Name", "Quantity", "UnitCost", "LastUpdated"]].copy()
    out = out.drop_duplicates(subset=["SKU"], keep="last")
    _save_csv(user_data_file("inventory.csv"), out)


def load_inventory_deliveries() -> pd.DataFrame:
    df = _load_csv(user_data_file("inventory_deliveries.csv"))
    if df.empty:
        return pd.DataFrame(columns=["Date", "SKU", "Name", "Quantity", "UnitCost", "Vendor", "Notes"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["Name"] = df.get("Name", "").astype(str)
    df["Quantity"] = pd.to_numeric(df.get("Quantity", 0), errors="coerce").fillna(0.0)
    df["UnitCost"] = pd.to_numeric(df.get("UnitCost", 0), errors="coerce").fillna(0.0)
    df["Vendor"] = df.get("Vendor", "").astype(str)
    df["Notes"] = df.get("Notes", "").astype(str)
    return df


def save_inventory_deliveries(df: pd.DataFrame) -> None:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["SKU"] = out["SKU"].astype(str).str.strip()
    out["Name"] = out.get("Name", "").astype(str)
    out["Quantity"] = pd.to_numeric(out.get("Quantity", 0), errors="coerce").fillna(0.0)
    out["UnitCost"] = pd.to_numeric(out.get("UnitCost", 0), errors="coerce").fillna(0.0)
    out["Vendor"] = out.get("Vendor", "").astype(str)
    out["Notes"] = out.get("Notes", "").astype(str)
    out = out[["Date", "SKU", "Name", "Quantity", "UnitCost", "Vendor", "Notes"]].copy()
    _save_csv(user_data_file("inventory_deliveries.csv"), out)


def _parse_money(value):
    if pd.isna(value):
        return float("nan")
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return float("nan")
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def load_pricebook() -> pd.DataFrame:
    df = _load_csv(user_data_file("pricebook_current.csv"))
    if df.empty:
        return pd.DataFrame(columns=["SKU", "Name", "RetailPrice", "UnitCost"])
    if "SKU" not in df.columns and "Sku" in df.columns:
        df = df.rename(columns={"Sku": "SKU"})
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str)
    df["RetailPrice"] = df["RetailPrice"].apply(_parse_money).fillna(0.0)
    df["UnitCost"] = df["UnitCost"].apply(_parse_money).fillna(0.0)
    return df


def save_pricebook(df: pd.DataFrame) -> None:
    if "SKU" not in df.columns and "Sku" in df.columns:
        df = df.rename(columns={"Sku": "SKU"})
    out = df[["SKU", "Name", "RetailPrice", "UnitCost"]].copy()
    out["SKU"] = out["SKU"].astype(str).str.strip()
    out["Name"] = out["Name"].astype(str)
    out["RetailPrice"] = out["RetailPrice"].apply(_parse_money).fillna(0.0)
    out["UnitCost"] = out["UnitCost"].apply(_parse_money).fillna(0.0)
    out = out.drop_duplicates(subset=["SKU"], keep="last")
    _save_csv(user_data_file("pricebook_current.csv"), out)


def load_inside_daily_totals() -> pd.DataFrame:
    df = _load_csv(user_data_file("inside_daily_totals_history.csv"))
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "TotalUnits",
                "TotalCOGS",
                "RetailSalesEstimateTotal",
                "EstimatedGrossProfitTotal",
                "ActualSalesTotal",
                "ActualGrossProfitTotal",
                "CreditCardFeesTotal",
                "NetInsideProfitTotal",
                "CoverageUnitsPct",
                "MissingSkuCount",
            ]
        )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    for col in [
        "TotalUnits",
        "TotalCOGS",
        "RetailSalesEstimateTotal",
        "EstimatedGrossProfitTotal",
        "ActualSalesTotal",
        "ActualGrossProfitTotal",
        "CreditCardFeesTotal",
        "NetInsideProfitTotal",
        "CoverageUnitsPct",
        "MissingSkuCount",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def save_inside_daily_totals(df: pd.DataFrame) -> None:
    existing = load_inside_daily_totals()
    if not existing.empty:
        existing_dates = pd.to_datetime(existing["Date"], errors="coerce").dt.date
        incoming_dates = pd.to_datetime(df["Date"], errors="coerce").dt.date
        existing = existing[~existing_dates.isin(set(incoming_dates))]
    result = pd.concat([existing, df], ignore_index=True)
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").dt.date
    for col in [
        "TotalUnits",
        "TotalCOGS",
        "RetailSalesEstimateTotal",
        "EstimatedGrossProfitTotal",
        "ActualSalesTotal",
        "ActualGrossProfitTotal",
        "CreditCardFeesTotal",
        "NetInsideProfitTotal",
        "CoverageUnitsPct",
        "MissingSkuCount",
    ]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    _save_csv(user_data_file("inside_daily_totals_history.csv"), result)
