from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_sku(sku):
    if pd.isna(sku):
        return ""
    s = str(sku).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = s.replace(" ", "").replace("-", "")
    return s


def parse_money(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return np.nan
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def read_csv_flexible(upload):
    def _read(**kwargs):
        upload.seek(0)
        return pd.read_csv(upload, **kwargs)

    last_err = None
    for kwargs in [{}, {"sep": None, "engine": "python"}, {"sep": None, "engine": "python", "encoding": "latin1"}]:
        try:
            df = _read(**kwargs)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise last_err

    if df.shape[1] == 1:
        for sep in [";", "\t", "|"]:
            try:
                df2 = _read(sep=sep)
                if df2.shape[1] > 1:
                    return df2
            except Exception:
                continue
    return df


def map_pricebook_columns(df: pd.DataFrame) -> dict:
    sku_aliases = {"Sku", "SKU", "UPC", "UPC Code", "UPC/PLU", "UPC PLU", "Barcode", "PLU", "ItemCode", "Item Code"}
    name_aliases = {"Name", "Description", "Item Name"}
    retail_price_aliases = {"RetailPrice", "Retail Price", "Retail", "Price", "Sell Price"}
    unit_cost_aliases = {"UnitCost", "Unit Cost", "Cost", "Avg Cost"}

    df_cols_lower = {col.lower(): col for col in df.columns}

    def _find(alias_set):
        for alias in alias_set:
            if alias.lower() in df_cols_lower:
                return df_cols_lower[alias.lower()]
        return None

    sku_col = _find(sku_aliases)
    name_col = _find(name_aliases)
    retail_col = _find(retail_price_aliases)
    cost_col = _find(unit_cost_aliases)

    missing = []
    if not sku_col:
        missing.append("UPC/PLU (Sku/UPC/PLU)")
    if not name_col:
        missing.append("Name")
    if not retail_col:
        missing.append("Retail Price")
    if not cost_col:
        missing.append("Unit Cost")

    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    return {
        "Sku": sku_col,
        "Name": name_col,
        "RetailPrice": retail_col,
        "UnitCost": cost_col,
    }
