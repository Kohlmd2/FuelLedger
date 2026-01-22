
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import calendar

# Optional (recommended) grid component for persistent column widths.
# Install with: pip install streamlit-aggrid
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
except Exception:
    AgGrid = None
    GridOptionsBuilder = None
    GridUpdateMode = None
    DataReturnMode = None
    JsCode = None

# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="Fuel Profit Tracker", layout="wide")

DATA_DIR = Path(".fuel_profit_data")
DATA_DIR.mkdir(exist_ok=True)

HISTORY_FILE = DATA_DIR / "daily_totals_history.csv"          # fuel daily totals (all grades)
FIXED_COSTS_FILE = DATA_DIR / "fixed_costs.csv"               # monthly fixed costs
STORE_DAILY_FILE = DATA_DIR / "store_daily.csv"               # inside-store daily numbers
TANK_DELIVERIES_FILE = DATA_DIR / "tank_deliveries.csv"       # deliveries log
TANK_BASELINE_FILE = DATA_DIR / "tank_baseline.csv"           # starting tank levels & avg cost

# ============================================================
# Formatting helpers
# ============================================================

def fmt_currency(x):
    return f"${x:,.2f}" if pd.notna(x) else ""

# AgGrid JS formatter (used only if st_aggrid is installed)
if JsCode is not None:
    CURRENCY_JS = JsCode("""
        function(params) {
            if (params.value === null || params.value === undefined || params.value === "") {
                return "";
            }
            const num = Number(params.value);
            if (isNaN(num)) {
                return params.value;
            }
            return num.toLocaleString(undefined, {
                style: "currency",
                currency: "USD",
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            });
        }
    """)
else:
    CURRENCY_JS = None
def fmt_percent(x):
    return f"{x:.2%}" if pd.notna(x) else ""

def fmt_number(x):
    return f"{x:,.3f}" if pd.notna(x) else ""

# For conditional coloring in tables (Streamlit dataframe styling)

def _color_profit(v):
    if pd.isna(v):
        return ""
    if v > 0:
        return "color: #00c853; font-weight: 700;"  # green
    if v < 0:
        return "color: #ff5252; font-weight: 700;"  # red
    return ""

def _color_diff(v):
    if pd.isna(v):
        return ""
    return "color: #00c853;" if v >= 0 else "color: #ff5252;"

# ============================================================
# Fuel logic
# ============================================================

GRADE_MAP = {
    "REGUNL": 87,
    "PLSUNL": 89,
    "SUPUNL": 93,
}

CREDIT_TENDERS = {"creditCards", "generic", "creditCards"}
CASH_TENDERS = {"cash", "debitCards"}


def require_columns(df: pd.DataFrame, needed: list[str]) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        st.stop()


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    needed = ["Date Sold", "Grade Name", "Gallons Sold", "Final Amount", "Primary Tender Code"]
    require_columns(df, needed)

    df["Date"] = pd.to_datetime(df["Date Sold"], errors="coerce").dt.date
    df["Grade"] = df["Grade Name"].map(GRADE_MAP)

    tender = df["Primary Tender Code"].astype(str).str.strip()
    df["TenderType"] = np.where(
        tender.eq("cash"), "CASH",
        np.where(tender.eq("debitCards"), "DEBIT",
                 np.where(tender.isin(CREDIT_TENDERS) | tender.eq("creditCards"), "CREDIT", "OTHER"))
    )
    df["PriceTier"] = np.where(df["TenderType"].eq("CREDIT"), "CREDIT", "CASH")

    df["Gallons"] = pd.to_numeric(df["Gallons Sold"], errors="coerce").fillna(0.0)
    df["FinalAmount"] = pd.to_numeric(df["Final Amount"], errors="coerce").fillna(0.0)

    df = df[df["Date"].notna()]
    df = df[df["Grade"].isin([87, 89, 93])]

    return df[["Date", "Grade", "Gallons", "FinalAmount", "TenderType", "PriceTier"]]


def summarize_by_day(df_clean: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df_clean
        .groupby(["Date", "Grade", "PriceTier"], as_index=False)
        .agg(Gallons=("Gallons", "sum"), POSAmount=("FinalAmount", "sum"))
    )

    out = pivot.pivot_table(
        index=["Date", "Grade"],
        columns="PriceTier",
        values=["Gallons", "POSAmount"],
        aggfunc="sum",
        fill_value=0.0
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
    """Normalize Date-like columns to pandas midnight Timestamps for consistent joins."""
    if df is None or df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def build_profit_table(summary: pd.DataFrame, prices: pd.DataFrame, costs: pd.DataFrame, credit_fee_rate: float) -> pd.DataFrame:
    """
    Build profit table by applying *as-of* posted Prices/Costs to each sale date.
    For each Grade, we use the most recent posted price/cost on or before the sale Date.
    """
    if summary is None or summary.empty:
        return summary

    # Normalize types
    profit = summary.copy()
    profit = normalize_date_column(profit, "Date")
    prices_n = normalize_date_column(prices.copy() if prices is not None else pd.DataFrame(), "Date")
    costs_n = normalize_date_column(costs.copy() if costs is not None else pd.DataFrame(), "Date")

    # Coerce Grade to numeric (nullable int) across all frames
    for df in (profit, prices_n, costs_n):
        if df is not None and not df.empty and "Grade" in df.columns:
            df["Grade"] = pd.to_numeric(df["Grade"], errors="coerce").astype("Int64")

    # Ensure expected columns exist
    if prices_n is None or prices_n.empty:
        prices_n = pd.DataFrame(columns=["Date", "Grade", "CashPrice", "CreditPrice"])
    if "PricePerGallon" in prices_n.columns and not {"CashPrice", "CreditPrice"}.issubset(prices_n.columns):
        prices_n["CashPrice"] = prices_n["PricePerGallon"]
        prices_n["CreditPrice"] = prices_n.get("CreditPrice", prices_n["PricePerGallon"])

    if costs_n is None or costs_n.empty:
        costs_n = pd.DataFrame(columns=["Date", "Grade", "CostPerGallon"])

    # As-of merge helper (per grade)
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

    profit = _asof_by_grade(profit, prices_n[["Date", "Grade", "CashPrice", "CreditPrice"]].dropna(subset=["Grade"]), ["CashPrice", "CreditPrice"])
    profit = _asof_by_grade(profit, costs_n[["Date", "Grade", "CostPerGallon"]].dropna(subset=["Grade"]), ["CostPerGallon"])

    # Fill missing postings with 0.0 so calculations don't break; UI can warn elsewhere.
    for c in ["CashPrice", "CreditPrice", "CostPerGallon"]:
        if c not in profit.columns:
            profit[c] = 0.0
        profit[c] = pd.to_numeric(profit[c], errors="coerce").fillna(0.0)

    # Calculations (matches original spreadsheet logic)
    profit["ExpectedRevenue"] = profit["Gallons_CASH"] * profit["CashPrice"] + profit["Gallons_CREDIT"] * profit["CreditPrice"]
    profit["COGS"] = profit["TotalGallons"] * profit["CostPerGallon"]
    profit["GrossProfit"] = profit["POSRevenue"] - profit["COGS"]
    profit["CreditCardFees"] = (profit["Gallons_CREDIT"] * profit["CreditPrice"]) * float(credit_fee_rate or 0.0)
    profit["NetFuelProfit"] = profit["GrossProfit"] - profit["CreditCardFees"]
    profit["MarginPerGallon"] = profit["NetFuelProfit"] / profit["TotalGallons"].replace({0: np.nan})
    profit["MarginPerGallon"] = profit["MarginPerGallon"].fillna(0.0)
    profit["RevenueDiff_POS_minus_Expected"] = profit["POSRevenue"] - profit["ExpectedRevenue"]

    return profit
def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _save_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def load_history() -> pd.DataFrame:
    df = _load_csv(HISTORY_FILE)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    return df


def save_history(df: pd.DataFrame) -> None:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date
    _save_csv(HISTORY_FILE, out)


def load_fixed_costs() -> pd.DataFrame:
    df = _load_csv(FIXED_COSTS_FILE)
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
    _save_csv(FIXED_COSTS_FILE, out)


# -----------------------------
# Tank baseline helpers
# -----------------------------

def load_tank_baseline() -> pd.DataFrame:
    # Columns: Grade, StartingGallons, StartingAvgCost, BaselineDate
    df = _load_csv(TANK_BASELINE_FILE)
    if df.empty:
        df = pd.DataFrame({
            "Grade": [87, 93],
            "StartingGallons": [0.0, 0.0],
            "StartingAvgCost": [0.0, 0.0],
            "BaselineDate": [pd.Timestamp.today().date().isoformat()] * 2,
        })
        _save_csv(TANK_BASELINE_FILE, df)
        return df

    # Ensure required columns exist
    if "Grade" not in df.columns:
        df["Grade"] = [87, 93][: len(df)]

    for col, default in [("StartingGallons", 0.0), ("StartingAvgCost", 0.0), ("BaselineDate", pd.Timestamp.today().date().isoformat())]:
        if col not in df.columns:
            df[col] = default

    # Keep only 87/93, sort
    df["Grade"] = pd.to_numeric(df["Grade"], errors="coerce")
    df = df[df["Grade"].isin([87, 93])].copy()
    if df.empty:
        df = pd.DataFrame({
            "Grade": [87, 93],
            "StartingGallons": [0.0, 0.0],
            "StartingAvgCost": [0.0, 0.0],
            "BaselineDate": [pd.Timestamp.today().date().isoformat()] * 2,
        })

    # Deduplicate by grade (keep last)
    df = df.sort_values(["Grade"]).drop_duplicates(subset=["Grade"], keep="last")

    # Coerce numeric
    df["StartingGallons"] = pd.to_numeric(df["StartingGallons"], errors="coerce").fillna(0.0)
    df["StartingAvgCost"] = pd.to_numeric(df["StartingAvgCost"], errors="coerce").fillna(0.0)

    # BaselineDate as string YYYY-MM-DD
    df["BaselineDate"] = df["BaselineDate"].astype(str)

    # Ensure both grades exist
    for g in [87, 93]:
        if g not in set(df["Grade"].tolist()):
            df = pd.concat([
                df,
                pd.DataFrame({
                    "Grade": [g],
                    "StartingGallons": [0.0],
                    "StartingAvgCost": [0.0],
                    "BaselineDate": [pd.Timestamp.today().date().isoformat()],
                })
            ], ignore_index=True)

    df = df.sort_values("Grade").reset_index(drop=True)
    _save_csv(TANK_BASELINE_FILE, df)
    return df


def save_tank_baseline(df: pd.DataFrame) -> None:
    out = df.copy()
    # enforce schema
    out = out[["Grade", "StartingGallons", "StartingAvgCost", "BaselineDate"]].copy()
    out["Grade"] = pd.to_numeric(out["Grade"], errors="coerce")
    out = out[out["Grade"].isin([87, 93])].copy()
    out["StartingGallons"] = pd.to_numeric(out["StartingGallons"], errors="coerce").fillna(0.0)
    out["StartingAvgCost"] = pd.to_numeric(out["StartingAvgCost"], errors="coerce").fillna(0.0)
    out["BaselineDate"] = out["BaselineDate"].astype(str)
    out = out.sort_values("Grade").reset_index(drop=True)
    _save_csv(TANK_BASELINE_FILE, out)



def load_store_daily() -> pd.DataFrame:
    df = _load_csv(STORE_DAILY_FILE)
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
    _save_csv(STORE_DAILY_FILE, out)

# ============================================================
# Month helpers
# ============================================================

def month_days(month_str: str) -> int:
    y, m = month_str.split("-")
    return calendar.monthrange(int(y), int(m))[1]

# ============================================================
# UI / Navigation
# ============================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Calculator", "Daily Totals History", "Tank Deliveries", "Store Profit (Day + Month)"],
    index=0,
)

# ============================================================
# Page: Calculator (Fuel)
# ============================================================

if page == "Calculator":
    st.header("Calculator")
    st.caption("Upload Petro Outlet CSV → enter posted prices & cost/gal → get fuel profit.")

    uploaded = st.file_uploader("Upload Petro Outlet CSV", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV to begin.")
        st.stop()

    raw = pd.read_csv(uploaded)
    with st.expander("Preview RAW CSV"):
        st.dataframe(raw.head(30), use_container_width=True)

    clean = clean_transactions(raw)
    with st.expander("Preview CLEAN transactions"):
        st.dataframe(clean.head(50), use_container_width=True)

    summary = summarize_by_day(clean)

    st.subheader("Enter Prices & Costs")
    dates = sorted(summary["Date"].unique())
    grades = [87, 89, 93]

    credit_fee_rate = st.number_input(
        "Credit card fee rate (e.g., 0.0275 = 2.75%)",
        min_value=0.0,
        max_value=0.10,
        value=0.0275,
        step=0.0005,
        format="%.4f",
    )

    default_prices = pd.DataFrame([(d, g, np.nan, np.nan) for d in dates for g in grades],
                                  columns=["Date", "Grade", "CashPrice", "CreditPrice"])
    default_costs = pd.DataFrame([(d, g, np.nan) for d in dates for g in grades],
                                 columns=["Date", "Grade", "CostPerGallon"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Prices (posted)")
        prices = st.data_editor(default_prices, num_rows="fixed", use_container_width=True)
    with c2:
        st.markdown("### Costs (your cost/gal)")
        costs = st.data_editor(default_costs, num_rows="fixed", use_container_width=True)

    def has_missing(df, cols):
        return df[cols].isna().any().any()

    if has_missing(prices, ["CashPrice", "CreditPrice"]) or has_missing(costs, ["CostPerGallon"]):
        st.warning("Fill in all prices and costs to compute profit.")
        st.stop()

    profit = build_profit_table(summary, prices, costs, credit_fee_rate)

    st.subheader("Results")

    # Styled display with green/red
    view = profit.copy()

    # Format numeric columns for display
    for col in ["Gallons_CASH", "Gallons_CREDIT", "TotalGallons"]:
        view[col] = view[col].map(fmt_number)
    for col in ["CashPrice", "CreditPrice", "CostPerGallon",
                "ExpectedRevenue", "POSRevenue", "COGS",
                "GrossProfit", "CreditCardFees", "NetFuelProfit",
                "RevenueDiff_POS_minus_Expected"]:
        view[col] = view[col].map(fmt_currency)
    view["MarginPerGallon"] = profit["MarginPerGallon"].map(fmt_percent)

    st.markdown("### By Day + Grade")
    st.dataframe(view[[
        "Date", "Grade",
        "Gallons_CASH", "Gallons_CREDIT", "TotalGallons",
        "CashPrice", "CreditPrice", "CostPerGallon",
        "ExpectedRevenue", "POSRevenue", "COGS",
        "GrossProfit", "CreditCardFees", "NetFuelProfit",
        "MarginPerGallon", "RevenueDiff_POS_minus_Expected"
    ]], use_container_width=True)

    # Daily totals (all grades)
    daily = (
        profit.groupby("Date", as_index=False)
        .agg(
            TotalGallons=("TotalGallons", "sum"),
            ExpectedRevenue=("ExpectedRevenue", "sum"),
            POSRevenue=("POSRevenue", "sum"),
            COGS=("COGS", "sum"),
            GrossProfit=("GrossProfit", "sum"),
            CreditCardFees=("CreditCardFees", "sum"),
            NetFuelProfit=("NetFuelProfit", "sum"),
        )
    )
    daily["MarginPerGallon"] = np.where(daily["TotalGallons"] > 0, daily["NetFuelProfit"] / daily["TotalGallons"], np.nan)

    st.markdown("### Daily Totals (all grades)")

    dview = daily.copy()
    dview["TotalGallons"] = dview["TotalGallons"].map(fmt_number)
    for col in ["ExpectedRevenue", "POSRevenue", "COGS", "GrossProfit", "CreditCardFees", "NetFuelProfit"]:
        dview[col] = dview[col].map(fmt_currency)
    dview["MarginPerGallon"] = daily["MarginPerGallon"].map(fmt_percent)

    st.dataframe(dview.sort_values("Date"), use_container_width=True)

    # Save into history (upsert by Date)
    st.divider()
    if st.button("Save these daily totals to History"):
        hist = load_history()
        if hist.empty:
            hist = daily.copy()
        else:
            # drop rows for same date(s) then append
            existing = pd.to_datetime(hist["Date"], errors="coerce").dt.date
            incoming = pd.to_datetime(daily["Date"], errors="coerce").dt.date
            hist = hist[~existing.isin(set(incoming))]
            hist = pd.concat([hist, daily], ignore_index=True)

        save_history(hist)
        st.success("Saved. Go to Daily Totals History or Store Profit page.")

# ============================================================
# Page: Daily Totals History
# ============================================================

elif page == "Daily Totals History":
    st.header("Daily Totals History")

    history = load_history()
    if history.empty:
        st.info("No history yet. Go to Calculator, upload a CSV, compute totals, then Save to History.")
        st.stop()

    history["MarginPerGallon"] = np.where(
        history["TotalGallons"] > 0,
        history["NetFuelProfit"] / history["TotalGallons"],
        np.nan
    )

    # Display
    view = history.copy()
    view["TotalGallons"] = view["TotalGallons"].map(fmt_number)
    for col in ["ExpectedRevenue", "POSRevenue", "COGS", "GrossProfit", "CreditCardFees", "NetFuelProfit"]:
        view[col] = view[col].map(fmt_currency)
    view["MarginPerGallon"] = history["MarginPerGallon"].map(fmt_percent)

    st.dataframe(view.sort_values("Date"), use_container_width=True)

    st.download_button(
        "Download Full History CSV",
        data=history.sort_values("Date").to_csv(index=False).encode("utf-8"),
        file_name="daily_totals_history.csv",
        mime="text/csv"
    )

    st.divider()
    if st.button("Reset history (delete saved days)"):
        HISTORY_FILE.unlink(missing_ok=True)
        st.warning("History cleared. Refresh the page.")

# ============================================================
# Page: Tank Deliveries (simple log)
# ============================================================

elif page == "Tank Deliveries":
    st.header("Tank Deliveries")
    st.caption("Log deliveries for 87 and 93 only. (89 is blended 50/50 from 87 + 93.)")



    # --- Baseline (starting) tank levels ---
    st.subheader("1) Baseline (starting) tank levels")
    st.caption("Enter your measured tank levels + average fuel cost. Use **Update baseline tank levels** if you need to reset the running tank math so it lines up with reality.")

    baseline_df = load_tank_baseline()

    # One shared baseline date (stored on each row)
    default_dt = None
    try:
        if "BaselineDate" in baseline_df.columns and baseline_df["BaselineDate"].notna().any():
            default_dt = pd.to_datetime(baseline_df.loc[baseline_df["BaselineDate"].notna(), "BaselineDate"].iloc[0]).date()
    except Exception:
        default_dt = None

    baseline_date = st.date_input("Baseline date (when these levels were measured)", value=default_dt or date.today(), key="baseline_date")

    # Editable baseline numbers
    edit_baseline = baseline_df[["Grade", "StartingGallons", "StartingAvgCost"]].copy()
    edit_baseline = st.data_editor(
        edit_baseline,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Grade": st.column_config.NumberColumn(disabled=True),
            "StartingGallons": st.column_config.NumberColumn("StartingGallons", format="%.0f"),
            "StartingAvgCost": st.column_config.NumberColumn("StartingAvgCost", format="$%.4f"),
        },
        key="baseline_editor",
    )

    colb1, colb2 = st.columns(2)
    with colb1:
        if st.button("Save baseline tank levels", use_container_width=True):
            out = edit_baseline.copy()
            out["Grade"] = pd.to_numeric(out["Grade"], errors="coerce")
            out["StartingGallons"] = pd.to_numeric(out["StartingGallons"], errors="coerce").fillna(0.0)
            out["StartingAvgCost"] = pd.to_numeric(out["StartingAvgCost"], errors="coerce").fillna(0.0)
            out["BaselineDate"] = baseline_date.strftime("%Y-%m-%d")
            save_tank_baseline(out)
            st.success("Baseline saved.")
            st.rerun()

    with colb2:
        if st.button("Update baseline tank levels", use_container_width=True, type="primary"):
            out = edit_baseline.copy()
            out["Grade"] = pd.to_numeric(out["Grade"], errors="coerce")
            out["StartingGallons"] = pd.to_numeric(out["StartingGallons"], errors="coerce").fillna(0.0)
            out["StartingAvgCost"] = pd.to_numeric(out["StartingAvgCost"], errors="coerce").fillna(0.0)
            out["BaselineDate"] = baseline_date.strftime("%Y-%m-%d")
            save_tank_baseline(out)
            st.info("Baseline updated (reset point). Existing delivery history is kept; your baseline date is used as the reference.")
            st.rerun()

    st.divider()


    # Load existing
    deliveries = _load_csv(TANK_DELIVERIES_FILE)
    if deliveries.empty:
        deliveries = pd.DataFrame(columns=["Date", "Grade", "GallonsDelivered", "PricePerGallon", "Notes"])

    # Add delivery form
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 2])
    with c1:
        d = st.date_input("Delivery Date")
    with c2:
        grade = st.selectbox("Grade", [87, 93])
    with c3:
        gal = st.number_input("Gallons Delivered", min_value=0.0, value=0.0, step=50.0)
    with c4:
        ppg = st.number_input("Price/gal", min_value=0.0, value=0.0, step=0.01, format="%.4f")

    notes = st.text_input("Notes (optional)")

    if st.button("Add delivery"):
        new_row = pd.DataFrame([{
            "Date": pd.to_datetime(d),
            "Grade": int(grade),
            "GallonsDelivered": float(gal),
            "PricePerGallon": float(ppg),
            "Notes": notes,
        }])
        deliveries2 = pd.concat([deliveries, new_row], ignore_index=True)
        deliveries2["Date"] = pd.to_datetime(deliveries2["Date"], errors="coerce")
        deliveries2["Grade"] = pd.to_numeric(deliveries2["Grade"], errors="coerce").astype("Int64")
        deliveries2["GallonsDelivered"] = pd.to_numeric(deliveries2["GallonsDelivered"], errors="coerce").fillna(0.0)
        deliveries2["PricePerGallon"] = pd.to_numeric(deliveries2["PricePerGallon"], errors="coerce").fillna(0.0)
        _save_csv(TANK_DELIVERIES_FILE, deliveries2)
        st.success("Delivery saved.")
        st.rerun()

    st.subheader("Delivery history")
    if deliveries.empty:
        st.info("No deliveries logged yet.")
    else:
        dv = deliveries.copy()
        dv["Date"] = pd.to_datetime(dv["Date"], errors="coerce").dt.date
        dv["GallonsDelivered"] = pd.to_numeric(dv["GallonsDelivered"], errors="coerce").fillna(0.0).map(fmt_number)
        dv["PricePerGallon"] = pd.to_numeric(dv["PricePerGallon"], errors="coerce").fillna(0.0).map(fmt_currency)
        st.dataframe(dv.sort_values("Date", ascending=False), use_container_width=True)

# ============================================================
# Page: Store Profit (Day + Month)
# ============================================================

else:
    st.header("Store Profit (Day + Month)")
    st.caption("Combines inside-store profit + fuel profit and subtracts monthly fixed costs.")

    # ---- Month selection (keep, but default current month)
    default_month = pd.Timestamp.today().strftime("%Y-%m")
    month = st.text_input("Month (YYYY-MM)", value=default_month)

    # ---- Fixed costs editor
    st.subheader("Monthly fixed costs (rent, electric, internet, etc.)")

    fixed = load_fixed_costs()

    # Build a view for this month (and seed a few defaults if none)
    month_rows = fixed[fixed["Month"].astype(str) == str(month)].copy()
    if month_rows.empty:
        month_rows = pd.DataFrame([
            {"Month": month, "Category": "Rent", "Amount": 0.0},
            {"Month": month, "Category": "Electric", "Amount": 0.0},
            {"Month": month, "Category": "Internet", "Amount": 0.0},
        ])

    # Keep an editable copy in session_state so we can auto-fill Month on newly added rows.
    if (
        "fixed_costs_df" not in st.session_state
        or st.session_state.get("fixed_costs_month") != str(month)
    ):
        month_rows = month_rows.copy()
        month_rows["Month"] = str(month)
        st.session_state["fixed_costs_df"] = month_rows
        st.session_state["fixed_costs_month"] = str(month)

    # --------------------------------------------------------
    # Fixed costs table with persistent column widths (AgGrid)
    # --------------------------------------------------------
    st.caption("Tip: With the grid below, you can resize columns and the widths will persist while you add rows.")

    # Add/Delete controls (we manage rows in session_state for stability)
    c_add, c_del, c_help = st.columns([1, 1, 2])
    with c_add:
        if st.button("➕ Add fixed cost line"):
            st.session_state["fixed_costs_df"] = pd.concat(
                [
                    st.session_state["fixed_costs_df"],
                    pd.DataFrame([{ "Month": str(month), "Category": "", "Amount": 0.0 }]),
                ],
                ignore_index=True,
            )
            st.rerun()

    selected_rows = []

    with c_help:
        st.caption("Tip: Resize columns once; widths will persist while you add rows. Press Enter to commit a cell edit.")

    if AgGrid is None:
        st.error("For persistent column widths and reliable Enter-to-save, install AgGrid in your venv:\n\n    pip install streamlit-aggrid\n\nThen restart Streamlit (Ctrl+C, then run it again).")
        st.stop()

    # --- AgGrid editor (no auto-fit; keep user column widths) ---
    selected_month = str(month)
    month_str = selected_month

    fixed_df = st.session_state["fixed_costs_df"].copy()
    # Keep month aligned to the current selected month
    fixed_df["Month"] = selected_month

    fixed_df_for_grid = fixed_df.copy()
    fixed_df_for_grid.insert(0, "Row", range(1, len(fixed_df_for_grid) + 1))

    gb = GridOptionsBuilder.from_dataframe(fixed_df_for_grid)
    gb.configure_default_column(editable=True, resizable=True, sortable=False, filter=False)
    gb.configure_column("Row", editable=False, pinned="left", width=80)
    gb.configure_column("Month", editable=False)
    gb.configure_column("Category", editable=True)
    gb.configure_column("Amount", editable=True, type=["numericColumn"], precision=2, valueFormatter=CURRENCY_JS)
    gb.configure_selection("multiple", use_checkbox=True, header_checkbox=True)
    gb.configure_grid_options(
        stopEditingWhenCellsLoseFocus=True,
        singleClickEdit=True,
        enterMovesDownAfterEdit=True,
        enterMovesDown=True,
    )
    grid_options = gb.build()

    grid_response = AgGrid(
        fixed_df_for_grid,
        gridOptions=grid_options,
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        data_return_mode=DataReturnMode.AS_INPUT,
        fit_columns_on_grid_load=False,
        theme="streamlit",
        height=260,
        key=f"fixed_costs_grid_{month_str}",
        reload_data=False,
    )

    edited = pd.DataFrame(grid_response["data"])
    selected_rows = grid_response.get("selected_rows", [])
    fixed_edit = edited.drop(columns=["Row"], errors="ignore").copy()
    st.session_state["fixed_costs_df"] = fixed_edit

    if st.button("Save monthly fixed costs"):
        # Merge back into the full fixed-costs table (replace this month)
        fixed_other = fixed[fixed["Month"].astype(str) != str(month)].copy()
        merged = pd.concat([fixed_other, fixed_edit], ignore_index=True)
        save_fixed_costs(merged)
        st.success("Saved.")
        st.rerun()

    fixed_total = float(pd.to_numeric(fixed_edit["Amount"], errors="coerce").fillna(0.0).sum())
    st.write(f"**Total fixed costs for {month}:** {fmt_currency(fixed_total)}")

    st.divider()

    # ---- Daily inside-store inputs
    st.subheader("Daily inside-store numbers")

    store_daily = load_store_daily()
    store_daily["Date"] = pd.to_datetime(store_daily["Date"], errors="coerce")

    y, m = month.split("-")
    y, m = int(y), int(m)
    start = pd.Timestamp(y, m, 1)
    end = pd.Timestamp(y, m, month_days(month))

    store_month = store_daily[(store_daily["Date"] >= start) & (store_daily["Date"] <= end)].copy()
    store_month["Date"] = store_month["Date"].dt.date

    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.2, 1.2, 2.0])
    with c1:
        d = st.date_input("Date", value=start.date())
    with c2:
        inside_sales = st.number_input("Inside Sales ($)", min_value=0.0, value=0.0, step=100.0, format="%.2f")
    with c3:
        inside_cogs = st.number_input("Inside COGS ($)", min_value=0.0, value=0.0, step=50.0, format="%.2f")
    with c4:
        other_var = st.number_input("Other Var Costs ($)", min_value=0.0, value=0.0, step=25.0, format="%.2f")
    with c5:
        notes = st.text_input("Notes (optional)")

    if st.button("Add / Update Day"):
        new_row = pd.DataFrame([{
            "Date": pd.to_datetime(d),
            "InsideSales": inside_sales,
            "InsideCOGS": inside_cogs,
            "OtherVariableCosts": other_var,
            "Notes": notes,
        }])

        full = store_daily.copy()
        full = full[~(full["Date"].dt.date == d)]
        full = pd.concat([full, new_row], ignore_index=True).sort_values("Date")
        save_store_daily(full)

        st.success(f"Saved {d}.")
        st.rerun()

    st.markdown("### Saved days (this month)")
    if store_month.empty:
        st.info("No saved inside-store days for this month yet.")
    else:
        sm = store_month.copy()
        sm["InsideSales"] = sm["InsideSales"].map(fmt_currency)
        sm["InsideCOGS"] = sm["InsideCOGS"].map(fmt_currency)
        sm["OtherVariableCosts"] = sm["OtherVariableCosts"].map(fmt_currency)
        st.dataframe(sm.sort_values("Date"), use_container_width=True)

        del_date = st.selectbox("Delete a day", sorted(store_month["Date"].unique()))
        if st.button("Delete selected day"):
            full = load_store_daily()
            full["Date"] = pd.to_datetime(full["Date"], errors="coerce")
            full = full[full["Date"].dt.date != del_date]
            save_store_daily(full)
            st.success(f"Deleted {del_date}.")
            st.rerun()

    st.divider()

    # ---- Combine fuel + inside-store
    st.subheader("Profit summary")

    fuel_hist = load_history()
    if fuel_hist.empty:
        st.info("No fuel history yet. Go to Calculator and Save daily totals to History.")
        st.stop()

    fuel_hist["Date"] = pd.to_datetime(fuel_hist["Date"], errors="coerce").dt.date
    fuel_month = fuel_hist[(pd.to_datetime(fuel_hist["Date"]) >= start) & (pd.to_datetime(fuel_hist["Date"]) <= end)].copy()

    # Build daily table for the month
    days = pd.date_range(start, end, freq="D").date
    daily = pd.DataFrame({"Date": days})

    # Fuel profit
    fuel_month2 = fuel_month.copy()
    fuel_month2["Date"] = pd.to_datetime(fuel_month2["Date"], errors="coerce").dt.date
    daily = daily.merge(fuel_month2[["Date", "NetFuelProfit"]], on="Date", how="left")

    # Inside store profit
    store_month2 = load_store_daily()
    store_month2["Date"] = pd.to_datetime(store_month2["Date"], errors="coerce").dt.date
    store_month2 = store_month2[(pd.to_datetime(store_month2["Date"]) >= start) & (pd.to_datetime(store_month2["Date"]) <= end)].copy()

    if not store_month2.empty:
        store_month2["InsideProfit"] = (
            pd.to_numeric(store_month2["InsideSales"], errors="coerce").fillna(0.0)
            - pd.to_numeric(store_month2["InsideCOGS"], errors="coerce").fillna(0.0)
            - pd.to_numeric(store_month2["OtherVariableCosts"], errors="coerce").fillna(0.0)
        )
    else:
        store_month2 = pd.DataFrame(columns=["Date", "InsideProfit"])

    daily = daily.merge(store_month2[["Date", "InsideProfit"]], on="Date", how="left")

    daily["NetFuelProfit"] = pd.to_numeric(daily["NetFuelProfit"], errors="coerce").fillna(0.0)
    daily["InsideProfit"] = pd.to_numeric(daily["InsideProfit"], errors="coerce").fillna(0.0)

    # Allocate fixed costs evenly across days in month
    days_in_month = len(days)
    daily["FixedCostAllocated"] = fixed_total / days_in_month if days_in_month else 0.0

    daily["TotalStationProfit"] = daily["NetFuelProfit"] + daily["InsideProfit"] - daily["FixedCostAllocated"]

    # Display
    show = daily.copy()
    show["NetFuelProfit"] = show["NetFuelProfit"].map(fmt_currency)
    show["InsideProfit"] = show["InsideProfit"].map(fmt_currency)
    show["FixedCostAllocated"] = show["FixedCostAllocated"].map(fmt_currency)
    show["TotalStationProfit"] = show["TotalStationProfit"].map(fmt_currency)

    st.dataframe(show, use_container_width=True)

    # Month totals
    month_fuel = float(daily["NetFuelProfit"].replace("", 0).apply(lambda x: float(str(x).replace("$","").replace(",","")) if isinstance(x,str) else 0).sum()) if False else float(daily["NetFuelProfit"].astype(float).sum())
    month_inside = float(daily["InsideProfit"].astype(float).sum())
    month_station = float(daily["TotalStationProfit"].astype(float).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fuel profit (month)", fmt_currency(month_fuel))
    m2.metric("Inside profit (month)", fmt_currency(month_inside))
    m3.metric("Fixed costs (month)", fmt_currency(fixed_total))
    m4.metric("Total station profit (month)", fmt_currency(month_station))

