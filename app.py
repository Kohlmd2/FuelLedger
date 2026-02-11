
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import calendar
from datetime import date, datetime
import hashlib
import hmac
import re
import secrets
import sqlite3

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

AUTH_DB = DATA_DIR / "auth.db"


def _auth_conn():
    conn = sqlite3.connect(AUTH_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db() -> None:
    with _auth_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _normalize_username(username: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", username.strip().lower())


def _hash_password(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return dk.hex()


def _create_user(username: str, email: str, password: str, is_admin: bool = False) -> tuple[bool, str]:
    username_n = _normalize_username(username)
    if not username_n:
        return False, "Username must be letters, numbers, or underscores."
    if "@" not in email or "." not in email:
        return False, "Enter a valid email."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    salt_hex = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt_hex)
    try:
        with _auth_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, email, password_hash, salt, is_admin, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (username_n, email.strip().lower(), pw_hash, salt_hex, 1 if is_admin else 0),
            )
            conn.commit()
        return True, "User created."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."


def _get_user_by_login(login: str):
    login_l = login.strip().lower()
    with _auth_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (login_l, login_l),
        ).fetchone()
    return row


def _verify_user(login: str, password: str):
    row = _get_user_by_login(login)
    if not row:
        return None
    expected = row["password_hash"]
    salt = row["salt"]
    actual = _hash_password(password, salt)
    if hmac.compare_digest(expected, actual):
        return row
    return None


def _admin_exists() -> bool:
    with _auth_conn() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE is_admin = 1 LIMIT 1").fetchone()
    return row is not None


def require_login() -> None:
    init_auth_db()

    if st.session_state.get("user_id"):
        return

    if not _admin_exists():
        st.warning("Set up your first admin account to secure this app.")
        with st.form("admin_setup"):
            admin_user = st.text_input("Admin username")
            admin_email = st.text_input("Admin email")
            admin_pw = st.text_input("Admin password", type="password")
            admin_pw2 = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create admin")
        if submitted:
            if admin_pw != admin_pw2:
                st.error("Passwords do not match.")
            else:
                ok, msg = _create_user(admin_user, admin_email, admin_pw, is_admin=True)
                if ok:
                    st.success("Admin created. Please log in.")
                    st.rerun()
                else:
                    st.error(msg)
        st.stop()

    st.sidebar.subheader("Login")
    with st.sidebar.form("login_form"):
        login = st.text_input("Username or email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        row = _verify_user(login, password)
        if row:
            st.session_state["user_id"] = int(row["id"])
            st.session_state["username"] = row["username"]
            st.session_state["is_admin"] = bool(row["is_admin"])
            for k in [
                "pricebook_df",
                "pricebook_loaded_at",
                "fixed_costs_df",
                "fixed_costs_month",
            ]:
                st.session_state.pop(k, None)
            st.rerun()
        else:
            st.sidebar.error("Invalid login.")

    st.stop()


def get_user_data_dir() -> Path:
    user_id = st.session_state.get("user_id")
    if not user_id:
        return DATA_DIR
    user_dir = DATA_DIR / "users" / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def user_data_file(filename: str) -> Path:
    return get_user_data_dir() / filename


LEGACY_FILES = [
    "daily_totals_history.csv",
    "fixed_costs.csv",
    "store_daily.csv",
    "tank_deliveries.csv",
    "tank_baseline.csv",
    "pricebook_current.csv",
    "inside_daily_totals_history.csv",
]


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

CREDIT_TENDERS = {"creditCards", "generic"}
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
    # Treat OTHER as CASH for pricing so totals aren't undercounted; warn in UI if any appear.
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

    # 89 cost blending: if 89 has no cost posted, use the as-of average of 87 and 93.
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

    # Track missing postings for UI warnings before filling with 0.0.
    profit["_MissingPrice"] = profit[["CashPrice", "CreditPrice"]].isna().any(axis=1)
    profit["_MissingCost"] = profit["CostPerGallon"].isna()

    # Fill missing postings with 0.0 so calculations don't break; UI will warn.
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


# -----------------------------
# Tank baseline helpers
# -----------------------------

def load_tank_baseline() -> pd.DataFrame:
    # Columns: Grade, StartingGallons, StartingAvgCost, BaselineDate
    df = _load_csv(user_data_file("tank_baseline.csv"))
    if df.empty:
        df = pd.DataFrame({
            "Grade": [87, 93],
            "StartingGallons": [0.0, 0.0],
            "StartingAvgCost": [0.0, 0.0],
            "BaselineDate": [pd.Timestamp.today().date().isoformat()] * 2,
        })
        _save_csv(user_data_file("tank_baseline.csv"), df)
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
    _save_csv(user_data_file("tank_baseline.csv"), df)
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

# -----------------------------
# Invoice helpers
# -----------------------------

def load_invoices() -> pd.DataFrame:
    df = _load_csv(user_data_file("invoices.csv"))
    if df.empty:
        return pd.DataFrame(columns=["Date", "Vendor", "Amount", "Notes"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Vendor"] = df.get("Vendor", "").astype(str)
    df["Amount"] = pd.to_numeric(df.get("Amount", 0.0), errors="coerce").fillna(0.0)
    if "Notes" not in df.columns:
        df["Notes"] = ""
    return df


def save_invoices(df: pd.DataFrame) -> None:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Vendor"] = out.get("Vendor", "").astype(str)
    out["Amount"] = pd.to_numeric(out.get("Amount", 0.0), errors="coerce").fillna(0.0)
    if "Notes" not in out.columns:
        out["Notes"] = ""
    _save_csv(user_data_file("invoices.csv"), out)


def load_invoice_vendors() -> pd.DataFrame:
    df = _load_csv(user_data_file("invoice_vendors.csv"))
    if df.empty:
        return pd.DataFrame(columns=[
            "Vendor",
            "ContactPerson",
            "ContactPhone",
            "ContactEmail",
            "Order",
            "OrderDay",
            "DeliveryDay",
            "Notes",
        ])
    for col in ["Vendor", "ContactPerson", "ContactPhone", "ContactEmail", "OrderDay", "DeliveryDay", "Notes"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)
    if "Order" not in df.columns:
        df["Order"] = ""
    df["Order"] = df["Order"].astype(str)
    return df


def save_invoice_vendors(df: pd.DataFrame) -> None:
    out = df.copy()
    for col in ["Vendor", "ContactPerson", "ContactPhone", "ContactEmail", "Order", "OrderDay", "DeliveryDay", "Notes"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str)
    out = out[[
        "Vendor",
        "ContactPerson",
        "ContactPhone",
        "ContactEmail",
        "Order",
        "OrderDay",
        "DeliveryDay",
        "Notes",
    ]]
    _save_csv(user_data_file("invoice_vendors.csv"), out)

# ============================================================
# Price Book helpers
# ============================================================

def normalize_sku(sku):
    """Normalize SKU/UPC for consistent matching: string, strip, remove .0 from Excel, keep leading zeros."""
    if pd.isna(sku):
        return ""
    s = str(sku).strip()
    # Remove .0 if it appears (Excel artifact)
    if s.endswith(".0"):
        s = s[:-2]
    # Remove spaces and dashes commonly found in UPC/SKU exports
    s = s.replace(" ", "").replace("-", "")
    return s


def parse_money(value):
    """Parse a money field: strip $, commas, whitespace; convert to float. Returns NaN on error."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return np.nan
    # Remove $ and commas
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def read_csv_flexible(upload):
    """
    Read a CSV from a Streamlit upload with delimiter/encoding fallbacks.
    Returns a DataFrame or raises the last exception if all attempts fail.
    """
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

    # If everything landed in one column, try common delimiters.
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
    """
    Map pricebook columns to standard names.
    Returns dict with keys: Sku, Name, RetailPrice, UnitCost
    Raises ValueError if required columns cannot be found.
    """
    # Define column aliases for each standard field
    sku_aliases = {"Sku", "SKU", "UPC", "UPC Code", "UPC/PLU", "UPC PLU", "Barcode", "PLU", "ItemCode", "Item Code"}
    name_aliases = {"Name", "Description", "Item Name"}
    retail_price_aliases = {"RetailPrice", "Retail Price", "Retail", "Price", "Sell Price"}
    unit_cost_aliases = {"UnitCost", "Unit Cost", "Cost", "Avg Cost"}

    # Find which column exists for each field (case-insensitive)
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
        "UnitCost": cost_col
    }


def load_pricebook() -> pd.DataFrame:
    """Load current price book; return empty if not found."""
    df = _load_csv(user_data_file("pricebook_current.csv"))
    if df.empty:
        return pd.DataFrame(columns=["SKU", "Name", "RetailPrice", "UnitCost"])
    # Normalize legacy column name
    if "SKU" not in df.columns and "Sku" in df.columns:
        df = df.rename(columns={"Sku": "SKU"})
    # Coerce types
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str)
    df["RetailPrice"] = pd.to_numeric(df["RetailPrice"], errors="coerce").fillna(0.0)
    df["UnitCost"] = pd.to_numeric(df["UnitCost"], errors="coerce").fillna(0.0)
    return df


def save_pricebook(df: pd.DataFrame) -> None:
    """Save price book with enforced schema."""
    if "SKU" not in df.columns and "Sku" in df.columns:
        df = df.rename(columns={"Sku": "SKU"})
    out = df[["SKU", "Name", "RetailPrice", "UnitCost"]].copy()
    out["SKU"] = out["SKU"].astype(str).str.strip()
    out["Name"] = out["Name"].astype(str)
    out["RetailPrice"] = pd.to_numeric(out["RetailPrice"], errors="coerce").fillna(0.0)
    out["UnitCost"] = pd.to_numeric(out["UnitCost"], errors="coerce").fillna(0.0)
    out = out.drop_duplicates(subset=["SKU"], keep="last")
    _save_csv(user_data_file("pricebook_current.csv"), out)


def load_inside_daily_totals() -> pd.DataFrame:
    """Load inside daily totals history."""
    df = _load_csv(user_data_file("inside_daily_totals_history.csv"))
    if df.empty:
        return pd.DataFrame(columns=[
            "Date", "TotalUnits", "TotalCOGS", "RetailSalesEstimateTotal",
            "EstimatedGrossProfitTotal", "ActualSalesTotal", "ActualGrossProfitTotal",
            "CreditCardFeesTotal", "NetInsideProfitTotal", "CoverageUnitsPct", "MissingSkuCount"
        ])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    for col in ["TotalUnits", "TotalCOGS", "RetailSalesEstimateTotal", "EstimatedGrossProfitTotal",
                "ActualSalesTotal", "ActualGrossProfitTotal", "CreditCardFeesTotal", "NetInsideProfitTotal",
                "CoverageUnitsPct", "MissingSkuCount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def save_inside_daily_totals(df: pd.DataFrame) -> None:
    """Append one row (or replace if date exists) to inside daily totals history."""
    existing = load_inside_daily_totals()
    if not existing.empty:
        existing_dates = pd.to_datetime(existing["Date"], errors="coerce").dt.date
        incoming_dates = pd.to_datetime(df["Date"], errors="coerce").dt.date
        existing = existing[~existing_dates.isin(set(incoming_dates))]
    result = pd.concat([existing, df], ignore_index=True)
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").dt.date
    for col in ["TotalUnits", "TotalCOGS", "RetailSalesEstimateTotal", "EstimatedGrossProfitTotal",
                "ActualSalesTotal", "ActualGrossProfitTotal", "CreditCardFeesTotal", "NetInsideProfitTotal",
                "CoverageUnitsPct", "MissingSkuCount"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    _save_csv(user_data_file("inside_daily_totals_history.csv"), result)

# ============================================================
# Month helpers
# ============================================================

def month_days(month_str: str) -> int:
    y, m = month_str.split("-")
    return calendar.monthrange(int(y), int(m))[1]

# ============================================================
# UI / Navigation
# ============================================================

require_login()
migrate_legacy_data_if_present()

st.sidebar.markdown(f"**Signed in as:** `{st.session_state.get('username')}`")
if st.sidebar.button("Log out"):
    for k in [
        "user_id",
        "username",
        "is_admin",
        "pricebook_df",
        "pricebook_loaded_at",
        "fixed_costs_df",
        "fixed_costs_month",
    ]:
        st.session_state.pop(k, None)
    st.rerun()

if st.session_state.get("is_admin"):
    with st.sidebar.expander("Admin: Create user"):
        with st.form("admin_create_user"):
            new_user = st.text_input("Username", key="admin_new_user")
            new_email = st.text_input("Email", key="admin_new_email")
            new_pw = st.text_input("Password", type="password", key="admin_new_pw")
            new_pw2 = st.text_input("Confirm password", type="password", key="admin_new_pw2")
            submitted = st.form_submit_button("Create user")
        if submitted:
            if new_pw != new_pw2:
                st.error("Passwords do not match.")
            else:
                ok, msg = _create_user(new_user, new_email, new_pw, is_admin=False)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Fuel Calculator", "Tank Deliveries", "Inside COGS Calculator", "Daily Totals History", "Invoices", "Store Profit (Day + Month)"],
    index=0,
)

# ============================================================
# Page: Fuel Calculator
# ============================================================

if page == "Fuel Calculator":
    st.header("Fuel Calculator")
    st.caption("Upload Petro Outlet CSV → enter posted prices & cost/gal → get fuel profit.")

    uploaded = st.file_uploader("Upload Petro Outlet CSV", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV to begin.")
        st.stop()

    raw = pd.read_csv(uploaded)
    with st.expander("Preview RAW CSV"):
        st.dataframe(raw.head(30), use_container_width=True)

    clean = clean_transactions(raw)
    if (clean["TenderType"] == "OTHER").any():
        st.warning("Some transactions have unrecognized tender types and are treated as CASH for pricing. Review tender codes if this looks wrong.")
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

    default_prices = pd.DataFrame(
        [(d, g, np.nan, np.nan) for d in dates for g in grades],
        columns=["Date", "Grade", "CashPrice", "CreditPrice"],
    )
    default_costs = pd.DataFrame(
        [(d, g, np.nan) for d in dates for g in grades],
        columns=["Date", "Grade", "CostPerGallon"],
    )

    # Load last posted prices to use as defaults
    last_prices = load_last_posted_prices()
    for grade, (cash_p, credit_p) in last_prices.items():
        default_prices.loc[default_prices["Grade"] == grade, "CashPrice"] = cash_p
        default_prices.loc[default_prices["Grade"] == grade, "CreditPrice"] = credit_p

    # Load last posted costs to use as defaults
    try:
        last_costs_df = _load_csv(user_data_file("last_posted_costs.csv"))
        if not last_costs_df.empty:
            last_costs_df["Grade"] = pd.to_numeric(last_costs_df["Grade"], errors="coerce")
            last_costs_df["CostPerGallon"] = pd.to_numeric(last_costs_df["CostPerGallon"], errors="coerce")
            for _, row in last_costs_df.dropna(subset=["Grade"]).iterrows():
                grade = int(row["Grade"])
                cost = row.get("CostPerGallon")
                if pd.notna(cost):
                    default_costs.loc[default_costs["Grade"] == grade, "CostPerGallon"] = cost
    except Exception:
        pass

    # Initialize session state for prices and costs if not present
    if "fuel_prices_editor" not in st.session_state:
        st.session_state.fuel_prices_editor = default_prices.copy()
    if "fuel_costs_editor" not in st.session_state:
        st.session_state.fuel_costs_editor = default_costs.copy()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Prices (posted)")
        st.session_state.fuel_prices_editor = st.data_editor(
            st.session_state.fuel_prices_editor,
            num_rows="fixed",
            use_container_width=True,
            key="prices_editor_data",
            column_config={
                "CashPrice": st.column_config.NumberColumn("CashPrice", format="$%.4f"),
                "CreditPrice": st.column_config.NumberColumn("CreditPrice", format="$%.4f"),
            }
        )
    with c2:
        st.markdown("### Costs (your cost/gal)")
        st.session_state.fuel_costs_editor = st.data_editor(
            st.session_state.fuel_costs_editor,
            num_rows="fixed",
            use_container_width=True,
            key="costs_editor_data",
            column_config={
                "CostPerGallon": st.column_config.NumberColumn("CostPerGallon", format="$%.4f"),
            }
        )

    # Reference for easier use
    prices = st.session_state.fuel_prices_editor
    costs = st.session_state.fuel_costs_editor

    # Auto-calculate 89 cost as average of 87 and 93 if 89 is empty but 87 and 93 are filled
    for idx, row in costs.iterrows():
        if pd.notna(row["Grade"]) and int(row["Grade"]) == 89:
            grade_87_cost = costs[(costs["Grade"] == 87) & (costs["Date"] == row["Date"])]["CostPerGallon"]
            grade_93_cost = costs[(costs["Grade"] == 93) & (costs["Date"] == row["Date"])]["CostPerGallon"]
            if not grade_87_cost.empty and not grade_93_cost.empty and pd.notna(grade_87_cost.iloc[0]) and pd.notna(grade_93_cost.iloc[0]):
                costs.at[idx, "CostPerGallon"] = (grade_87_cost.iloc[0] + grade_93_cost.iloc[0]) / 2
                st.session_state.fuel_costs_editor.at[idx, "CostPerGallon"] = costs.at[idx, "CostPerGallon"]

    # Coerce prices and costs to numeric types
    prices = prices.copy()
    prices["Grade"] = pd.to_numeric(prices["Grade"], errors="coerce")
    prices["CashPrice"] = pd.to_numeric(prices["CashPrice"], errors="coerce")
    prices["CreditPrice"] = pd.to_numeric(prices["CreditPrice"], errors="coerce")
    
    costs = costs.copy()
    costs["Grade"] = pd.to_numeric(costs["Grade"], errors="coerce")
    costs["CostPerGallon"] = pd.to_numeric(costs["CostPerGallon"], errors="coerce")

    def has_missing(df, cols):
        return df[cols].isna().any().any()

    def has_missing_required_costs(costs_df):
        if costs_df is None or costs_df.empty:
            return True
        tmp = costs_df.copy()
        tmp["Grade"] = pd.to_numeric(tmp["Grade"], errors="coerce")
        req = tmp[tmp["Grade"].isin([87, 93])]
        return req["CostPerGallon"].isna().any()

    if has_missing(prices, ["CashPrice", "CreditPrice"]) or has_missing_required_costs(costs):
        st.warning("Fill in all posted prices and costs for grades 87 and 93 to compute profit.")
        st.stop()

    profit = build_profit_table(summary, prices, costs, credit_fee_rate)
    if profit.get("_MissingPrice", pd.Series(dtype=bool)).any() or profit.get("_MissingCost", pd.Series(dtype=bool)).any():
        missing_prices = int(profit.get("_MissingPrice", pd.Series(dtype=bool)).sum())
        missing_costs = int(profit.get("_MissingCost", pd.Series(dtype=bool)).sum())
        st.warning(
            f"Some rows are missing posted prices or costs. "
            f"Missing price rows: {missing_prices}; missing cost rows: {missing_costs}. "
            "Those values were treated as $0.00."
        )

    st.subheader("Results")

    # Styled display with green/red
    view = profit.copy()

    # Format numeric columns for display
    view["Date"] = pd.to_datetime(view["Date"]).dt.strftime("%m-%d-%Y")
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
    dview["Date"] = pd.to_datetime(dview["Date"]).dt.strftime("%m-%d-%Y")
    dview["TotalGallons"] = dview["TotalGallons"].map(fmt_number)
    for col in ["ExpectedRevenue", "POSRevenue", "COGS", "GrossProfit", "CreditCardFees", "NetFuelProfit"]:
        dview[col] = dview[col].map(fmt_currency)
    dview["MarginPerGallon"] = daily["MarginPerGallon"].map(fmt_percent)

    st.dataframe(dview.sort_values("Date"), use_container_width=True)

    # Save into history (upsert by Date)
    st.divider()
    if st.button("Save these daily totals to History"):
        # Save prices and costs for next time
        try:
            last_prices_dict = {}
            for _, row in prices.dropna(subset=["Grade"]).iterrows():
                grade = int(row["Grade"])
                if pd.notna(row.get("CashPrice")) and pd.notna(row.get("CreditPrice")):
                    last_prices_dict[grade] = (row["CashPrice"], row["CreditPrice"])
            if last_prices_dict:
                save_last_posted_prices(last_prices_dict)
        except Exception as e:
            st.warning(f"Could not save prices: {e}")

        try:
            last_costs_rows = []
            for _, row in costs.dropna(subset=["Grade"]).iterrows():
                grade = int(row["Grade"])
                if pd.notna(row.get("CostPerGallon")):
                    last_costs_rows.append({"Grade": grade, "CostPerGallon": row["CostPerGallon"]})
            if last_costs_rows:
                last_costs_df = pd.DataFrame(last_costs_rows, columns=["Grade", "CostPerGallon"])
                _save_csv(user_data_file("last_posted_costs.csv"), last_costs_df)
        except Exception as e:
            st.warning(f"Could not save costs: {e}")

        # Save history
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
        st.info("No history yet. Go to Fuel Calculator, upload a CSV, compute totals, then Save to History.")
        st.stop()

    st.subheader("📊 Fuel Daily Totals")

    # ---- Add / Update Day for fuel ----
    with st.expander("➕ Add or Update a Day", expanded=False):
        st.caption("Manually add or update daily fuel totals")
        
        hc1, hc2, hc3, hc4 = st.columns([1.2, 1.2, 1.2, 1.2])
        with hc1:
            h_date = st.date_input("Date", value=date.today(), key="history_date")
        with hc2:
            h_gallons = st.number_input("Total Gallons", min_value=0.0, value=0.0, step=10.0, format="%.1f", key="history_gallons")
        with hc3:
            h_pos_revenue = st.number_input("POS Revenue ($)", min_value=0.0, value=0.0, step=100.0, format="%.2f", key="history_pos_revenue")
        with hc4:
            h_cogs = st.number_input("COGS ($)", min_value=0.0, value=0.0, step=50.0, format="%.2f", key="history_cogs")
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("💾 Save", key="history_add", use_container_width=True):
                new_entry = pd.DataFrame([{
                    "Date": pd.to_datetime(h_date),
                    "Gallons_CASH": 0.0,
                    "Gallons_CREDIT": h_gallons,
                    "POSAmount_CASH": 0.0,
                    "POSAmount_CREDIT": h_pos_revenue,
                    "TotalGallons": h_gallons,
                    "POSRevenue": h_pos_revenue,
                    "COGS": h_cogs,
                    "ExpectedRevenue": h_pos_revenue,
                    "GrossProfit": h_pos_revenue - h_cogs,
                    "CreditCardFees": 0.0,
                    "NetFuelProfit": h_pos_revenue - h_cogs,
                }])
                
                full = history.copy()
                full = full[~(full["Date"].dt.date == h_date)]
                full = pd.concat([full, new_entry], ignore_index=True).sort_values("Date")
                save_history(full)
                
                st.success(f"✓ Saved {h_date}")
                st.rerun()

    st.divider()

    # Parse history dates
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    
    # Get unique months from data
    history_sorted = history.sort_values("Date")
    date_range = history_sorted["Date"].dt.date.unique()
    
    if len(date_range) > 0:
        # Create list of available months (YYYY-MM format)
        available_months = sorted(set(pd.to_datetime(d).strftime("%Y-%m") for d in date_range))
        
        # Default to current month if it's in the data, otherwise use most recent
        current_month = datetime.now().strftime("%Y-%m")
        if current_month in available_months:
            default_month = current_month
        else:
            default_month = available_months[-1]
        
        # Month selector dropdown
        col1, col_space = st.columns([2, 2])
        with col1:
            selected_month = st.selectbox(
                "📅 Filter by month",
                options=available_months,
                index=available_months.index(default_month),
                key="month_filter"
            )
        
        # Filter history by selected month
        history_filtered = history[
            history["Date"].dt.strftime("%Y-%m") == selected_month
        ]
    else:
        history_filtered = history

    history_filtered["MarginPerGallon"] = np.where(
        history_filtered["TotalGallons"] > 0,
        history_filtered["NetFuelProfit"] / history_filtered["TotalGallons"],
        np.nan
    )

    # Display
    view = history_filtered.copy()
    view["Date"] = pd.to_datetime(view["Date"]).dt.strftime("%m-%d-%Y")
    view["TotalGallons"] = view["TotalGallons"].map(fmt_number)
    for col in ["ExpectedRevenue", "POSRevenue", "COGS", "GrossProfit", "CreditCardFees", "NetFuelProfit"]:
        view[col] = view[col].map(fmt_currency)
    view["MarginPerGallon"] = history_filtered["MarginPerGallon"].map(fmt_percent)

    st.markdown("### Daily Fuel Summary")
    st.dataframe(view.sort_values("Date"), use_container_width=True)

    col_download, col_space2 = st.columns([2, 3])
    with col_download:
        st.download_button(
            "📥 Download Month Data",
            data=history_filtered.sort_values("Date").to_csv(index=False).encode("utf-8"),
            file_name=f"daily_totals_history_{selected_month if len(date_range) > 0 else 'all'}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.divider()
    with st.expander("🗑️ Delete a Day", expanded=False):
        if len(date_range) > 0:
            del_fuel_date = st.selectbox(
                "Select day to delete",
                sorted(history_filtered["Date"].dt.date.unique()),
                key="history_delete",
            )
            if st.button("Delete selected day", key="history_delete_btn", use_container_width=True):
                full = history.copy()
                full = full[full["Date"].dt.date != del_fuel_date]
                save_history(full)
                st.success(f"✓ Deleted {del_fuel_date}")
                st.rerun()
        else:
            st.info("No days to delete.")

    st.divider()

    # ---- Inside-store daily history
    st.subheader("🏪 Daily Inside Store Totals")

    store_daily = load_store_daily()
    store_daily["Date"] = pd.to_datetime(store_daily["Date"], errors="coerce")

    # ---- Add / Update Day for inside-store ----
    with st.expander("➕ Add or Update a Day (Inside Store)", expanded=False):
        st.caption("Record daily inside-store sales, COGS, and other variable costs")
        
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
        with c1:
            d = st.date_input("Date", value=date.today(), key="inside_history_date")
        with c2:
            inside_sales = st.number_input("Inside Sales ($)", min_value=0.0, value=0.0, step=100.0, format="%.2f", key="inside_history_sales")
        with c3:
            inside_cogs = st.number_input("Inside COGS ($)", min_value=0.0, value=0.0, step=50.0, format="%.2f", key="inside_history_cogs")
        with c4:
            other_var = st.number_input("Other Var Costs ($)", min_value=0.0, value=0.0, step=25.0, format="%.2f", key="inside_history_other")

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("💾 Save", key="inside_history_add", use_container_width=True):
                new_row = pd.DataFrame([{
                    "Date": pd.to_datetime(d),
                    "InsideSales": inside_sales,
                    "InsideCOGS": inside_cogs,
                    "OtherVariableCosts": other_var,
                    "Notes": "",
                }])

                full = store_daily.copy()
                full = full[~(full["Date"].dt.date == d)]
                full = pd.concat([full, new_row], ignore_index=True).sort_values("Date")
                save_store_daily(full)

                st.success(f"✓ Saved {d}")
                st.rerun()

    # --- Month selector for inside-store saved days (filter view) ---
    if store_daily.empty:
        st.info("No saved inside-store days yet.")
        store_daily_filtered = store_daily
    else:
        st.markdown("### Daily Inside Sales Summary")
        store_daily_sorted = store_daily.sort_values("Date")
        inside_date_range = store_daily_sorted["Date"].dt.date.unique()

        if len(inside_date_range) > 0:
            inside_available_months = sorted(set(pd.to_datetime(d).strftime("%Y-%m") for d in inside_date_range))
            current_month = datetime.now().strftime("%Y-%m")
            if current_month in inside_available_months:
                default_inside_month = current_month
            else:
                default_inside_month = inside_available_months[-1]

            colm1, colm_space = st.columns([2, 2])
            with colm1:
                selected_inside_month = st.selectbox(
                    "📅 Filter by month",
                    options=inside_available_months,
                    index=inside_available_months.index(default_inside_month),
                    key="inside_month_filter",
                )

            store_daily_filtered = store_daily[store_daily["Date"].dt.strftime("%Y-%m") == selected_inside_month]
        else:
            store_daily_filtered = store_daily.copy()

        if store_daily_filtered.empty:
            st.info("No saved inside-store days for the selected month.")
        else:
            sm = store_daily_filtered.copy()
            sm["Date"] = pd.to_datetime(sm["Date"]).dt.strftime("%m-%d-%Y")
            sm["NetInsideSales"] = (
                pd.to_numeric(sm["InsideSales"], errors="coerce").fillna(0.0)
                - pd.to_numeric(sm["InsideCOGS"], errors="coerce").fillna(0.0)
                - pd.to_numeric(sm["OtherVariableCosts"], errors="coerce").fillna(0.0)
            )
            sm["MarginPct"] = np.where(
                pd.to_numeric(sm["InsideSales"], errors="coerce").fillna(0.0) > 0,
                sm["NetInsideSales"] / pd.to_numeric(sm["InsideSales"], errors="coerce").fillna(0.0),
                np.nan,
            )
            sm["InsideSales"] = sm["InsideSales"].map(fmt_currency)
            sm["InsideCOGS"] = sm["InsideCOGS"].map(fmt_currency)
            sm["OtherVariableCosts"] = sm["OtherVariableCosts"].map(fmt_currency)
            sm["NetInsideSales"] = sm["NetInsideSales"].map(fmt_currency)
            sm["MarginPct"] = sm["MarginPct"].map(fmt_percent)
            sm = sm.drop(columns=["Notes"], errors="ignore")
            st.dataframe(sm.sort_values("Date"), use_container_width=True)

            with st.expander("🗑️ Delete a Day", expanded=False):
                del_date = st.selectbox(
                    "Select day to delete",
                    sorted(store_daily_filtered["Date"].dt.date.unique()),
                    key="inside_history_delete",
                )
                if st.button("Delete selected day", key="inside_history_delete_btn", use_container_width=True):
                    full = load_store_daily()
                    full["Date"] = pd.to_datetime(full["Date"], errors="coerce")
                    full = full[full["Date"].dt.date != del_date]
                    save_store_daily(full)
                    st.success(f"✓ Deleted {del_date}")
                    st.rerun()

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
    deliveries = _load_csv(user_data_file("tank_deliveries.csv"))
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
        _save_csv(user_data_file("tank_deliveries.csv"), deliveries2)
        st.success("Delivery saved.")
        st.rerun()

    st.subheader("Delivery history")
    if deliveries.empty:
        st.info("No deliveries logged yet.")
    else:
        dv = deliveries.copy()
        dv["Date"] = pd.to_datetime(dv["Date"], errors="coerce").dt.strftime("%m-%d-%Y")
        dv["GallonsDelivered"] = pd.to_numeric(dv["GallonsDelivered"], errors="coerce").fillna(0.0).map(fmt_number)
        dv["PricePerGallon"] = pd.to_numeric(dv["PricePerGallon"], errors="coerce").fillna(0.0).map(fmt_currency)
        st.dataframe(dv.sort_values("Date", ascending=False), use_container_width=True)

# ============================================================
# Page: Inside COGS (Price Book + Daily Product Report)
# ============================================================

elif page == "Inside COGS Calculator":
    st.header("Inside COGS Calculator")
    st.caption("Upload a price book and daily product report to calculate inside-store COGS and profit.")

    # --- Load price book from session_state or disk ---
    if "pricebook_df" not in st.session_state:
        pb_from_disk = load_pricebook()
        if not pb_from_disk.empty:
            st.session_state["pricebook_df"] = pb_from_disk
            st.session_state["pricebook_loaded_at"] = None  # Loaded from disk, not this session
    
    current_pb = st.session_state.get("pricebook_df", pd.DataFrame())
    pricebook_available = not current_pb.empty

    # --- Price Book Management ---
    st.subheader("1) Price Book Management")
    st.caption("Upload a price book CSV/XLSX with SKU, Name, RetailPrice, and UnitCost columns.")

    if pricebook_available:
        st.success(f"Price book loaded: {len(current_pb)} SKUs.")
        with st.expander("View current price book (preview)"):
            st.dataframe(current_pb.head(20), use_container_width=True)
    else:
        st.info("No price book loaded yet. Upload one below to get started.")

    st.markdown("**Upload a new price book to replace the current one:**")
    pb_upload = st.file_uploader(
        "Upload Price Book CSV/XLSX",
        type=["csv", "xlsx"],
        key="inside_cogs_pricebook_uploader"
    )

    if pb_upload is not None:
        try:
            if pb_upload.name.endswith(".xlsx"):
                pb_raw = pd.read_excel(pb_upload, sheet_name=0)
            else:
                pb_raw = pd.read_csv(pb_upload)

            st.info(f"Price book loaded: {pb_raw.shape[0]} rows, {pb_raw.shape[1]} columns.")
            with st.expander("Preview: Price Book Raw Columns"):
                st.write(list(pb_raw.columns))
                st.dataframe(pb_raw.head(10), use_container_width=True)

            # Try column-based mapping first (supports aliases)
            pb_clean = None
            try:
                col_map = map_pricebook_columns(pb_raw)
                pb_clean = pd.DataFrame({
                    "SKU": pb_raw[col_map["Sku"]].astype(str).apply(normalize_sku),
                    "Name": pb_raw[col_map["Name"]].astype(str),
                    "RetailPrice": pb_raw[col_map["RetailPrice"]].apply(parse_money),
                    "UnitCost": pb_raw[col_map["UnitCost"]].apply(parse_money),
                })
            except ValueError:
                # Fallback: try position-based extraction if columns have standard names or positions
                # Column B (index 1): Sku, Column C (index 2): Name, Column G (index 6): RetailPrice, Column H (index 7): UnitCost
                if pb_raw.shape[1] >= 8:
                    pb_clean = pd.DataFrame({
                        "SKU": pb_raw.iloc[:, 1].astype(str).apply(normalize_sku),
                        "Name": pb_raw.iloc[:, 2].astype(str),
                        "RetailPrice": pb_raw.iloc[:, 6].apply(parse_money),
                        "UnitCost": pb_raw.iloc[:, 7].apply(parse_money),
                    })
                else:
                    st.error(
                        "Price book columns not recognized. Supported column names:\n"
                        "- SKU: Sku, SKU, UPC, PLU, ItemCode\n"
                        "- Name: Name, Description, Item Name\n"
                        "- RetailPrice: RetailPrice, Retail, Price, Sell Price\n"
                        "- UnitCost: UnitCost, Cost, Unit Cost, Avg Cost\n\n"
                        "Or provide 8+ columns with standard Excel layout (B=SKU, C=Name, G=Price, H=Cost)"
                    )
                    pb_clean = None

            if pb_clean is not None:
                pb_clean = pb_clean[pb_clean["SKU"].str.len() > 0]  # Remove empty SKUs
                save_pricebook(pb_clean)
                st.session_state["pricebook_df"] = pb_clean
                st.session_state["pricebook_loaded_at"] = pd.Timestamp.now().isoformat()
                st.success(f"Price book saved: {len(pb_clean)} SKUs")
        except Exception as e:
            st.error(f"Error processing price book: {e}")
            st.caption("Tip: If the file is not a standard CSV/XLSX, export it again from your POS as a plain CSV.")

    st.divider()

    # --- Daily Product Report Processing ---
    st.subheader("2) Daily Product Report")
    st.caption("Upload a ProductReportExport CSV from your POS system.")

    inside_cc_fee_rate = st.number_input(
        "Inside credit card fee rate (e.g., 0.0275 = 2.75%)",
        min_value=0.0,
        max_value=0.10,
        value=0.0275,
        step=0.0005,
        format="%.4f",
    )

    product_upload_key = "inside_cogs_product_report_uploader_v2"
    product_report_upload = st.file_uploader(
        "Upload ProductReportExport CSV",
        type=["csv"],
        key=product_upload_key,
        disabled=not pricebook_available
    )
    if not pricebook_available:
        st.warning("⚠️ Price book required. Upload one above first before proceeding.")
        st.stop()

    # Streamlit sometimes shows a file in the widget while the value is still None.
    # Fallback to session_state and expose a reset button for debugging.
    if product_report_upload is None:
        product_report_upload = st.session_state.get(product_upload_key)
    if st.button("Reset Product Report Upload"):
        st.session_state.pop(product_upload_key, None)
        st.session_state.pop("inside_cogs_product_report_uploader", None)
        st.rerun()

    if product_report_upload is None:
        st.info("Waiting for ProductReportExport CSV upload...")
    else:
        st.write(f"Product report file received: {product_report_upload.name} ({product_report_upload.size} bytes)")

    with st.expander("Diagnostics: Upload State"):
        st.write({
            "pricebook_available": pricebook_available,
            "uploader_key": product_upload_key,
            "value_is_none": product_report_upload is None,
            "session_state_has_key": product_upload_key in st.session_state,
            "session_state_type": str(type(st.session_state.get(product_upload_key))),
            "legacy_key_present": "inside_cogs_product_report_uploader" in st.session_state,
        })

    if product_report_upload is not None:
        try:
            st.info("Processing product report upload...")
            report_raw = read_csv_flexible(product_report_upload)

            if report_raw.empty:
                st.error("Product report parsed with 0 rows. Please re-export as a plain CSV from your POS.")
                st.stop()

            st.info(f"Product report loaded: {report_raw.shape[0]} rows, {report_raw.shape[1]} columns.")
            with st.expander("Preview: Product Report Raw Columns"):
                st.write(list(report_raw.columns))
                st.dataframe(report_raw.head(10), use_container_width=True)

            # Extract columns by header mapping (preferred), with position fallback
            cols = list(report_raw.columns)
            cols_lower = {str(c).strip().lower(): c for c in cols}

            def _find_col(aliases):
                for a in aliases:
                    a_l = a.lower()
                    # exact match
                    if a_l in cols_lower:
                        return cols_lower[a_l]
                    # contains match
                    for k, orig in cols_lower.items():
                        if a_l in k:
                            return orig
                return None

            date_col = _find_col(["date sold", "date", "sold date", "transaction date", "sales date", "sale date"])
            sku_col = _find_col(["sku", "upc", "upc/plu", "upc plu", "upc code", "barcode", "item code", "itemcode", "plu"])
            name_col = _find_col(["name", "item name", "product name", "description"])
            qty_col  = _find_col(["quantity sold in transaction", "quantity", "qty", "units", "qty sold", "sold qty"])
            unit_price_col = _find_col(["actual unit price", "unit price", "price"])
            line_total_col = _find_col(["final products amount for transaction", "line total", "extended price", "ext price", "total"])

            report_clean = None
            if date_col and sku_col:
                report_clean = pd.DataFrame({
                    "DateSold": pd.to_datetime(report_raw[date_col], errors="coerce").dt.date,
                    "SKU": report_raw[sku_col].astype(str).apply(normalize_sku),
                    "Name": report_raw[name_col].astype(str) if name_col else "",
                    "ActualUnitPrice": report_raw[unit_price_col].apply(parse_money) if unit_price_col else np.nan,
                    "Quantity": pd.to_numeric(report_raw[qty_col], errors="coerce").fillna(1.0) if qty_col else 1.0,
                    "LineTotal": report_raw[line_total_col].apply(parse_money) if line_total_col else np.nan,
                })
            else:
                # Fallback: original position-based extraction (older exports)
                # Column B (index 1): Date Sold, Column H (index 7): Sku, Column J (index 9): Name,
                # Column L (index 11): Actual Unit Price, Column M (index 12): Quantity, Column R (index 17): Line Total
                if report_raw.shape[1] >= 18:
                    report_clean = pd.DataFrame({
                        "DateSold": pd.to_datetime(report_raw.iloc[:, 1], errors="coerce").dt.date,
                        "SKU": report_raw.iloc[:, 7].astype(str).apply(normalize_sku),
                        "Name": report_raw.iloc[:, 9].astype(str),
                        "ActualUnitPrice": report_raw.iloc[:, 11].apply(parse_money),
                        "Quantity": pd.to_numeric(report_raw.iloc[:, 12], errors="coerce").fillna(1.0),
                        "LineTotal": report_raw.iloc[:, 17].apply(parse_money),
                    })
                else:
                    st.error("Product report missing required columns. Need at least Date Sold + Sku/UPC.")
                    st.caption("Detected headers: " + ", ".join(map(str, cols)))
                    report_clean = None
            if report_clean is None:
                st.stop()
            if report_clean is not None:
                # Aggregate by (Date, SKU)
                report_clean = report_clean[report_clean["DateSold"].notna() & (report_clean["SKU"].str.len() > 0)]
                agg = report_clean.groupby(["DateSold", "SKU"], as_index=False).agg({
                    "Name": "first",
                    "Quantity": "sum",
                    "ActualUnitPrice": "first",
                    "LineTotal": "sum",
                })
                # Secondary matching key: strip leading zeros to handle UPCs stored as numbers
                agg["SKU_NOLEAD_SALE"] = agg["SKU"].str.lstrip("0")
                agg.loc[agg["SKU_NOLEAD_SALE"] == "", "SKU_NOLEAD_SALE"] = "0"

                # If LineTotal is all zeros or NaN, compute from Quantity * ActualUnitPrice
                agg["ActualSales"] = agg["LineTotal"]
                mask_missing = agg["ActualSales"].isna() | (agg["ActualSales"] == 0)
                agg.loc[mask_missing, "ActualSales"] = (
                    agg.loc[mask_missing, "Quantity"] * agg.loc[mask_missing, "ActualUnitPrice"]
                )
                agg["ActualSales"] = agg["ActualSales"].fillna(0.0)

                # Merge with price book on SKU (from session_state), using explicit suffixes
                current_pb_for_merge = st.session_state.get("pricebook_df", pd.DataFrame()).copy()
                if "SKU" not in current_pb_for_merge.columns and "Sku" in current_pb_for_merge.columns:
                    current_pb_for_merge = current_pb_for_merge.rename(columns={"Sku": "SKU"})
                if current_pb_for_merge.empty and "SKU" not in current_pb_for_merge.columns:
                    current_pb_for_merge = pd.DataFrame(columns=["SKU", "Name", "RetailPrice", "UnitCost"])
                if "SKU" in current_pb_for_merge.columns:
                    current_pb_for_merge["SKU"] = current_pb_for_merge["SKU"].astype(str).apply(normalize_sku)
                    current_pb_for_merge["SKU_NOLEAD_PB"] = current_pb_for_merge["SKU"].str.lstrip("0")
                    current_pb_for_merge.loc[current_pb_for_merge["SKU_NOLEAD_PB"] == "", "SKU_NOLEAD_PB"] = "0"
                else:
                    current_pb_for_merge["SKU_NOLEAD_PB"] = ""

                with st.expander("Diagnostics: Matching Summary"):
                    st.caption("If overlap is 0, the two files likely use different identifiers (e.g., UPC vs internal SKU).")
                    pb_skus = set(current_pb_for_merge.get("SKU", pd.Series([], dtype=str)).astype(str))
                    report_skus = set(agg["SKU"].astype(str))
                    pb_nolead = set(current_pb_for_merge.get("SKU_NOLEAD_PB", pd.Series([], dtype=str)).astype(str))
                    report_nolead = set(agg["SKU_NOLEAD_SALE"].astype(str))
                    overlap_exact = len(pb_skus & report_skus)
                    overlap_nolead = len(pb_nolead & report_nolead)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("PB SKUs", len(pb_skus))
                    c2.metric("Report SKUs", len(report_skus))
                    c3.metric("Exact Overlap", overlap_exact)
                    c4.metric("No-Lead Overlap", overlap_nolead)

                    st.markdown("**Detected Product Report Columns**")
                    st.write({
                        "Date Column": date_col,
                        "SKU/UPC Column": sku_col,
                        "Name Column": name_col,
                        "Quantity Column": qty_col,
                    })

                    st.markdown("**Sample Values (normalized)**")
                    c5, c6 = st.columns(2)
                    with c5:
                        st.markdown("**Price Book**")
                        if "SKU" in current_pb_for_merge.columns:
                            st.dataframe(current_pb_for_merge[["SKU", "SKU_NOLEAD_PB"]].head(10), use_container_width=True)
                        else:
                            st.info("Price book SKU column not detected.")
                    with c6:
                        st.markdown("**Product Report**")
                        st.dataframe(agg[["SKU", "SKU_NOLEAD_SALE"]].head(10), use_container_width=True)
                
                # Merge with suffixes for Name column conflict
                merged = agg.merge(
                    current_pb_for_merge, 
                    on="SKU", 
                    how="left", 
                    suffixes=("_sale", "_pb"),
                    indicator=True
                )

                # If no direct match, try SKU without leading zeros (handles UPC stored as numeric)
                if not current_pb_for_merge.empty:
                    pb_nolead = current_pb_for_merge.drop_duplicates(subset=["SKU_NOLEAD_PB"], keep="last").set_index("SKU_NOLEAD_PB")
                    needs_fallback = merged["_merge"] == "left_only"
                    if needs_fallback.any():
                        fb_index = merged.loc[needs_fallback, "SKU_NOLEAD_SALE"]
                        name_fb = fb_index.map(pb_nolead["Name"])
                        retail_fb = fb_index.map(pb_nolead["RetailPrice"])
                        cost_fb = fb_index.map(pb_nolead["UnitCost"])
                        sku_fb = fb_index.map(pb_nolead["SKU"])

                        merged.loc[needs_fallback, "Name_pb"] = name_fb.values
                        merged.loc[needs_fallback, "RetailPrice"] = retail_fb.values
                        merged.loc[needs_fallback, "UnitCost"] = cost_fb.values
                        matched_by_nolead = sku_fb.notna()
                        merged.loc[needs_fallback, "_merge"] = np.where(matched_by_nolead, "both", "left_only")
                        merged.loc[needs_fallback & matched_by_nolead, "MatchedBy"] = "SKU_NOLEAD"


                # Create single Name column: prefer product report name
                merged["Name"] = merged["Name_sale"].fillna(merged["Name_pb"])
                merged = merged.drop(columns=["Name_sale", "Name_pb"], errors="ignore")
                if "MatchedBy" not in merged.columns:
                    merged["MatchedBy"] = ""
                merged["MatchedBy"] = merged["MatchedBy"].fillna("")
                merged.loc[merged["MatchedBy"] == "", "MatchedBy"] = np.where(
                    merged["_merge"] == "both", "SKU", "Unmatched"
                )

                # Compute COGS and profits
                DEFAULT_MISSING_MARGIN = 0.25
                merged["UnitCostOriginal"] = merged["UnitCost"]
                merged["UnitCostMissing"] = merged["UnitCostOriginal"].isna() | (merged["UnitCostOriginal"] == 0)
                merged["UnitCost"] = merged["UnitCostOriginal"]
                has_qty = merged["Quantity"] > 0
                missing_with_qty = merged["UnitCostMissing"] & has_qty
                merged.loc[missing_with_qty, "UnitCost"] = (
                    (merged.loc[missing_with_qty, "ActualSales"] / merged.loc[missing_with_qty, "Quantity"])
                    * (1 - DEFAULT_MISSING_MARGIN)
                )
                merged.loc[merged["UnitCostMissing"] & ~has_qty, "UnitCost"] = 0.0
                merged["UnitCost"] = merged["UnitCost"].fillna(0.0)
                merged["RetailPrice"] = merged["RetailPrice"].fillna(0.0)
                merged["COGS"] = merged["Quantity"] * merged["UnitCost"]
                merged["RetailSalesEstimate"] = merged["Quantity"] * merged["RetailPrice"]
                merged["GrossProfit"] = merged["ActualSales"] - merged["COGS"]
                merged["CreditCardFees"] = merged["ActualSales"] * float(inside_cc_fee_rate or 0.0)
                merged["NetInsideProfit"] = merged["GrossProfit"] - merged["CreditCardFees"]

                # Split into matched and unmatched (based on merge indicator)
                matched_items = merged[merged["_merge"] == "both"].copy()
                unmatched_skus = merged[merged["_merge"] == "left_only"].copy()

                # Summary metrics
                total_skus_sold = agg["SKU"].nunique()
                total_units = agg["Quantity"].sum()
                total_cogs = merged["COGS"].sum()
                total_actual_sales = agg["ActualSales"].sum()
                total_gross_profit = merged["GrossProfit"].sum()
                total_cc_fees = merged["CreditCardFees"].sum()
                total_net_inside = merged["NetInsideProfit"].sum()

                # Show results
                with st.expander("Preview: Raw Uploaded Data"):
                    st.dataframe(report_clean.head(30), use_container_width=True)
                with st.expander("Diagnostics: Matching Keys"):
                    st.caption("Sample of normalized keys from each file to confirm alignment.")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Price Book (normalized)**")
                        if "SKU" in current_pb_for_merge.columns and "SKU_NOLEAD_PB" in current_pb_for_merge.columns:
                            st.dataframe(current_pb_for_merge[["SKU", "SKU_NOLEAD_PB"]].head(20), use_container_width=True)
                        else:
                            st.info("Price book keys not available yet.")
                    with c2:
                        st.markdown("**Product Report (normalized)**")
                        st.dataframe(agg[["SKU", "SKU_NOLEAD_SALE"]].head(20), use_container_width=True)

                st.markdown("### Summary")
                s1, s2, s3, s4, s5, s6, s7, s8, s9 = st.columns(9)
                s1.metric("Total SKUs Sold", fmt_number(total_skus_sold))
                s2.metric("Total Units", fmt_number(total_units))
                s3.metric("Total COGS", fmt_currency(total_cogs))
                s4.metric("Total Actual Sales", fmt_currency(total_actual_sales))
                s5.metric("Total Gross Profit", fmt_currency(total_gross_profit))
                s6.metric("Credit Card Fees", fmt_currency(total_cc_fees))
                s7.metric("Net Inside Profit", fmt_currency(total_net_inside))
                s8.metric("Unmatched SKUs", len(unmatched_skus))
                # Missing unit cost count calculated later once grouped
                s9.metric("Missing Unit Cost", "—")

                st.markdown("### Matched Items (in Price Book)")
                view_matched = matched_items[[
                    "DateSold", "SKU", "Name", "Quantity", "RetailPrice", "UnitCost", "COGS",
                    "RetailSalesEstimate", "ActualSales", "GrossProfit", "CreditCardFees", "NetInsideProfit"
                ]].copy()
                view_matched = view_matched.rename(columns={
                    "DateSold": "Date Sold",
                    "SKU": "SKU",
                    "Name": "Name",
                    "Quantity": "Qty",
                    "RetailPrice": "Retail Price",
                    "UnitCost": "Unit Cost",
                    "COGS": "COGS",
                    "RetailSalesEstimate": "Est. Retail Sales",
                    "ActualSales": "Actual Sales",
                    "GrossProfit": "Gross Profit",
                    "CreditCardFees": "CC Fees",
                    "NetInsideProfit": "Net Inside Profit"
                })
                view_matched["Date Sold"] = pd.to_datetime(view_matched["Date Sold"]).dt.strftime("%m-%d-%Y")
                view_matched["Qty"] = view_matched["Qty"].map(fmt_number)
                for col in ["Unit Cost", "COGS", "Retail Price", "Est. Retail Sales", "Actual Sales", "Gross Profit", "CC Fees", "Net Inside Profit"]:
                    view_matched[col] = view_matched[col].map(fmt_currency)
                st.dataframe(view_matched.sort_values(["Date Sold", "SKU"]), use_container_width=True)

                # Coverage metrics
                total_units = agg["Quantity"].sum()
                matched_units = matched_items["Quantity"].sum() if not matched_items.empty else 0
                total_sales = agg["ActualSales"].sum()
                matched_sales = matched_items["ActualSales"].sum() if not matched_items.empty else 0
                coverage_units_pct = (matched_units / total_units * 100) if total_units > 0 else 0
                coverage_sales_pct = (matched_sales / total_sales * 100) if total_sales > 0 else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Units", fmt_number(total_units))
                c2.metric("Matched Units", fmt_number(matched_units))
                c3.metric("Coverage %", f"{coverage_units_pct:.1f}%")
                c4.metric("Unmatched SKUs", len(unmatched_skus))

                if coverage_units_pct < 95:
                    st.warning(f"⚠️ Low coverage: {coverage_units_pct:.1f}% of units matched. Check unmatched SKUs below.")

                # Unmatched items (SKUs sold but not found in price book)
                if not unmatched_skus.empty:
                    st.markdown("### Unmatched SKUs (sold but not in price book)")
                    
                    # Group by SKU to show totals
                    unmatched_summary = unmatched_skus.groupby("SKU", as_index=False).agg({
                        "Name": "first",
                        "Quantity": "sum",
                        "ActualSales": "sum",
                    })
                    
                    view_unmatched = unmatched_summary[[
                        "SKU", "Name", "Quantity", "ActualSales"
                    ]].copy()
                    view_unmatched = view_unmatched.rename(columns={
                        "SKU": "SKU",
                        "Name": "Name",
                        "Quantity": "Qty Sold",
                        "ActualSales": "Total Sales"
                    })
                    view_unmatched["Qty Sold"] = view_unmatched["Qty Sold"].map(fmt_number)
                    view_unmatched["Total Sales"] = view_unmatched["Total Sales"].map(fmt_currency)
                    st.dataframe(view_unmatched.sort_values("Qty Sold", ascending=False), use_container_width=True)

                    # Download unmatched SKUs
                    unmatched_csv = unmatched_summary[["SKU", "Name", "Quantity", "ActualSales"]].to_csv(index=False)
                    st.download_button(
                        "📥 Download Unmatched SKUs CSV",
                        data=unmatched_csv.encode("utf-8"),
                        file_name="unmatched_skus.csv",
                        mime="text/csv"
                    )

                # Download matched items
                matched_csv = matched_items[[
                    "DateSold", "SKU", "Name", "Quantity", "RetailPrice", "UnitCost", "COGS",
                    "RetailSalesEstimate", "ActualSales", "GrossProfit"
                ]].to_csv(index=False)
                st.download_button(
                    "📥 Download Matched Items CSV",
                    data=matched_csv.encode("utf-8"),
                    file_name="matched_items.csv",
                    mime="text/csv"
                )

                st.divider()

                # --- Missing Unit Cost Section ---
                st.markdown("### Missing Unit Cost (needs price book update)")
                st.caption("Matched items with missing or zero unit cost (affects COGS calculations).")

                # Filter items with missing/zero unit cost
                missing_cost_items = matched_items[matched_items["UnitCostMissing"]].copy()

                if missing_cost_items.empty:
                    st.success("✅ All matched items have unit cost defined!")
                    s9.metric("Missing Unit Cost", 0)
                else:
                    # Group by SKU to show unique items with totals
                    missing_cost_grouped = missing_cost_items.groupby("SKU", as_index=False).agg({
                        "Name": "first",
                        "Quantity": "sum",
                        "RetailPrice": "first",
                        "UnitCostOriginal": "first",
                    }).sort_values("Quantity", ascending=False)

                    # Calculate metrics
                    missing_skus = len(missing_cost_grouped)
                    missing_rows = len(missing_cost_items)
                    total_qty_missing = missing_cost_items["Quantity"].sum()

                    # Display metrics
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Items w/ Missing Cost", missing_rows, help="Total rows/occurrences")
                    mc2.metric("Missing Cost SKUs", missing_skus, help="Unique SKUs")
                    mc3.metric("Qty Affected", fmt_number(total_qty_missing), help="Total units sold")
                    s9.metric("Missing Unit Cost", missing_skus)

                    # Display table
                    st.markdown("**Summary by SKU (sorted by Qty):**")
                    view_missing = missing_cost_grouped[[
                        "SKU", "Name", "Quantity", "RetailPrice", "UnitCostOriginal"
                    ]].copy()
                    view_missing = view_missing.rename(columns={
                        "SKU": "SKU",
                        "Name": "Name",
                        "Quantity": "Qty",
                        "RetailPrice": "Retail Price",
                        "UnitCostOriginal": "Unit Cost (Original)"
                    })
                    view_missing["Qty"] = view_missing["Qty"].map(fmt_number)
                    view_missing["Retail Price"] = view_missing["Retail Price"].map(fmt_currency)
                    view_missing["Unit Cost (Original)"] = view_missing["Unit Cost (Original)"].map(fmt_currency)
                    st.dataframe(view_missing, use_container_width=True)

                    # Download button for missing cost items
                    missing_cost_csv = missing_cost_grouped[[
                        "SKU", "Name", "Quantity", "RetailPrice", "UnitCostOriginal"
                    ]].to_csv(index=False)
                    st.download_button(
                        "📥 Download Missing Unit Cost Items CSV",
                        data=missing_cost_csv.encode("utf-8"),
                        file_name="missing_unit_cost.csv",
                        mime="text/csv"
                    )

                st.divider()

                # --- Save Daily Inside Totals & Store Profit ---
                st.subheader("Save Daily Inside Totals")

                dates_in_report = sorted(agg["DateSold"].unique())
                selected_date = st.selectbox("Select date to save", dates_in_report)

                st.caption("Saves inside totals and writes Inside Sales/COGS to Store Profit (Day + Month).")
                if st.button("Save Inside Totals & Store Profit", use_container_width=True):
                    try:
                        # --- Save Daily Inside Totals ---
                        daily_matched = matched_items[matched_items["DateSold"] == selected_date] if not matched_items.empty else pd.DataFrame()
                        daily_all = merged[merged["DateSold"] == selected_date]
                        total_units_daily = daily_all["Quantity"].sum()
                        matched_units_daily = daily_matched["Quantity"].sum() if not daily_matched.empty else 0
                        coverage_units_pct_daily = (matched_units_daily / total_units_daily * 100) if total_units_daily > 0 else 0
                        total_cogs_daily = daily_all["COGS"].sum()
                        retail_sales_est = daily_all["RetailSalesEstimate"].sum()
                        est_gp = daily_all["RetailSalesEstimate"].sum() - total_cogs_daily
                        actual_sales_daily = daily_all["ActualSales"].sum()
                        actual_gp = daily_all["GrossProfit"].sum()
                        cc_fees_daily = daily_all["CreditCardFees"].sum()
                        net_inside_daily = daily_all["NetInsideProfit"].sum()
                        missing_count = len(unmatched_skus[unmatched_skus["DateSold"] == selected_date]) if not unmatched_skus.empty else 0

                        row_to_save = pd.DataFrame([{
                            "Date": selected_date,
                            "TotalUnits": total_units_daily,
                            "TotalCOGS": total_cogs_daily,
                            "RetailSalesEstimateTotal": retail_sales_est,
                            "EstimatedGrossProfitTotal": est_gp,
                            "ActualSalesTotal": actual_sales_daily,
                            "ActualGrossProfitTotal": actual_gp,
                            "CreditCardFeesTotal": cc_fees_daily,
                            "NetInsideProfitTotal": net_inside_daily,
                            "CoverageUnitsPct": coverage_units_pct_daily,
                            "MissingSkuCount": missing_count,
                        }])

                        save_inside_daily_totals(row_to_save)
                        st.success(f"Saved inside totals for {selected_date}")

                        # --- Save to Store Profit ---
                        inside_sales_daily = float(daily_all["ActualSales"].sum())
                        inside_cogs_daily = float(daily_all["COGS"].sum())
                        inside_cc_fees_daily = float(daily_all["CreditCardFees"].sum())
                        notes = f"From ProductReportExport (coverage {coverage_units_pct_daily:.1f}%). CC fees at {inside_cc_fee_rate:.4f}."

                        store_daily = load_store_daily()
                        store_daily["Date"] = pd.to_datetime(store_daily["Date"], errors="coerce")
                        new_row = pd.DataFrame([{
                            "Date": pd.to_datetime(selected_date),
                            "InsideSales": inside_sales_daily,
                            "InsideCOGS": inside_cogs_daily,
                            "OtherVariableCosts": inside_cc_fees_daily,
                            "Notes": notes,
                        }])
                        # Upsert by date
                        store_daily = store_daily[store_daily["Date"].dt.date != selected_date]
                        store_daily = pd.concat([store_daily, new_row], ignore_index=True).sort_values("Date")
                        save_store_daily(store_daily)
                        st.success(f"Saved Store Profit day for {selected_date}")
                    except Exception as e:
                        st.error(f"Error saving: {e}")
        except Exception as e:
            st.error(f"Error processing product report: {e}")
            st.caption("If you can, share the column list from the POS export or a tiny redacted sample.")

# ============================================================
# Page: Invoices
# ============================================================

elif page == "Invoices":
    st.header("Invoices")
    st.caption("Log invoices for new products. These totals are applied to the daily profit summary.")

    st.subheader("Vendor directory")
    st.caption("Keep vendor contact info and ordering schedules here.")

    vendors = load_invoice_vendors()
    vendors_edit = st.data_editor(
        vendors,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Vendor": st.column_config.TextColumn("Vendor"),
            "ContactPerson": st.column_config.TextColumn("Contact Person"),
            "ContactPhone": st.column_config.TextColumn("Contact Phone"),
            "ContactEmail": st.column_config.TextColumn("Contact Email"),
            "Order": st.column_config.SelectboxColumn(
                "Order",
                options=["Online", "Call", "Rep In Store"],
                help="Preferred order method.",
            ),
            "OrderDay": st.column_config.TextColumn("Order Day"),
            "DeliveryDay": st.column_config.TextColumn("Delivery Day"),
            "Notes": st.column_config.TextColumn("Notes"),
        },
    )
    if st.button("Save vendors"):
        save_invoice_vendors(vendors_edit)
        st.success("Vendors saved.")
        st.rerun()

    st.divider()

    invoices = load_invoices()
    invoices["Date"] = pd.to_datetime(invoices["Date"], errors="coerce")

    c1, c2, c3 = st.columns([1.2, 1.6, 1])
    with c1:
        inv_date = st.date_input("Invoice Date", value=date.today(), key="invoice_date")
    with c2:
        inv_vendor = st.text_input("Vendor", key="invoice_vendor")
    with c3:
        inv_amount = st.number_input("Amount ($)", min_value=0.0, value=0.0, step=10.0, format="%.2f", key="invoice_amount")

    inv_notes = st.text_input("Notes (optional)", key="invoice_notes")

    if st.button("Add invoice"):
        new_row = pd.DataFrame([{
            "Date": pd.to_datetime(inv_date),
            "Vendor": inv_vendor.strip(),
            "Amount": float(inv_amount),
            "Notes": inv_notes.strip(),
        }])
        full = pd.concat([invoices, new_row], ignore_index=True)
        save_invoices(full)
        st.success(f"Saved invoice for {inv_date}.")
        st.rerun()

    st.subheader("Invoice history")
    if invoices.empty:
        st.info("No invoices logged yet.")
    else:
        view = invoices.copy()
        view["Date"] = pd.to_datetime(view["Date"], errors="coerce").dt.date
        view["Amount"] = pd.to_numeric(view["Amount"], errors="coerce").fillna(0.0).map(fmt_currency)
        st.dataframe(view.drop(columns=["Notes"], errors="ignore").sort_values("Date", ascending=False), use_container_width=True)

        del_date = st.selectbox("Delete a day of invoices", sorted(view["Date"].unique()), key="invoice_delete_date")
        if st.button("Delete all invoices for selected day"):
            full = load_invoices()
            full["Date"] = pd.to_datetime(full["Date"], errors="coerce")
            full = full[full["Date"].dt.date != del_date]
            save_invoices(full)
            st.success(f"Deleted invoices for {del_date}.")
            st.rerun()

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
            st.session_state["fixed_costs_grid_reload"] = True
            st.session_state["fixed_costs_grid_version"] = (
                int(st.session_state.get("fixed_costs_grid_version", 0)) + 1
            )
            st.rerun()

    selected_rows = []

    with c_help:
        st.caption("Tip: Resize columns once; widths will persist while you add rows. Press Enter to commit a cell edit.")

    # --- Fixed costs editor ---
    selected_month = str(month)
    month_str = selected_month

    fixed_df = st.session_state["fixed_costs_df"].copy()
    # Keep month aligned to the current selected month
    fixed_df["Month"] = selected_month

    if AgGrid is None:
        st.warning("AgGrid not installed. Using basic editor (column widths won’t persist).")
        fixed_edit = st.data_editor(
            fixed_df,
            num_rows="dynamic",
            use_container_width=True,
        )
        st.session_state["fixed_costs_df"] = fixed_edit
    else:
        # --- AgGrid editor (no auto-fit; keep user column widths) ---
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
            key=f"fixed_costs_grid_{month_str}_{st.session_state.get('fixed_costs_grid_version', 0)}",
            reload_data=bool(st.session_state.get("fixed_costs_grid_reload", False)),
        )
        st.session_state["fixed_costs_grid_reload"] = False

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

    y, m = month.split("-")
    y, m = int(y), int(m)
    start = pd.Timestamp(y, m, 1)
    end = pd.Timestamp(y, m, month_days(month))

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
        store_month2["NetInsideProfit"] = (
            pd.to_numeric(store_month2["InsideSales"], errors="coerce").fillna(0.0)
            - pd.to_numeric(store_month2["InsideCOGS"], errors="coerce").fillna(0.0)
            - pd.to_numeric(store_month2["OtherVariableCosts"], errors="coerce").fillna(0.0)
        )
    else:
        store_month2 = pd.DataFrame(columns=["Date", "NetInsideProfit"])

    daily = daily.merge(store_month2[["Date", "NetInsideProfit"]], on="Date", how="left")

    daily["NetFuelProfit"] = pd.to_numeric(daily["NetFuelProfit"], errors="coerce").fillna(0.0)
    daily["NetInsideProfit"] = pd.to_numeric(daily["NetInsideProfit"], errors="coerce").fillna(0.0)

    # Invoices (variable cost)
    invoices_month = load_invoices()
    invoices_month["Date"] = pd.to_datetime(invoices_month["Date"], errors="coerce").dt.date
    invoices_month = invoices_month[(pd.to_datetime(invoices_month["Date"]) >= start) & (pd.to_datetime(invoices_month["Date"]) <= end)].copy()
    if not invoices_month.empty:
        invoices_daily = invoices_month.groupby("Date", as_index=False).agg(DailyInvoices=("Amount", "sum"))
    else:
        invoices_daily = pd.DataFrame(columns=["Date", "DailyInvoices"])
    daily = daily.merge(invoices_daily, on="Date", how="left")
    daily["DailyInvoices"] = pd.to_numeric(daily.get("DailyInvoices", 0.0), errors="coerce").fillna(0.0)

    # Allocate fixed costs evenly across days in month
    days_in_month = len(days)
    daily["FixedCostAllocated"] = fixed_total / days_in_month if days_in_month else 0.0

    daily["TotalProfit"] = daily["NetFuelProfit"] + daily["NetInsideProfit"]
    daily["NetDailyProfit"] = (
        daily["NetFuelProfit"]
        + daily["NetInsideProfit"]
        - daily["DailyInvoices"]
        - daily["FixedCostAllocated"]
    )

    # Display
    show = daily.copy()
    show["Date"] = pd.to_datetime(show["Date"]).dt.strftime("%m-%d-%Y")
    show["NetFuelProfit"] = show["NetFuelProfit"].map(fmt_currency)
    show["NetInsideProfit"] = show["NetInsideProfit"].map(fmt_currency)
    show["TotalProfit"] = show["TotalProfit"].map(fmt_currency)
    show["DailyInvoices"] = show["DailyInvoices"].map(fmt_currency)
    show["FixedCostAllocated"] = show["FixedCostAllocated"].map(fmt_currency)
    show["NetDailyProfit"] = show["NetDailyProfit"].map(fmt_currency)

    st.dataframe(show, use_container_width=True)

    # Month totals
    month_fuel = float(daily["NetFuelProfit"].replace("", 0).apply(lambda x: float(str(x).replace("$","").replace(",","")) if isinstance(x,str) else 0).sum()) if False else float(daily["NetFuelProfit"].astype(float).sum())
    month_inside = float(daily["NetInsideProfit"].astype(float).sum())
    month_invoices = float(daily["DailyInvoices"].astype(float).sum())
    month_station = float(daily["NetDailyProfit"].astype(float).sum())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Fuel profit (month)", fmt_currency(month_fuel))
    m2.metric("Inside profit (month)", fmt_currency(month_inside))
    m3.metric("Invoices (month)", fmt_currency(month_invoices))
    m4.metric("Fixed costs (month)", fmt_currency(fixed_total))
    m5.metric("Total station profit (month)", fmt_currency(month_station))
