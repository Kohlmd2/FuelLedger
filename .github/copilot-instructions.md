# Fuel Profit Tracker: AI Coding Agent Instructions

## Architecture Overview

This is a **single-file Streamlit application** (`app.py`, ~3,194 lines) that tracks fuel, inside-store, invoice, and inventory profitability workflows for gas stations. No external services—all data is persisted to `.fuel_profit_data/` (CSV files).

### Data Flow

1. **Input CSVs**: User uploads Petro Outlet transaction CSV (contains Date, Grade, Gallons, Amount, Tender)
2. **Calculator Page**: Cleans transactions → summarizes by date/grade → accepts posted prices & costs → computes fuel profit (gross margin, credit card fees, variance to POS)
3. **History Storage**: Saves daily totals to `.fuel_profit_data/daily_totals_history.csv` (upsert by date)
4. **Tank Tracking**: Logs deliveries + baseline tank levels to compute fuel inventory (87/93 only; 89 is blended 50/50)
5. **Store Profit**: Combines fuel + inside-store sales/COGS + fixed costs → monthly P&L

### Key Files

- `app.py`: All UI (Streamlit) + business logic (pandas transforms)
- `.fuel_profit_data/`: Runtime CSV store (includes history, fixed costs, store daily, tank baseline, tank deliveries, invoices, invoice vendors, inventory, inventory deliveries, and auth artifacts)
- `data/`: Example CSVs for development
- `AGENTS.md`: Existing project guidelines (coding style, build commands, testing approach)

## Critical Business Logic & Data Transforms

### Transaction Cleaning (`clean_transactions`)
- Validates required columns: `Date Sold`, `Grade Name`, `Gallons Sold`, `Final Amount`, `Primary Tender Code`
- Maps tender codes to `TenderType` (CASH, DEBIT, CREDIT, OTHER)
- Assigns `PriceTier` (CREDIT vs CASH) for pricing
- **Assumption**: Only grades 87, 89, 93 are valid (others filtered out)

### Daily Summarization (`summarize_by_day`)
- Pivots by Date/Grade → splits gallons & revenue by PriceTier (CASH/CREDIT)
- Produces: `Gallons_CASH`, `Gallons_CREDIT`, `POSAmount_CASH`, `POSAmount_CREDIT`, `TotalGallons`, `POSRevenue`

### Profit Calculation (`build_profit_table`)
- **As-of merge pattern**: For each sale date & grade, looks up most recent posted price/cost on or *before* that date
- Formulas:
  - `ExpectedRevenue = (Gallons_CASH × CashPrice) + (Gallons_CREDIT × CreditPrice)`
  - `COGS = TotalGallons × CostPerGallon`
  - `GrossProfit = POSRevenue - COGS`
  - `CreditCardFees = (Gallons_CREDIT × CreditPrice) × credit_fee_rate`
  - `NetFuelProfit = GrossProfit - CreditCardFees`
  - `RevenueDiff_POS_minus_Expected = POSRevenue - ExpectedRevenue` (variance)
- **Grade blending**: 89 is not stored separately; use 87/93 only for tank math

### Data Date Handling
- Dates normalized to pandas midnight Timestamps in `normalize_date_column()` for consistent joins
- CSV stores dates as strings (YYYY-MM-DD); always coerce on load/save

## Page Organization

1. **Fuel Calculator**: Upload → clean → summarize → enter prices/costs → compute profit → save to history
2. **Tank Deliveries**: Set baseline levels + log deliveries (87/93 only)
3. **Inside COGS Calculator**: Upload product report and compute COGS/profit from price book
4. **Daily Totals History**: View all saved days, download CSV, reset history
5. **Invoices**: Manage vendor directory and invoice history
6. **Inventory**: Current levels, add/update stock, delivery logging, and price book management
7. **Store Profit (Day + Month)**: Fixed costs editor + combined monthly P&L

## Development Patterns

### CSV I/O (`_load_csv`, `_save_csv`)
- All data persisted as CSVs to `.fuel_profit_data/`
- No schema validation on load; defensive coercion (`.astype()`, `.fillna()`) on all numeric columns
- Save functions enforce schema (drop unwanted columns, coerce types, deduplicate)

### Streamlit UI Conventions
- Use `st.data_editor()` for inline edits (lightweight)
- Use `AgGrid` (optional) for persistent column widths; falls back to plain dataframe if not installed
- Format display columns separately (e.g., `fmt_currency`, `fmt_percent`, `_color_profit`) to preserve numeric underlying data
- Use `st.session_state` to persist form state across reruns (e.g., `fixed_costs_df`)

### Error Handling
- Validate required columns early with `require_columns()`
- Use `.errors="coerce"` to handle mixed/invalid data types
- Show `.warning()` if user input is incomplete (e.g., missing prices)
- Empty DataFrames signal no data → show `.info()` and `st.stop()`

## Common Tasks & Patterns

### Adding a New Input Field
1. Add persistent CSV if needed (e.g., new file in `.fuel_profit_data/`)
2. Write `load_*()` and `save_*()` helper functions
3. Use `st.data_editor()` or form inputs + `st.button("Save")` to capture
4. Coerce types defensively on save

### Modifying Profit Formula
- Edit calculations in `build_profit_table()` or the aggregation at the bottom of each page
- Always maintain both gross and variance (`POSRevenue` vs `ExpectedRevenue`)
- Document assumptions (e.g., 89 blending, credit fee rate)

### Debugging CSV Issues
- Check `.fuel_profit_data/` files directly (Excel, `cat` in terminal)
- Use `st.write(df.info())` to inspect types
- Verify date normalization with `pd.to_datetime(df["Date"]).dt.date`

## Deployment & Testing

- **Run**: `streamlit run app.py` (requires `.venv` with `streamlit`, `pandas`, `numpy`)
- **Optional**: `pip install streamlit-aggrid` for better grid UI
- **Test workflow**: Upload example CSV from `data/` → exercise all 7 pages → verify calculations in `.fuel_profit_data/` CSVs
- **No automated tests**: Validation is manual; verify key flows (daily totals, tank math, profit summaries)

## Conventions & Gotchas

- **No formatters/linters configured**: Keep diffs clean (4-space indentation, `snake_case`)
- **Single-file architecture**: All ~3,194 lines in `app.py`; keep business logic & UI close until phased module extraction
- **No Git history**: Commit messages should be small & descriptive
- **Grade constants**: `GRADE_MAP` (87, 89, 93) & `CREDIT_TENDERS`, `CASH_TENDERS` are hardcoded; change in one place
- **Month format**: Use `"YYYY-MM"` string consistently (see `month_days()` parser)
- **Column naming**: Maintain snake_case in CSVs; transform on load/display (e.g., `"Date Sold"` → `"Date"`)
