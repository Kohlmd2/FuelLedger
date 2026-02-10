# Inside COGS Feature Implementation

## Summary

Added a new **"Inside COGS"** page to FuelLedger that calculates daily inside-store COGS (Cost of Goods Sold) and profit from a stored price book and daily product report CSVs.

## Changes Made

### 1. Core Data Files (app.py)
- Added `PRICEBOOK_FILE = DATA_DIR / "pricebook_current.csv"`
- Added `INSIDE_DAILY_TOTALS_FILE = DATA_DIR / "inside_daily_totals_history.csv"`

### 2. Helper Functions (app.py, lines 386–464)

#### `normalize_sku(sku)`
- Converts SKU to string, strips whitespace
- Removes `.0` artifact from Excel imports
- Preserves leading zeros for accurate matching

#### `load_pricebook() -> pd.DataFrame`
- Loads `pricebook_current.csv` with columns: Sku, Name, RetailPrice, UnitCost
- Returns empty DataFrame if not found
- Defensive type coercion on all numeric columns

#### `save_pricebook(df: pd.DataFrame)`
- Enforces schema: Sku, Name, RetailPrice, UnitCost
- Deduplicates by SKU (keeps last)
- Persists to `.fuel_profit_data/pricebook_current.csv`

#### `load_inside_daily_totals() -> pd.DataFrame`
- Loads cumulative inside daily totals history
- Returns empty if not found
- Coerces Date to date type and numeric columns

#### `save_inside_daily_totals(df: pd.DataFrame)`
- Upsert pattern: replaces existing row if date matches, otherwise appends
- Enforces schema with 9 columns
- Persists to `.fuel_profit_data/inside_daily_totals_history.csv`

### 3. Navigation
- Updated sidebar radio to include "Inside COGS" (inserted between "Tank Deliveries" and "Store Profit (Day + Month)")

### 4. New Page: Inside COGS (app.py, lines 747–922)

#### UI Sections:

**Section 1: Price Book Management**
- Upload new price book (CSV/XLSX)
- Extract columns by position:
  - Column B (index 1): SKU
  - Column C (index 2): Name
  - Column G (index 6): RetailPrice
  - Column H (index 7): UnitCost
- Fallback to column name matching if positional extraction fails
- Display success message with SKU count
- Preview table in expander

**Section 2: Daily Product Report**
- Upload ProductReportExport CSV
- Extract columns by position:
  - Column B (index 1): DateSold
  - Column H (index 7): SKU
  - Column J (index 9): Name
  - Column L (index 11): ActualUnitPrice
  - Column M (index 12): Quantity (defaults to 1.0 if missing)
  - Column R (index 17): LineTotal (computed from Qty × Price if missing)
- Aggregate by (Date, SKU)
- Merge with price book on SKU (left join to capture unmatched)

#### Calculations:
- `COGS = Quantity × UnitCost`
- `RetailSalesEstimate = Quantity × RetailPrice`
- `EstimatedGrossProfit = RetailSalesEstimate - COGS`
- `ActualGrossProfit = ActualSales - COGS`

#### Display Tables:
1. **Matched Items** (with formatting):
   - Date, SKU, Name, Quantity, UnitCost, COGS, RetailPrice, RetailSalesEstimate, ActualSales, ActualGrossProfit

2. **Coverage Metrics** (4-column display):
   - Total Units
   - Matched Units
   - Coverage % (matched / total)
   - Missing SKU Count

3. **Missing SKUs** (if any):
   - Date, SKU, Name, Quantity, ActualSales
   - Coverage warning if < 95%

#### Download Buttons:
- Missing SKUs CSV (unique SKUs not in price book)
- Matched Items CSV (all matched transactions with calculations)

#### Section 3: Save Daily Inside Totals
- Selectbox to pick date from report
- Button to save one row with:
  - Date
  - TotalUnits (sum of Quantity for date)
  - TotalCOGS (sum of COGS for date)
  - RetailSalesEstimateTotal
  - EstimatedGrossProfitTotal
  - ActualSalesTotal
  - ActualGrossProfitTotal
  - CoverageUnitsPct
  - MissingSkuCount
- Persists to `inside_daily_totals_history.csv` (upsert by date)

### 5. Sample Data Files

#### `data_samples/pricebook_sample.csv`
- 10 SKUs with realistic products:
  - Beverages: Red Energy Drink, Blue Gatorade, Coffee Beans, Milk
  - Snacks: Snickers, Doritos
  - Other: Bread, Cigarettes, Lottery Ticket, Air Freshener
- Columns: LevelId, Sku, Name, Department, Category, SubCategory, RetailPrice, UnitCost

#### `data_samples/product_report_sample.csv`
- 16 transactions across Jan 20–21, 2026
- 10 SKUs (9 matched to price book, 1 intentionally unmatched)
- Realistic quantities and sales amounts

#### Updated `data_samples/README.md`
- New "Inside COGS Testing" section with sample file descriptions
- Step-by-step testing instructions
- Expected results for price book and product report
- Consolidated testing guide for both fuel and inside COGS workflows

## Validation

✅ Python syntax verified (no compile errors)  
✅ All 5 helper functions added  
✅ Page added to sidebar navigation  
✅ Sample data files created and formatted  
✅ No existing pages modified (backward compatible)  

## Testing Workflow

1. Run: `streamlit run app.py`
2. Go to "Inside COGS" page
3. Upload `data_samples/pricebook_sample.csv`
4. Upload `data_samples/product_report_sample.csv`
5. Verify 15 matched items, 1 missing SKU, ~94% coverage
6. Select Jan 20 and save daily totals
7. Check `.fuel_profit_data/inside_daily_totals_history.csv` for saved row

## Data Persistence

New CSV files created automatically on save:
- `.fuel_profit_data/pricebook_current.csv` — Current active price book
- `.fuel_profit_data/inside_daily_totals_history.csv` — Running history of daily inside totals

Both files are gitignored and persist across sessions.

## Error Handling

- **Missing price book**: Warning shown, page stops
- **Invalid product report**: Error message with exception details
- **Low coverage**: Warning if < 95% of units matched
- **Column mismatch**: Error with helpful message directing to fallback column names

## Features Implemented

✅ Price book upload (CSV/XLSX with position-based or name-based column extraction)  
✅ SKU normalization (handles Excel artifacts, preserves leading zeros)  
✅ Daily product report parsing  
✅ As-of price/cost application (merged on SKU)  
✅ COGS and profit calculations  
✅ Matched/unmatched item separation  
✅ Coverage metrics and warnings  
✅ Download buttons for missing SKUs and matched items  
✅ Daily totals persistence (upsert by date)  
✅ Sample data for testing  

## User-Facing Documentation

- Inside COGS page has clear section headers and captions
- Expanders for raw data previews
- Inline metrics for quick coverage assessment
- Download buttons for follow-up (add missing SKUs to price book, verify matched items)
- Sample files included in `data_samples/` with step-by-step instructions
