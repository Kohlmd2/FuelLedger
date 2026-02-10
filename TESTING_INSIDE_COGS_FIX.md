# Inside COGS Fix - Testing Guide

## Prerequisites
- Streamlit installed: `pip install streamlit`
- Sample data files available in `data_samples/`

## Running the App
```bash
cd /Users/kohl/Desktop/gas-profit-app
streamlit run app.py
```

## Test Workflow

### Step 1: Upload Price Book
1. Navigate to "Inside COGS" page
2. Click "Upload Price Book CSV/XLSX"
3. Use `data_samples/pricebook_sample.csv`
   - Should load with columns: SKU, Name, RetailPrice, UnitCost
   - System handles money formats like `$10.50`

### Step 2: Upload Product Report
1. Click "Upload ProductReportExport CSV"
2. Use `data_samples/product_report_sample.csv`
   - Should merge with price book without "Name not in index" error
   - Matched items should show with unified Name from product report

### Step 3: Verify Outputs

#### Matched Items Table
- Check columns: Date Sold, SKU, Name, Qty, Retail Price, Unit Cost, COGS, Est. Retail Sales, Actual Sales, Gross Profit
- Verify Name column is present (not Name_x or Name_pb)
- No "Name not in index" error

#### Unmatched SKUs Section
- Shows unique SKUs that were sold but not in price book
- Grouped by SKU with total quantities
- Download CSV button works

#### Missing Unit Cost Section
- Displays 3 metrics: Items w/ Missing Cost, Missing Cost SKUs, Qty Affected
- Shows table of items with missing/zero unit costs
- Download CSV button works

### Step 4: Edge Cases to Test

1. **Money Field Parsing**
   - Price book with `$1,234.56` format
   - Should parse correctly as 1234.56

2. **Missing Unit Costs**
   - Add price book entries with UnitCost = 0 or empty
   - Should appear in "Missing Unit Cost" section

3. **Unmatched SKUs**
   - Product report with SKU not in price book
   - Should appear in "Unmatched SKUs" section

4. **Save Daily Totals**
   - Select date and click "Save Daily Inside Totals"
   - Should complete without errors
   - Data saved to `.fuel_profit_data/inside_daily_totals_history.csv`

## Expected Behavior Changes

### Before Fix
- ❌ Crash with "'Name' not in index" error
- ❌ No way to distinguish product report name vs price book name
- ❌ Unmatched SKUs showed all rows, not aggregated

### After Fix
- ✅ Merge succeeds with explicit suffixes
- ✅ Single Name column preferred from product report
- ✅ Money fields with $ and commas parse correctly
- ✅ Unmatched SKUs aggregated by unique SKU
- ✅ Missing Unit Cost reporting enhanced with metrics

## Troubleshooting

### "['Name'] not in index" Still Appears
- Likely cached session state
- Press Ctrl+R to refresh Streamlit
- Clear browser cache

### Money Fields Not Parsing
- Check if file contains actual `$` symbols
- parse_money function should handle it
- If still failing, check column indices match expected positions

### Missing SKUs Not Showing
- Verify product report SKUs don't match price book exactly
- Check SKU normalization (leading zeros, whitespace, .0 removal)
- Use preview to see actual SKU values

## File Locations

- Updated: `/Users/kohl/Desktop/gas-profit-app/app.py`
- Sample data: `/Users/kohl/Desktop/gas-profit-app/data_samples/`
- Runtime data: `/Users/kohl/Desktop/gas-profit-app/.fuel_profit_data/`
