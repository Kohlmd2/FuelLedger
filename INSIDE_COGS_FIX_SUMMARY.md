# Inside COGS Crash Fix - Summary of Changes

## Overview
Fixed the Inside COGS page crash caused by the Name column conflict during merge operation ("'Name' not in index" error). Implemented robust money field parsing and unified SKU handling.

## Changes Made

### 1. Added `parse_money()` Helper Function (Line 398-410)
- Parses money fields by stripping `$`, commas, and whitespace
- Handles NaN and invalid values gracefully
- Returns `np.nan` on parse errors instead of failing

### 2. Updated Price Book Loading (Lines 828-849)
- Changed SKU column name from `Sku` to `SKU` (normalized key)
- Applied `parse_money()` to `RetailPrice` and `UnitCost` columns
- Added fallback path that renames `Sku` → `SKU` for column-name-based extraction

### 3. Updated Product Report Processing (Lines 870-900)
- Changed SKU column name from `Sku` to `SKU`
- Applied `parse_money()` to `ActualUnitPrice` and `LineTotal` columns
- Fixed computation of `ActualSales`: now handles NaN and zero properly using mask
- Aggregation by `(DateSold, SKU)` instead of `(DateSold, Sku)`

### 4. Fixed Merge with Explicit Suffixes (Lines 902-925)
- Added `suffixes=("_sale", "_pb")` to merge call
- After merge, creates unified `Name` column: `merged["Name"] = merged["Name_sale"].fillna(merged["Name_pb"])`
- Drops `Name_sale` and `Name_pb` after consolidation
- Uses merge indicator to split matched vs. unmatched SKUs

### 5. Updated Results Display (Lines 928-1020)
- **Matched Items Table**: Shows columns: Date Sold, SKU, Name, Qty, Retail Price, Unit Cost, COGS, Est. Retail Sales, Actual Sales, Gross Profit
- **Unmatched SKUs Table**: Grouped by SKU, shows total quantities and sales
- **Missing Unit Cost Section**: Enhanced with:
  - 3 metrics: Items w/ Missing Cost, Missing Cost SKUs, Qty Affected
  - Grouped table by SKU sorted by quantity
  - Download button for CSV export

### 6. Updated Save Daily Inside Totals (Lines 1072-1082)
- Fixed references to use new `GrossProfit` column (computed as `ActualSales - COGS`)
- Fixed reference from `unmatched` to `unmatched_skus`

## Key Improvements

1. **Eliminates Name Conflict**: Explicit suffixes prevent the merge from creating ambiguous `Name_x`/`Name_y` columns
2. **Money Field Safety**: `parse_money()` handles currency formatting (e.g., `$1,234.56`)
3. **Unified SKU Key**: Single `SKU` column used consistently across both datasets
4. **Better Unmatched SKUs View**: Now shows aggregated totals by SKU instead of per-transaction rows
5. **Enhanced Missing Cost Reporting**: Separate metrics and grouped table for better visibility

## Testing

All changes tested with:
- Parse money function: Handles `$`, commas, invalid values
- Merge logic: Correctly joins product report to pricebook with explicit suffixes
- Missing cost detection: Properly identifies and aggregates items with zero/NaN unit cost
- Unmatched SKUs: Shows only unique SKUs with total quantities

## Backward Compatibility

- **No breaking changes to fuel pages** (Calculator, Daily Totals History, Tank Deliveries)
- **Existing data format preserved** in CSV storage
- **Session state handling** remains unchanged

## Remaining Notes

- Pre-existing issue: `date` import missing in Tank Deliveries page (line 674) - not introduced by these changes
- All column references updated from `Sku` to `SKU` throughout Inside COGS section
