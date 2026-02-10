# Inside COGS Page Fix - Complete Implementation Summary

## Problem Statement
The Inside COGS page crashed with error: `"['Name'] not in index"` when merging ProductReportExport with the price book. This occurred because both DataFrames had a "Name" column, and pandas defaulted to creating `Name_x` and `Name_y` suffixed columns, making the original "Name" reference invalid.

## Root Causes Fixed

### 1. **Name Column Collision on Merge**
   - **Problem**: Both product report and price book had "Name" columns
   - **Solution**: Explicit suffixes `("_sale", "_pb")` on merge, then consolidate with `merged["Name"] = merged["Name_sale"].fillna(merged["Name_pb"])`

### 2. **Money Field Parsing**
   - **Problem**: Retail Price and Unit Cost may contain `$`, commas, or whitespace
   - **Solution**: New `parse_money()` function strips currency symbols and parses to float

### 3. **Inconsistent SKU Column Names**
   - **Problem**: Product report uses "Sku", price book uses "Sku" → inconsistent after loading
   - **Solution**: Normalize both to "SKU" uppercase for consistent join key

### 4. **Poor Unmatched SKU Reporting**
   - **Problem**: Showed all rows, making it hard to see unique unmatched SKUs
   - **Solution**: Aggregate by SKU with total quantities

### 5. **Missing Unit Cost Visibility**
   - **Problem**: Hidden in data, no specific reporting
   - **Solution**: Dedicated section with metrics and grouped table

## Implementation Details

### New Helper Function: `parse_money()` (Lines 398-410)
```python
def parse_money(value):
    """Parse a money field: strip $, commas, whitespace; convert to float."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return np.nan
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan
```

### Price Book Loading Update (Lines 828-849)
**Before**: Used "Sku" with numeric coercion
**After**: 
- Renamed to "SKU" for consistency
- Applied `parse_money()` to RetailPrice and UnitCost
- Handles both column-position and column-name extraction

### Product Report Processing (Lines 870-900)
**Before**: Used "Sku", numeric coercion, simple division for ActualSales
**After**:
- Renamed to "SKU"
- Applied `parse_money()` to ActualUnitPrice and LineTotal
- Safe ActualSales calculation with mask for zero/NaN values

### Merge Operation (Lines 899-925)
**Before**:
```python
matched = agg.merge(current_pb_for_merge, on="Sku", how="left", indicator=True)
# Result: Name_x, Name_y columns → 'Name' not found later
```

**After**:
```python
merged = agg.merge(
    current_pb_for_merge, 
    on="SKU", 
    how="left", 
    suffixes=("_sale", "_pb"),
    indicator=True
)
merged["Name"] = merged["Name_sale"].fillna(merged["Name_pb"])
merged = merged.drop(columns=["Name_sale", "Name_pb"], errors="ignore")
```

### Results Display (Lines 928-1020)

#### Matched Items Table
- **Columns**: Date Sold, SKU, Name, Qty, Retail Price, Unit Cost, COGS, Est. Retail Sales, Actual Sales, Gross Profit
- **Verified**: No "Name" column conflicts, all data accessible
- **Download**: CSV export available

#### Unmatched SKUs Section (NEW)
- **Grouping**: Aggregated by SKU with total quantities
- **Display**: SKU, Name, Qty Sold, Total Sales (sorted by quantity descending)
- **Download**: CSV export available
- **Improvement**: Much clearer than showing all individual rows

#### Missing Unit Cost Section (ENHANCED)
- **Metrics**: 
  - Items w/ Missing Cost (total row count)
  - Missing Cost SKUs (unique count)
  - Qty Affected (total units sold)
- **Table**: Grouped by SKU with retail price and unit cost
- **Download**: CSV export for price book updates
- **Success State**: Green message if no missing costs

### Save Daily Inside Totals (Lines 1072-1082)
**Updated**: References now use `GrossProfit` (computed as ActualSales - COGS) and `unmatched_skus` instead of old `unmatched`

## Testing Results

### Test Data Used
- Product Report: 4 SKUs (3 matched, 1 unmatched)
- Price Book: 3 SKUs (1 with cost, 2 without)

### Results
✅ parse_money function:
- `parse_money("$1,234.56")` → 1234.56
- `parse_money("invalid")` → NaN

✅ Merge logic:
- Matched items correctly identified (3 rows)
- Unmatched SKUs correctly identified (1 row)
- Name column successfully consolidated

✅ Missing Unit Cost:
- Identified 2 SKUs with missing/zero cost
- Total qty affected: 35 units
- Metrics and table generated correctly

## Column Mappings

### Price Book
| Old | New |
|-----|-----|
| Sku | SKU |
| Name | Name |
| RetailPrice | RetailPrice |
| UnitCost | UnitCost |

### Matched Items Output
| Column | Source | Computation |
|--------|--------|-------------|
| DateSold | Product Report | Original |
| SKU | Join Key | Both |
| Name | Consolidated | Product Report preferred |
| Quantity | Product Report | Aggregated sum |
| RetailPrice | Price Book | Direct |
| UnitCost | Price Book | Direct |
| COGS | Computed | Quantity × UnitCost |
| RetailSalesEstimate | Computed | Quantity × RetailPrice |
| ActualSales | Product Report | Calculated or line total |
| GrossProfit | Computed | ActualSales - COGS |

## Breaking Changes
**None** - All changes are backward compatible:
- Fuel pages (Calculator, Daily Totals, Tank Deliveries) unaffected
- CSV storage format unchanged
- Session state handling preserved

## Known Issues (Pre-Existing)
- Tank Deliveries page has missing `date` import (line 674) - not caused by these changes

## Files Modified
- `/Users/kohl/Desktop/gas-profit-app/app.py` (main implementation)

## Documentation Created
- `INSIDE_COGS_FIX_SUMMARY.md` - Technical summary
- `TESTING_INSIDE_COGS_FIX.md` - Testing guide

## Deployment Notes
1. Replace app.py with fixed version
2. No data migration needed
3. Existing pricebook and product reports will work with new logic
4. Test with sample data first: `data_samples/pricebook_sample.csv` and `product_report_sample.csv`
