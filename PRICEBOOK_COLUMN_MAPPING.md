# Pricebook Column Mapping Implementation

## Overview
Enhanced the pricebook uploader to support multiple column name aliases, making it compatible with various POS export formats including Petro Outlet exports.

## Changes Made

### 1. New Helper Function: `map_pricebook_columns()` (Lines 413-474)
Maps pricebook columns to standard names using intelligent alias detection.

**Supported Aliases:**

| Field | Aliases |
|-------|---------|
| **SKU** | Sku, SKU, UPC, PLU, ItemCode |
| **Name** | Name, Description, Item Name |
| **RetailPrice** | RetailPrice, Retail, Price, Sell Price |
| **UnitCost** | UnitCost, Cost, Unit Cost, Avg Cost |

**Key Features:**
- Case-insensitive column matching
- Returns dictionary mapping standard names to actual column names
- Raises descriptive ValueError if required columns missing
- Validates all 4 required fields are present

### 2. Updated Pricebook Upload Logic (Lines 868-907)
Modified upload handler to use flexible column mapping:

**Flow:**
1. **Primary**: Try `map_pricebook_columns()` - attempts to match by aliases
2. **Fallback**: If aliases not found, try position-based extraction (8+ columns)
3. **Error**: Show helpful message with supported column names if both fail

**Error Message:**
Users see:
```
Price book columns not recognized. Supported column names:
- SKU: Sku, SKU, UPC, PLU, ItemCode
- Name: Name, Description, Item Name
- RetailPrice: RetailPrice, Retail, Price, Sell Price
- UnitCost: UnitCost, Cost, Unit Cost, Avg Cost

Or provide 8+ columns with standard Excel layout (B=SKU, C=Name, G=Price, H=Cost)
```

## Testing Results

All tests passed:
✅ Standard column names (Sku, Name, RetailPrice, UnitCost)
✅ Petro Outlet style (UPC, Description, Retail, Avg Cost)
✅ Alternative aliases (PLU, Item Name, Price, Cost)
✅ Missing column detection (proper error messaging)
✅ Case-insensitive matching (sku, name, retailPrice, etc.)

## Example Usage Scenarios

### Scenario 1: Petro Outlet Export
```
Input CSV columns: UPC, Description, Retail, Avg Cost
Result: ✓ Detected and mapped correctly
```

### Scenario 2: Generic POS Export
```
Input CSV columns: PLU, Item Name, Price, Cost
Result: ✓ Detected and mapped correctly
```

### Scenario 3: Mixed Case Names
```
Input CSV columns: sku, itemName, retailPrice, unitCost
Result: ✓ Case-insensitive matching works
```

### Scenario 4: Standard Format
```
Input CSV columns: Sku, Name, RetailPrice, UnitCost
Result: ✓ Works as before (backward compatible)
```

## Data Processing

After column mapping:
1. SKU values normalized with `normalize_sku()`
2. Price/cost fields parsed with `parse_money()` (handles $, commas)
3. Data stored with standard schema (SKU, Name, RetailPrice, UnitCost)

## Backward Compatibility
✅ Fully backward compatible:
- Existing pricebooks continue to work
- Standard column names still supported
- Position-based fallback remains as safety net
- All downstream processing unchanged

## Files Modified
- `/Users/kohl/Desktop/gas-profit-app/app.py`
  - Added: `map_pricebook_columns()` function (lines 413-474)
  - Updated: Pricebook upload handler (lines 868-907)

## Benefits

1. **Flexibility**: Supports multiple column naming conventions
2. **User-Friendly**: Clear error messages guide users on supported formats
3. **Robustness**: Case-insensitive matching reduces user input errors
4. **Maintainability**: Centralized alias definitions in one place
5. **Extensibility**: Easy to add new aliases by updating sets

## Implementation Notes

The column detection uses a priority-based approach:
- Iterates through alias list in order
- Uses first match found (case-insensitive)
- Ensures deterministic behavior
- Returns actual column name from DataFrame for extraction

## Future Enhancements (Optional)

Could add:
- Interactive column picker UI if mapping fails
- Preview of mapped columns before save
- Auto-detection of column patterns in data
- Custom column mapping save/load
