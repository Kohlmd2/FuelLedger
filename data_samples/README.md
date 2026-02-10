# Sample Data for Testing FuelLedger

This folder contains example transaction data and price books to test FuelLedger locally.

## Sample Files

### Fuel Testing (Calculator & Tank Deliveries)

#### `petro_outlet_sample.csv`

A mock Petro Outlet transaction export with **12 transactions** across:
- **2 dates**: January 19–20, 2026
- **3 grades**: 87 (REGUNL), 89 (PLSUNL), 93 (SUPUNL)
- **Multiple payment methods**: cash, debit cards, credit cards

### Inside COGS Testing (Price Book & Product Report)

#### `pricebook_sample.csv`

A sample price book with **10 SKUs** including:
- Beverages: Energy drinks, Gatorade, Coffee, Milk
- Snacks: Candy, Chips
- Other: Bread, Cigarettes, Lottery, Air freshener

#### `product_report_sample.csv`

A sample ProductReportExport with **16 transactions** across:
- **2 dates**: January 20–21, 2026
- **10 SKUs** (mostly matched to price book; 1 intentionally unmatched)
- Various quantities and transactions

## How to Test

### Fuel Profit Calculator

1. **Open FuelLedger**: `streamlit run app.py`

2. **Go to Calculator page** → Upload `petro_outlet_sample.csv`

3. **Enter test prices**:
   - **87 (Regular)**: Cash = $3.10, Credit = $3.20
   - **89 (Plus)**: Cash = $3.35, Credit = $3.45
   - **93 (Super)**: Cash = $3.60, Credit = $3.70
   - **All Cost/Gallon**: $2.75

4. **Review results** and save to history

### Inside COGS (Price Book & Product Report)

1. **Go to Inside COGS page** (sidebar navigation)

2. **Upload Price Book**: Select `pricebook_sample.csv`
   - Should show "Price book loaded: 10 SKUs"

3. **Upload Product Report**: Select `product_report_sample.csv`
   - Matched Items table shows 15 items (94% coverage)
   - Missing SKUs shows 1 unmatched item (SKU 4911234567890)

4. **Save daily totals** for the selected date

## Expected Results

**Fuel (Jan 19)**: ~485 gal, $2,191 revenue, ~$830 net profit  
**Fuel (Jan 20)**: ~496 gal, $2,027 revenue, ~$645 net profit  
**Inside COGS (Jan 20)**: ~17 units, ~$42 COGS, ~$54 retail estimate, ~$12 profit

## Notes

- All values are fake and chosen for testing convenience
- No real transaction or PII data included
- Use freely for development and testing
