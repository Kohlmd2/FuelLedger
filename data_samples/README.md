# Sample Data for Testing FuelLedger

This folder contains example Petro Outlet transaction data to test FuelLedger locally.

## Sample Files

### `petro_outlet_sample.csv`

A mock Petro Outlet transaction export with **12 transactions** across:
- **2 dates**: January 19–20, 2026
- **3 grades**: 87 (REGUNL), 89 (PLSUNL), 93 (SUPUNL)
- **Multiple payment methods**: cash, debit cards, credit cards

All values are fictional and designed for easy testing.

| Transaction | Date | Grade | Gallons | Amount | Tender |
|-------------|------|-------|---------|--------|--------|
| 1 | 2026-01-19 | 87 | 125.45 gal | $387.89 | Cash |
| 2 | 2026-01-19 | 87 | 98.32 gal | $303.42 | Credit |
| 3–12 | ... | 89, 93 | Various | Various | Mixed |

## How to Test

1. **Open FuelLedger**:
   ```bash
   streamlit run app.py
   ```

2. **Go to the Calculator page** (default view)

3. **Upload the CSV**:
   - Click "Upload Petro Outlet CSV"
   - Select `petro_outlet_sample.csv`

4. **Verify the data**:
   - Expand "Preview RAW CSV" to see the 12 transactions
   - Expand "Preview CLEAN transactions" to confirm parsing (should show Date, Grade, Gallons, FinalAmount, TenderType, PriceTier)

5. **Enter test prices**:
   - **87 (Regular)**: Cash = $3.10, Credit = $3.20
   - **89 (Plus)**: Cash = $3.35, Credit = $3.45
   - **93 (Super)**: Cash = $3.60, Credit = $3.70
   - **All Cost/Gallon**: $2.75

   _(These are deliberately round numbers for easy mental verification.)_

6. **Review the profit table**:
   - Expected Revenue, Actual POS Revenue, COGS, Gross Profit, and Net Profit should all compute
   - Apply a 2.75% credit card fee rate
   - Check the "Daily Totals (all grades)" section

7. **Save to history**:
   - Click "Save these daily totals to History"
   - Go to "Daily Totals History" page to verify persistence

## Expected Results (Rough)

With the suggested prices and 2.75% credit card fees:

**January 19, 2026**:
- Total Gallons: ~485 gal
- Cash: ~$1,255; Credit: ~$936
- Expected & Actual Revenue: ~$2,191
- COGS (~$1,334): 485 × $2.75
- Credit Card Fees: ~$26
- **Net Fuel Profit: ~$830** (before fees: ~$857)

**January 20, 2026**:
- Total Gallons: ~496 gal
- Cash: ~$1,360; Credit: ~$667
- Expected & Actual Revenue: ~$2,027
- COGS (~$1,364): 496 × $2.75
- Credit Card Fees: ~$18
- **Net Fuel Profit: ~$645** (before fees: ~$663)

Use these as sanity checks. Exact values depend on your posted prices and fee rate.

## Extending the Sample

To create a larger test dataset:
1. Duplicate rows and change dates to span a full month
2. Add more grades or payment methods
3. Introduce intentional errors (missing dates, invalid grades) to test validation

## Notes

- **All values are fake** and chosen for testing convenience, not realism
- No real transaction or PII data is included
- This file is checked into Git; use it freely for development and testing
