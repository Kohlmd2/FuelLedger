import unittest
from datetime import date

import pandas as pd

from fuel import build_profit_table, clean_transactions, summarize_by_day
from inside_cogs import normalize_sku


class CoreLogicTests(unittest.TestCase):
    def test_clean_transactions_maps_and_filters(self):
        raw = pd.DataFrame(
            [
                {"Date Sold": "2026-02-10", "Grade Name": "REGUNL", "Gallons Sold": "2", "Final Amount": "6.00", "Primary Tender Code": "cash"},
                {"Date Sold": "2026-02-10", "Grade Name": "PLSUNL", "Gallons Sold": "1", "Final Amount": "3.00", "Primary Tender Code": "creditCards"},
                {"Date Sold": "2026-02-10", "Grade Name": "BADGRADE", "Gallons Sold": "5", "Final Amount": "10.00", "Primary Tender Code": "cash"},
                {"Date Sold": "not-a-date", "Grade Name": "REGUNL", "Gallons Sold": "1", "Final Amount": "2.00", "Primary Tender Code": "other"},
            ]
        )

        cleaned = clean_transactions(raw)
        self.assertEqual(len(cleaned), 2)
        self.assertSetEqual(set(cleaned["Grade"].tolist()), {87, 89})
        self.assertSetEqual(set(cleaned["TenderType"].tolist()), {"CASH", "CREDIT"})
        self.assertSetEqual(set(cleaned["PriceTier"].tolist()), {"CASH", "CREDIT"})

    def test_summarize_by_day_aggregates(self):
        cleaned = pd.DataFrame(
            [
                {"Date": date(2026, 2, 10), "Grade": 87, "Gallons": 2.0, "FinalAmount": 6.0, "TenderType": "CASH", "PriceTier": "CASH"},
                {"Date": date(2026, 2, 10), "Grade": 87, "Gallons": 1.0, "FinalAmount": 3.5, "TenderType": "CREDIT", "PriceTier": "CREDIT"},
            ]
        )
        daily = summarize_by_day(cleaned)
        row = daily.iloc[0]
        self.assertEqual(row["Gallons_CASH"], 2.0)
        self.assertEqual(row["Gallons_CREDIT"], 1.0)
        self.assertEqual(row["TotalGallons"], 3.0)
        self.assertEqual(row["POSRevenue"], 9.5)

    def test_build_profit_table_calculates_expected_values(self):
        summary = pd.DataFrame(
            [
                {
                    "Date": pd.Timestamp("2026-02-10"),
                    "Grade": 87,
                    "Gallons_CASH": 1.0,
                    "Gallons_CREDIT": 1.0,
                    "POSAmount_CASH": 2.0,
                    "POSAmount_CREDIT": 3.0,
                    "TotalGallons": 2.0,
                    "POSRevenue": 5.0,
                }
            ]
        )
        prices = pd.DataFrame(
            [
                {"Date": pd.Timestamp("2026-02-01"), "Grade": 87, "CashPrice": 2.0, "CreditPrice": 3.0},
            ]
        )
        costs = pd.DataFrame(
            [
                {"Date": pd.Timestamp("2026-02-01"), "Grade": 87, "CostPerGallon": 1.0},
            ]
        )

        out = build_profit_table(summary, prices, costs, 0.10)
        row = out.iloc[0]
        self.assertAlmostEqual(row["ExpectedRevenue"], 5.0, places=6)
        self.assertAlmostEqual(row["COGS"], 2.0, places=6)
        self.assertAlmostEqual(row["GrossProfit"], 3.0, places=6)
        self.assertAlmostEqual(row["CreditCardFees"], 0.3, places=6)
        self.assertAlmostEqual(row["NetFuelProfit"], 2.7, places=6)
        self.assertFalse(bool(row["_MissingPrice"]))
        self.assertFalse(bool(row["_MissingCost"]))

    def test_build_profit_table_blends_89_cost(self):
        summary = pd.DataFrame(
            [
                {
                    "Date": pd.Timestamp("2026-02-10"),
                    "Grade": 89,
                    "Gallons_CASH": 1.0,
                    "Gallons_CREDIT": 0.0,
                    "POSAmount_CASH": 3.0,
                    "POSAmount_CREDIT": 0.0,
                    "TotalGallons": 1.0,
                    "POSRevenue": 3.0,
                }
            ]
        )
        prices = pd.DataFrame(
            [
                {"Date": pd.Timestamp("2026-02-01"), "Grade": 89, "CashPrice": 3.0, "CreditPrice": 3.2},
            ]
        )
        costs = pd.DataFrame(
            [
                {"Date": pd.Timestamp("2026-02-01"), "Grade": 87, "CostPerGallon": 1.0},
                {"Date": pd.Timestamp("2026-02-01"), "Grade": 93, "CostPerGallon": 3.0},
            ]
        )
        out = build_profit_table(summary, prices, costs, 0.0)
        row = out.iloc[0]
        self.assertAlmostEqual(row["CostPerGallon"], 2.0, places=6)
        self.assertAlmostEqual(row["COGS"], 2.0, places=6)

    def test_normalize_sku(self):
        self.assertEqual(normalize_sku(" 074323097504.0 "), "074323097504")
        self.assertEqual(normalize_sku("74 3230-97504"), "74323097504")
        self.assertEqual(normalize_sku(None), "")


if __name__ == "__main__":
    unittest.main()
