from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.ecommerce_intelligence import config
from src.ecommerce_intelligence.data_processing import ensure_directories


PRODUCTS = [
    ("10001", "RED HEART CERAMIC MUG", 3.5),
    ("10002", "WHITE CHRISTMAS STAR DECORATION", 2.2),
    ("10003", "VINTAGE GLASS CANDLE HOLDER", 4.8),
    ("10004", "WOODEN PHOTO FRAME", 6.0),
    ("10005", "BLUE POLKA DOT GIFT BAG", 1.5),
    ("10006", "SET OF TEA LIGHTS", 2.9),
    ("10007", "HANDMADE BIRTHDAY CARD", 1.2),
    ("10008", "CERAMIC STORAGE JAR", 5.5),
]


def main() -> None:
    ensure_directories()
    random.seed(42)
    rows = []
    start = datetime(2011, 1, 1)
    customers = list(range(17001, 17031))

    invoice_number = 500000
    early_dropout_customers = set(customers[-8:])

    for week in range(48):
        eligible_customers = [
            customer
            for customer in customers
            if not (customer in early_dropout_customers and week > 25)
        ]
        active_customers = random.sample(eligible_customers, k=random.randint(8, 16))
        for customer in active_customers:
            invoice_number += 1
            date = start + timedelta(days=week * 7 + random.randint(0, 6))
            basket = random.sample(PRODUCTS, k=random.randint(1, 4))
            for stock_code, description, price in basket:
                rows.append(
                    {
                        "Invoice": invoice_number,
                        "StockCode": stock_code,
                        "Description": description,
                        "Quantity": random.randint(1, 8),
                        "InvoiceDate": date,
                        "Price": price,
                        "Customer ID": customer,
                        "Country": random.choice(["United Kingdom", "France", "Germany"]),
                    }
                )

    # Add a few cancellations so the cleaning rules have something to detect.
    for row in rows[:5]:
        cancelled = row.copy()
        cancelled["Invoice"] = "C" + str(cancelled["Invoice"])
        cancelled["Quantity"] = -1
        rows.append(cancelled)

    sample = pd.DataFrame(rows)
    sample.to_csv(config.SAMPLE_RAW, index=False)
    print(f"Sample data created: {config.SAMPLE_RAW}")


if __name__ == "__main__":
    main()
