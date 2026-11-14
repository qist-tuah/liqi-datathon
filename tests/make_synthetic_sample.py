"""Writes a small FAKE csv into data/raw/ so the pipeline can be smoke-tested
without the real dataset. Column names here are guesses - delete the file
before putting the real dataset in data/raw/.

Usage (from the project root):  python tests/make_synthetic_sample.py
"""
import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "synthetic_sample.csv"
random.seed(42)
bics = ["0621US30", "0219US30", "0024TRI0", "0380CNB0", "0007JEH0", "0295GB20", "0017DEF0", "0087ZAJ0"]
header = ["", "TIMESTAMP", "CURRENCY", "USD_AMOUNT", "ORDERING_BIC", "BENEFICIARY_BIC", "BIC8_PATH",
          "TOTAL_DURATION", "CONTINUITY", "CURRENCY_CHANGES", "PARTICIPANTS", "TRANSACTION_TYPE",
          "ORDERING_NAME", "BENEFICIARY_NAME", "ORDERING_ACCOUNT", "BENEFICIARY_ACCOUNT",
          "ORDERING_CATEGORY", "BENEFICIARY_CATEGORY", "ANOMALY_TYPE"]
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    for i in range(5000):
        a, b = random.sample(bics, 2)
        w.writerow([
            "WARNING: fictitious data" if i == 0 else "",
            1_690_000_000 + random.randint(0, 5_000_000),
            random.choice(["USD", "EUR", "GBP", "NAD", "JPY"]),
            random.choice([500, round(random.uniform(5, 250000), 2)]),
            a, b, f"{a};{b}",
            random.randint(1, 5000),
            random.choice(["Y", "Y", "Y", "N"]),
            random.randint(0, 3), random.randint(2, 6),
            random.choice(["Direct", "Indirect", "Covered"]),
            f"_FAKE_{random.randint(0, 99999):05d}",
            random.choice(["_FAKE_NA", "NA", "line one\nline two", f"_FAKE_{i}"]),
            f"/{random.randint(0, 9999999):07d}",
            f"{random.randint(0, 9999999):07d}",
            random.choice(["B", "C"]), random.choice(["B", "C"]),
            random.choice([""] * 30 + ["STRUCTURING", "ROUND_AMOUNT"]),
        ])
print(f"wrote {OUT}")
