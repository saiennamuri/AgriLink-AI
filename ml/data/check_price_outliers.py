import pandas as pd
import os


# ============================================================
# AGRILINK AI - PRICE OUTLIER ANALYSIS
# M2: AI PRICE PREDICTION
# ============================================================

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

CLEAN_FILE = os.path.join(
    BASE_FOLDER,
    "clean_market_prices.csv"
)

print("=" * 70)
print("AGRILINK AI - PRICE OUTLIER ANALYSIS")
print("=" * 70)

print("\nReading cleaned dataset...")

# We only need the price and identifying columns
columns = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade",
    "Arrival_Date",
    "Min_Price",
    "Max_Price",
    "Modal_Price"
]

# ------------------------------------------------------------
# Store samples of suspicious records
# ------------------------------------------------------------

very_low = []
very_high = []

# ------------------------------------------------------------
# Price statistics
# ------------------------------------------------------------

prices = []

# Read in chunks
for chunk in pd.read_csv(
    CLEAN_FILE,
    usecols=columns,
    chunksize=200000,
    low_memory=False
):

    modal = pd.to_numeric(
        chunk["Modal_Price"],
        errors="coerce"
    )

    prices.extend(
        modal.dropna().tolist()
    )

    # Suspiciously low
    low_rows = chunk[modal < 10]

    if len(low_rows) > 0:
        very_low.append(low_rows)

    # Suspiciously high
    high_rows = chunk[modal > 100000]

    if len(high_rows) > 0:
        very_high.append(high_rows)


# ------------------------------------------------------------
# Convert prices to Series
# ------------------------------------------------------------

price_series = pd.Series(prices)

print("\nTotal prices analysed:", len(price_series))

print("\n--- PRICE DISTRIBUTION ---")

print(
    price_series.describe(
        percentiles=[
            0.001,
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
            0.999
        ]
    )
)


# ------------------------------------------------------------
# Suspicious low values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VERY LOW MODAL PRICES (< ₹10)")
print("=" * 70)

if very_low:

    low_df = pd.concat(
        very_low,
        ignore_index=True
    )

    print("\nNumber of records:", len(low_df))

    print(
        low_df[
            [
                "State",
                "District",
                "Market",
                "Commodity",
                "Variety",
                "Arrival_Date",
                "Min_Price",
                "Max_Price",
                "Modal_Price"
            ]
        ].head(20).to_string(index=False)
    )

else:

    print("No records found.")


# ------------------------------------------------------------
# Suspicious high values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VERY HIGH MODAL PRICES (> ₹1,00,000)")
print("=" * 70)

if very_high:

    high_df = pd.concat(
        very_high,
        ignore_index=True
    )

    print("\nNumber of records:", len(high_df))

    print(
        high_df[
            [
                "State",
                "District",
                "Market",
                "Commodity",
                "Variety",
                "Arrival_Date",
                "Min_Price",
                "Max_Price",
                "Modal_Price"
            ]
        ]
        .sort_values(
            "Modal_Price",
            ascending=False
        )
        .head(30)
        .to_string(index=False)
    )

else:

    print("No records found.")


print("\n" + "=" * 70)
print("OUTLIER ANALYSIS COMPLETED")
print("=" * 70)