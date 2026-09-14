import pandas as pd
import numpy as np
import os


# ============================================================
# AGRILINK AI - PREPARE ML DATASET
# M2: AI PRICE PREDICTION
# ============================================================

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_FOLDER,
    "clean_market_prices.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_FOLDER,
    "ml_ready_prices.csv"
)

print("=" * 70)
print("AGRILINK AI - ML DATA PREPARATION")
print("=" * 70)

print("\nReading dataset...")

# ------------------------------------------------------------
# Read required columns
# ------------------------------------------------------------

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
    "Modal_Price",
    "Commodity_Code"
]


# ------------------------------------------------------------
# First pass:
# Calculate commodity-wise Q1 and Q3
# ------------------------------------------------------------

print("\nCalculating commodity-wise price limits...")

all_stats = []

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=columns,
    chunksize=200000,
    low_memory=False
):

    chunk["Modal_Price"] = pd.to_numeric(
        chunk["Modal_Price"],
        errors="coerce"
    )

    grouped = chunk.groupby("Commodity")["Modal_Price"]

    stats = grouped.agg(
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
    )

    stats["IQR"] = stats["Q3"] - stats["Q1"]

    all_stats.append(stats)


# ------------------------------------------------------------
# Combine statistics
# ------------------------------------------------------------

stats = pd.concat(all_stats)

# Because statistics were calculated per chunk,
# calculate final limits from approximate combined values.
# For our large dataset, we use the 1st and 99th percentile
# from the complete commodity distributions in a second pass.
# ------------------------------------------------------------

print("Calculating final commodity-wise limits...")

commodity_prices = {}

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=["Commodity", "Modal_Price"],
    chunksize=200000,
    low_memory=False
):

    chunk["Modal_Price"] = pd.to_numeric(
        chunk["Modal_Price"],
        errors="coerce"
    )

    for commodity, values in chunk.groupby("Commodity")["Modal_Price"]:

        if commodity not in commodity_prices:
            commodity_prices[commodity] = []

        commodity_prices[commodity].extend(
            values.dropna().tolist()
        )


# ------------------------------------------------------------
# Build robust limits
# ------------------------------------------------------------

limits = {}

for commodity, values in commodity_prices.items():

    prices = np.array(values)

    q1 = np.percentile(prices, 25)
    q3 = np.percentile(prices, 75)

    iqr = q3 - q1

    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)

    # Price cannot be negative
    lower = max(0, lower)

    limits[commodity] = (
        lower,
        upper
    )


# Free memory
del commodity_prices


# ------------------------------------------------------------
# Second pass:
# Filter dataset
# ------------------------------------------------------------

print("\nFiltering anomalous records...")

total_rows = 0
removed_rows = 0
saved_rows = 0

first_write = True


for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=columns,
    chunksize=200000,
    low_memory=False
):

    total_rows += len(chunk)

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    chunk["Min_Price"] = pd.to_numeric(
        chunk["Min_Price"],
        errors="coerce"
    )

    chunk["Max_Price"] = pd.to_numeric(
        chunk["Max_Price"],
        errors="coerce"
    )

    chunk["Modal_Price"] = pd.to_numeric(
        chunk["Modal_Price"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Date conversion
    # --------------------------------------------------------

    chunk["Arrival_Date"] = pd.to_datetime(
        chunk["Arrival_Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Basic validity
    # --------------------------------------------------------

    valid = (
        chunk["Arrival_Date"].notna()
        &
        chunk["Min_Price"].notna()
        &
        chunk["Max_Price"].notna()
        &
        chunk["Modal_Price"].notna()
    )

    # --------------------------------------------------------
    # Price relationship
    #
    # Min <= Modal <= Max
    # --------------------------------------------------------

    valid = valid & (
        (chunk["Min_Price"] <= chunk["Modal_Price"])
        &
        (chunk["Modal_Price"] <= chunk["Max_Price"])
    )

    # --------------------------------------------------------
    # Commodity-wise IQR filtering
    # --------------------------------------------------------

    keep = []

    for commodity, price in zip(
        chunk["Commodity"],
        chunk["Modal_Price"]
    ):

        if commodity not in limits:

            keep.append(False)
            continue

        lower, upper = limits[commodity]

        keep.append(
            lower <= price <= upper
        )

    keep = np.array(keep)

    valid = valid & keep

    cleaned = chunk[valid].copy()

    removed_rows += len(chunk) - len(cleaned)
    saved_rows += len(cleaned)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cleaned.to_csv(
        OUTPUT_FILE,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )

    first_write = False

    print(
        f"Processed: {total_rows:,} | "
        f"Saved: {saved_rows:,} | "
        f"Removed: {removed_rows:,}"
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("ML DATA PREPARATION COMPLETED")
print("=" * 70)

print(f"\nOriginal rows: {total_rows:,}")
print(f"Rows removed: {removed_rows:,}")
print(f"Final ML rows: {saved_rows:,}")

print("\nML dataset saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)