import pandas as pd
import os
import glob


# ============================================================
# AGRILINK AI - DATA VALIDATION
# M2: AI PRICE PREDICTION
# ============================================================

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

CLEAN_FILE = os.path.join(
    BASE_FOLDER,
    "clean_market_prices.csv"
)

if not os.path.exists(CLEAN_FILE):
    print("Clean dataset not found!")
    exit()


print("=" * 70)
print("AGRILINK AI - CLEAN DATA VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

total_rows = 0

states = set()
commodities = set()
markets = set()
districts = set()

year_counts = {}

min_date = None
max_date = None

min_price = None
max_price = None

# ------------------------------------------------------------
# Read cleaned data in chunks
# ------------------------------------------------------------

for chunk in pd.read_csv(
    CLEAN_FILE,
    chunksize=200000,
    low_memory=False
):

    total_rows += len(chunk)

    # Convert date
    chunk["Arrival_Date"] = pd.to_datetime(
        chunk["Arrival_Date"],
        errors="coerce"
    )

    # Unique states
    states.update(
        chunk["State"].dropna().unique()
    )

    # Unique commodities
    commodities.update(
        chunk["Commodity"].dropna().unique()
    )

    # Unique markets
    markets.update(
        chunk["Market"].dropna().unique()
    )

    # Unique districts
    districts.update(
        chunk["District"].dropna().unique()
    )

    # Date range
    chunk_min_date = chunk["Arrival_Date"].min()
    chunk_max_date = chunk["Arrival_Date"].max()

    if min_date is None or chunk_min_date < min_date:
        min_date = chunk_min_date

    if max_date is None or chunk_max_date > max_date:
        max_date = chunk_max_date

    # Year counts
    years = chunk["Arrival_Date"].dt.year.value_counts()

    for year, count in years.items():

        year = int(year)

        year_counts[year] = (
            year_counts.get(year, 0) + int(count)
        )

    # Price range
    chunk_min_price = chunk["Modal_Price"].min()
    chunk_max_price = chunk["Modal_Price"].max()

    if min_price is None or chunk_min_price < min_price:
        min_price = chunk_min_price

    if max_price is None or chunk_max_price > max_price:
        max_price = chunk_max_price


# ============================================================
# FINAL REPORT
# ============================================================

print("\nTotal clean records:", total_rows)

print("\n--- GEOGRAPHICAL COVERAGE ---")

print("Number of states:", len(states))
print("Number of districts:", len(districts))
print("Number of markets:", len(markets))

print("\n--- COMMODITY COVERAGE ---")

print("Number of commodities:", len(commodities))

print("\n--- DATE COVERAGE ---")

print("Minimum date:", min_date)
print("Maximum date:", max_date)

print("\n--- RECORDS BY YEAR ---")

for year in sorted(year_counts):

    print(
        year,
        ":",
        f"{year_counts[year]:,}"
    )

print("\n--- MODAL PRICE RANGE ---")

print("Minimum Modal Price:", min_price)
print("Maximum Modal Price:", max_price)

print("\n" + "=" * 70)
print("VALIDATION COMPLETED")
print("=" * 70)