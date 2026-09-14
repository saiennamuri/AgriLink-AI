import pandas as pd
import os


# ============================================================
# AGRILINK AI - COMMODITY-WISE PRICE ANALYSIS
# M2: AI PRICE PREDICTION
# ============================================================

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

CLEAN_FILE = os.path.join(
    BASE_FOLDER,
    "clean_market_prices.csv"
)

print("=" * 70)
print("AGRILINK AI - COMMODITY-WISE PRICE ANALYSIS")
print("=" * 70)

print("\nReading dataset...")


# ------------------------------------------------------------
# Store price information by commodity
# ------------------------------------------------------------

commodity_prices = {}

extreme_records = []


# ------------------------------------------------------------
# Read dataset in chunks
# ------------------------------------------------------------

for chunk in pd.read_csv(
    CLEAN_FILE,
    usecols=[
        "State",
        "District",
        "Market",
        "Commodity",
        "Variety",
        "Arrival_Date",
        "Min_Price",
        "Max_Price",
        "Modal_Price"
    ],
    chunksize=200000,
    low_memory=False
):

    # Convert Modal_Price to numeric
    chunk["Modal_Price"] = pd.to_numeric(
        chunk["Modal_Price"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Store commodity prices
    # --------------------------------------------------------

    grouped = chunk.groupby("Commodity")["Modal_Price"]

    for commodity, prices in grouped:

        if commodity not in commodity_prices:
            commodity_prices[commodity] = []

        commodity_prices[commodity].extend(
            prices.dropna().tolist()
        )

    # --------------------------------------------------------
    # Collect extremely high records
    # --------------------------------------------------------

    high = chunk[
        chunk["Modal_Price"] > 100000
    ]

    if len(high) > 0:

        extreme_records.append(high)


# ============================================================
# COMMODITY STATISTICS
# ============================================================

results = []

for commodity, prices in commodity_prices.items():

    series = pd.Series(prices)

    results.append({
        "Commodity": commodity,
        "Records": len(series),
        "Minimum": series.min(),
        "Median": series.median(),
        "Mean": series.mean(),
        "95th_Percentile": series.quantile(0.95),
        "99th_Percentile": series.quantile(0.99),
        "99.9th_Percentile": series.quantile(0.999),
        "Maximum": series.max()
    })


stats = pd.DataFrame(results)

stats = stats.sort_values(
    "Maximum",
    ascending=False
)


# ============================================================
# DISPLAY TOP COMMODITIES
# ============================================================

print("\n")
print("=" * 70)
print("TOP 30 COMMODITIES BY MAXIMUM MODAL PRICE")
print("=" * 70)

print(
    stats.head(30).to_string(
        index=False
    )
)


# ============================================================
# EXTREME RECORDS
# ============================================================

print("\n")
print("=" * 70)
print("TOP 50 EXTREME PRICE RECORDS")
print("=" * 70)

if extreme_records:

    extreme_df = pd.concat(
        extreme_records,
        ignore_index=True
    )

    extreme_df = extreme_df.sort_values(
        "Modal_Price",
        ascending=False
    )

    print(
        extreme_df[
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
        ].head(50).to_string(
            index=False
        )
    )


# ============================================================
# SAVE COMMODITY STATISTICS
# ============================================================

output_file = os.path.join(
    BASE_FOLDER,
    "commodity_price_statistics.csv"
)

stats.to_csv(
    output_file,
    index=False
)

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print("\nStatistics saved to:")
print(output_file)