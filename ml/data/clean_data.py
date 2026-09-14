import pandas as pd
import os
import glob


# ============================================================
# AGRILINK AI - DATA CLEANING
# M2: AI PRICE PREDICTION
# ============================================================

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FILE = os.path.join(
    BASE_FOLDER,
    "clean_market_prices.csv"
)

files = sorted(
    glob.glob(os.path.join(BASE_FOLDER, "*.csv"))
)

# Do not include the output file if it already exists
files = [
    file for file in files
    if os.path.basename(file) != "clean_market_prices.csv"
]

if not files:
    print("No input CSV files found!")
    exit()


# Expected columns
required_columns = [
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
# Statistics
# ------------------------------------------------------------

total_rows = 0
total_duplicates = 0
total_missing = 0
total_invalid_prices = 0
total_invalid_ranges = 0
total_invalid_dates = 0
total_clean_rows = 0

first_write = True


print("=" * 70)
print("AGRILINK AI - DATA CLEANING")
print("=" * 70)

print("\nInput files:")

for file in files:
    print(" -", os.path.basename(file))


# ------------------------------------------------------------
# Process each CSV
# ------------------------------------------------------------

for file in files:

    print("\n" + "-" * 70)
    print("Processing:", os.path.basename(file))
    print("-" * 70)

    # Read large CSV in chunks
    for chunk in pd.read_csv(
        file,
        chunksize=200000,
        low_memory=False
    ):

        total_rows += len(chunk)

        # ----------------------------------------------------
        # Check columns
        # ----------------------------------------------------

        missing_columns = [
            col for col in required_columns
            if col not in chunk.columns
        ]

        if missing_columns:
            print(
                "ERROR: Missing columns:",
                missing_columns
            )
            exit()

        # Keep only required columns
        chunk = chunk[required_columns].copy()

        # ----------------------------------------------------
        # Clean text columns
        # ----------------------------------------------------

        text_columns = [
            "State",
            "District",
            "Market",
            "Commodity",
            "Variety",
            "Grade"
        ]

        for col in text_columns:

            chunk[col] = (
                chunk[col]
                .astype("string")
                .str.strip()
            )

            # Convert empty strings to missing values
            chunk[col] = chunk[col].replace(
                "",
                pd.NA
            )

        # ----------------------------------------------------
        # Convert numeric columns
        # ----------------------------------------------------

        numeric_columns = [
            "Min_Price",
            "Max_Price",
            "Modal_Price",
            "Commodity_Code"
        ]

        for col in numeric_columns:

            chunk[col] = pd.to_numeric(
                chunk[col],
                errors="coerce"
            )

        # ----------------------------------------------------
        # Convert date
        # ----------------------------------------------------

        chunk["Arrival_Date"] = pd.to_datetime(
            chunk["Arrival_Date"],
            errors="coerce",
            dayfirst=True
        )

        invalid_dates = chunk["Arrival_Date"].isna().sum()

        total_invalid_dates += invalid_dates

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        essential_columns = [
            "State",
            "District",
            "Market",
            "Commodity",
            "Arrival_Date",
            "Min_Price",
            "Max_Price",
            "Modal_Price"
        ]

        missing_rows = chunk[essential_columns].isna().any(axis=1)

        total_missing += missing_rows.sum()

        chunk = chunk[~missing_rows].copy()

        # ----------------------------------------------------
        # Remove non-positive prices
        # ----------------------------------------------------

        invalid_prices = (
            (chunk["Min_Price"] <= 0) |
            (chunk["Max_Price"] <= 0) |
            (chunk["Modal_Price"] <= 0)
        )

        total_invalid_prices += invalid_prices.sum()

        chunk = chunk[~invalid_prices].copy()

        # ----------------------------------------------------
        # Check price relationships
        # ----------------------------------------------------

        invalid_ranges = (
            (chunk["Min_Price"] > chunk["Max_Price"]) |
            (chunk["Modal_Price"] < chunk["Min_Price"]) |
            (chunk["Modal_Price"] > chunk["Max_Price"])
        )

        total_invalid_ranges += invalid_ranges.sum()

        chunk = chunk[~invalid_ranges].copy()

        # ----------------------------------------------------
        # Remove exact duplicate records
        # ----------------------------------------------------

        duplicates = chunk.duplicated().sum()

        total_duplicates += duplicates

        chunk = chunk.drop_duplicates()

        # ----------------------------------------------------
        # Convert date to standard format
        # ----------------------------------------------------

        chunk["Arrival_Date"] = (
            chunk["Arrival_Date"]
            .dt.strftime("%Y-%m-%d")
        )

        # ----------------------------------------------------
        # Write cleaned data
        # ----------------------------------------------------

        if len(chunk) > 0:

            chunk.to_csv(
                OUTPUT_FILE,
                mode="w" if first_write else "a",
                index=False,
                header=first_write
            )

            first_write = False

            total_clean_rows += len(chunk)

        print(
            "Processed:",
            total_rows,
            "rows | Clean:",
            total_clean_rows
        )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLEANING COMPLETED")
print("=" * 70)

print("\nOriginal rows:", total_rows)
print("Duplicate rows removed:", total_duplicates)
print("Rows with missing essential data:", total_missing)
print("Invalid dates:", total_invalid_dates)
print("Invalid/non-positive prices:", total_invalid_prices)
print("Invalid price ranges:", total_invalid_ranges)

print("\nFinal clean rows:", total_clean_rows)

print("\nClean dataset saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)