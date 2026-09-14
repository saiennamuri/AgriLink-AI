import pandas as pd

# --------------------------------------------------
# File path
# --------------------------------------------------

INPUT_FILE = r"E:\MAHEERA\PROJECTS\SIH-AGRILINK\ml\data\featured_market_prices.csv"

CHUNK_SIZE = 500000


# --------------------------------------------------
# Feature selection
# --------------------------------------------------

FEATURES = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade",
    "Commodity_Code",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "DayOfYear",
    "Quarter",
    "Season",
    "YearMonth"
]

TARGET = "Modal_Price"


# --------------------------------------------------
# Counters
# --------------------------------------------------

train_rows = 0
validation_rows = 0
test_rows = 0

invalid_rows = 0

print("========================================")
print("FEATURE SELECTION & TIME SPLIT")
print("========================================")

print("\nReading dataset in chunks...")


# --------------------------------------------------
# Read dataset
# --------------------------------------------------

for chunk_number, df in enumerate(
    pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE),
    start=1
):

    print(f"\nProcessing chunk {chunk_number}...")


    # --------------------------------------------------
    # Check required columns
    # --------------------------------------------------

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("Missing columns:", missing_columns)
        raise ValueError("Required columns are missing.")


    # --------------------------------------------------
    # Convert date
    # --------------------------------------------------

    df["Arrival_Date"] = pd.to_datetime(
        df["Arrival_Date"],
        errors="coerce"
    )


    # --------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------

    invalid = (
        df["Arrival_Date"].isna()
        | df[TARGET].isna()
    )

    invalid_rows += invalid.sum()

    df = df.loc[~invalid]


    # --------------------------------------------------
    # Get year
    # --------------------------------------------------

    year = df["Arrival_Date"].dt.year


    # --------------------------------------------------
    # Time-based split
    # --------------------------------------------------

    train = year <= 2024

    validation = year == 2025

    test = year >= 2026


    train_count = train.sum()
    validation_count = validation.sum()
    test_count = test.sum()


    train_rows += train_count
    validation_rows += validation_count
    test_rows += test_count


    print(
        f"Train: {train_count:,} | "
        f"Validation: {validation_count:,} | "
        f"Test: {test_count:,}"
    )


# --------------------------------------------------
# Final results
# --------------------------------------------------

print("\n========================================")
print("SPLIT COMPLETED")
print("========================================")

print(f"\nTraining rows:   {train_rows:,}")
print(f"Validation rows: {validation_rows:,}")
print(f"Testing rows:    {test_rows:,}")
print(f"Invalid rows:    {invalid_rows:,}")


# --------------------------------------------------
# Percentage calculation
# --------------------------------------------------

total = train_rows + validation_rows + test_rows

print("\nDataset distribution:")

print(
    f"Training:   {train_rows / total * 100:.2f}%"
)

print(
    f"Validation: {validation_rows / total * 100:.2f}%"
)

print(
    f"Testing:    {test_rows / total * 100:.2f}%"
)


# --------------------------------------------------
# Selected features
# --------------------------------------------------

print("\n========================================")
print("SELECTED FEATURES")
print("========================================")

for i, feature in enumerate(FEATURES, start=1):
    print(f"{i}. {feature}")


# --------------------------------------------------
# Target
# --------------------------------------------------

print("\nTarget variable:")
print(TARGET)


# --------------------------------------------------
# Removed leakage features
# --------------------------------------------------

print("\n========================================")
print("REMOVED FEATURES")
print("========================================")

print("Min_Price  -> Removed")
print("Max_Price  -> Removed")


print("\nReason:")
print(
    "Min_Price and Max_Price represent actual prices "
    "for the same market-day and can cause data leakage."
)

print("\nDone.")