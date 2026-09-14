import pandas as pd
import os

# --------------------------------------------------
# File paths
# --------------------------------------------------

INPUT_FILE = r"E:\MAHEERA\PROJECTS\SIH-AGRILINK\ml\data\ml_ready_prices.csv"

OUTPUT_FILE = r"E:\MAHEERA\PROJECTS\SIH-AGRILINK\ml\data\featured_market_prices.csv"


# --------------------------------------------------
# Settings
# --------------------------------------------------

CHUNK_SIZE = 500000


# --------------------------------------------------
# Process dataset in chunks
# --------------------------------------------------

print("========================================")
print("FEATURE ENGINEERING")
print("========================================")

print("\nInput file:")
print(INPUT_FILE)

print("\nProcessing dataset in chunks...")


first_chunk = True
total_rows = 0
total_invalid_dates = 0


for chunk_number, df in enumerate(
    pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE),
    start=1
):

    print(f"\nProcessing chunk {chunk_number}...")
    print("Rows:", len(df))


    # --------------------------------------------------
    # Convert date
    # --------------------------------------------------

    df["Arrival_Date"] = pd.to_datetime(
        df["Arrival_Date"],
        errors="coerce"
    )


    # Count invalid dates
    invalid_dates = df["Arrival_Date"].isna().sum()

    total_invalid_dates += invalid_dates


    # Remove invalid dates
    df = df.dropna(subset=["Arrival_Date"])


    # --------------------------------------------------
    # Date features
    # --------------------------------------------------

    df["Year"] = df["Arrival_Date"].dt.year.astype("int16")

    df["Month"] = df["Arrival_Date"].dt.month.astype("int8")

    df["Day"] = df["Arrival_Date"].dt.day.astype("int8")

    df["DayOfWeek"] = df["Arrival_Date"].dt.dayofweek.astype("int8")

    df["DayOfYear"] = df["Arrival_Date"].dt.dayofyear.astype("int16")

    df["Quarter"] = df["Arrival_Date"].dt.quarter.astype("int8")


    # --------------------------------------------------
    # Season
    # --------------------------------------------------

    df["Season"] = df["Month"].map({
        12: "Winter",
        1: "Winter",
        2: "Winter",

        3: "Summer",
        4: "Summer",
        5: "Summer",

        6: "Monsoon",
        7: "Monsoon",
        8: "Monsoon",
        9: "Monsoon",

        10: "Post-Monsoon",
        11: "Post-Monsoon"
    })


    # --------------------------------------------------
    # Month-Year feature
    # --------------------------------------------------

    df["YearMonth"] = (
        df["Year"].astype(str)
        + "-"
        + df["Month"].astype(str).str.zfill(2)
    )


    # --------------------------------------------------
    # Write processed chunk
    # --------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )


    first_chunk = False

    total_rows += len(df)

    print("Processed rows so far:", total_rows)


# --------------------------------------------------
# Final information
# --------------------------------------------------

print("\n========================================")
print("FEATURE ENGINEERING COMPLETED")
print("========================================")

print("Final rows:", total_rows)

print("Invalid dates removed:", total_invalid_dates)

print("\nNew features created:")

print("""
1. Year
2. Month
3. Day
4. DayOfWeek
5. DayOfYear
6. Quarter
7. Season
8. YearMonth
""")

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nDone.")