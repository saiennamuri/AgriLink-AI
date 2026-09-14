import pandas as pd
import numpy as np

# ==========================================
# 1. Load Dataset
# ==========================================

file_path = "data/market_prices.csv"

df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)


# ==========================================
# 2. Rename Price Columns
# ==========================================

df = df.rename(columns={
    "Min_x0020_Price": "Min_Price",
    "Max_x0020_Price": "Max_Price",
    "Modal_x0020_Price": "Modal_Price"
})


# ==========================================
# 3. Remove Duplicate Rows
# ==========================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("\nDuplicates removed:", before - after)


# ==========================================
# 4. Convert Date Column
# ==========================================

df["Arrival_Date"] = pd.to_datetime(
    df["Arrival_Date"],
    errors="coerce"
)

print("\nDate information:")
print("Minimum date:", df["Arrival_Date"].min())
print("Maximum date:", df["Arrival_Date"].max())
print("Unique dates:", df["Arrival_Date"].nunique())


# ==========================================
# 5. Convert Price Columns to Numeric
# ==========================================

price_columns = [
    "Min_Price",
    "Max_Price",
    "Modal_Price"
]

for column in price_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ==========================================
# 6. Check Missing Values
# ==========================================

print("\nMissing values before cleaning:")

print(df.isnull().sum())


# ==========================================
# 7. Remove Rows With Missing Values
# ==========================================

df = df.dropna()

print("\nShape after removing missing values:")
print(df.shape)


# ==========================================
# 8. Remove Invalid Prices
# ==========================================

df = df[
    (df["Min_Price"] > 0) &
    (df["Max_Price"] > 0) &
    (df["Modal_Price"] > 0)
]


# ==========================================
# 9. Check Price Relationships
# ==========================================

# Normally:
# Minimum Price <= Modal Price <= Maximum Price

invalid_price_rows = df[
    (df["Min_Price"] > df["Modal_Price"]) |
    (df["Modal_Price"] > df["Max_Price"])
]

print("\nInvalid price relationship rows:",
      len(invalid_price_rows))

# Remove invalid price relationships
df = df[
    (df["Min_Price"] <= df["Modal_Price"]) &
    (df["Modal_Price"] <= df["Max_Price"])
]


# ==========================================
# 10. Clean Text Columns
# ==========================================

text_columns = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade"
]

for column in text_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
    )


# ==========================================
# 11. Create Date Features
# ==========================================

df["Year"] = df["Arrival_Date"].dt.year
df["Month"] = df["Arrival_Date"].dt.month
df["Day"] = df["Arrival_Date"].dt.day


# ==========================================
# 12. Final Dataset Information
# ==========================================

print("\n========== CLEAN DATASET ==========")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nStates:", df["State"].nunique())
print("Districts:", df["District"].nunique())
print("Markets:", df["Market"].nunique())
print("Commodities:", df["Commodity"].nunique())
print("Varieties:", df["Variety"].nunique())
print("Grades:", df["Grade"].nunique())

print("\nDate range:")
print(df["Arrival_Date"].min())
print(df["Arrival_Date"].max())


# ==========================================
# 13. Price Statistics
# ==========================================

print("\n========== PRICE STATISTICS ==========")

print(df[
    ["Min_Price", "Max_Price", "Modal_Price"]
].describe())


# ==========================================
# 14. Save Clean Dataset
# ==========================================

output_file = "data/clean_market_prices.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nClean dataset saved to:")
print(output_file)

print("\n========== CLEANING COMPLETED ==========")