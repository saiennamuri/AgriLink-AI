import pandas as pd
import os

INPUT_FILE = "data/featured_market_prices.csv"
OUTPUT_FILE = "models/commodity_code_map.pkl"

print("Reading commodity data...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=["Commodity", "Commodity_Code"]
)

# Remove duplicate commodity-code combinations
df = df.drop_duplicates()

# Check whether a commodity has multiple codes
multiple_codes = (
    df.groupby("Commodity")["Commodity_Code"]
    .nunique()
)

problematic = multiple_codes[multiple_codes > 1]

if len(problematic) > 0:
    print("\nWARNING: Some commodities have multiple codes:")
    print(problematic)
else:
    print("All commodities have a single Commodity_Code.")

# Create mapping
commodity_code_map = (
    df.drop_duplicates("Commodity")
      .set_index("Commodity")["Commodity_Code"]
      .to_dict()
)

# Save mapping
import pickle

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(commodity_code_map, f)

print("\n==========================================")
print("Commodity code map created successfully")
print("==========================================")

print("Total commodities:", len(commodity_code_map))
print("Saved to:", OUTPUT_FILE)

print("\nSample:")
for commodity, code in list(commodity_code_map.items())[:20]:
    print(f"{commodity} -> {code}")