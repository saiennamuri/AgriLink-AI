import pandas as pd
import os
import glob

# Current folder where this Python file is located
DATA_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Find all CSV files in the same folder
files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

if not files:
    print("No CSV files found!")
    exit()

print("=" * 60)
print("AGRILINK AI - DATASET INSPECTION")
print("=" * 60)

print(f"\nCSV files found: {len(files)}")

for file in sorted(files):

    print("\n" + "-" * 60)
    print("FILE:", os.path.basename(file))
    print("-" * 60)

    df_sample = pd.read_csv(
        file,
        nrows=5,
        low_memory=False
    )

    print("\nColumns:")
    print(list(df_sample.columns))

    print("\nSample data:")
    print(df_sample.head())

    print("\nData types:")
    print(df_sample.dtypes)

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)