import pandas as pd
import numpy as np
import os
import pickle

# ==================================================
# FILE PATHS
# ==================================================

INPUT_FILE = r"E:\MAHEERA\PROJECTS\SIH-AGRILINK\ml\data\featured_market_prices.csv"

OUTPUT_DIR = r"E:\MAHEERA\PROJECTS\SIH-AGRILINK\ml\data\model_data"

ENCODER_FILE = r"E:\MAHEERA\PROJECTS\SIH-AGRILINK\ml\models\category_maps.pkl"


# ==================================================
# SETTINGS
# ==================================================

CHUNK_SIZE = 500000

# Number of rows used for model development
TRAIN_SAMPLE_SIZE = 1000000
VALIDATION_SAMPLE_SIZE = 300000
TEST_SAMPLE_SIZE = 100000


# ==================================================
# FEATURES
# ==================================================

CATEGORICAL_FEATURES = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade",
    "Season"
]

NUMERICAL_FEATURES = [
    "Commodity_Code",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "DayOfYear",
    "Quarter"
]

TARGET = "Modal_Price"


# ==================================================
# CREATE OUTPUT FOLDER
# ==================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(
    os.path.dirname(ENCODER_FILE),
    exist_ok=True
)


# ==================================================
# STEP 1 — FIND UNIQUE CATEGORIES
# ==================================================

print("========================================")
print("STEP 1: LEARNING CATEGORY MAPS")
print("========================================")

category_maps = {
    column: {}
    for column in CATEGORICAL_FEATURES
}


chunk_number = 0

for df in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1

    print(f"Reading training chunk {chunk_number}...")

    # Convert date
    df["Arrival_Date"] = pd.to_datetime(
        df["Arrival_Date"],
        errors="coerce"
    )

    # Only training period
    df = df[
        df["Arrival_Date"].dt.year <= 2024
    ]

    if len(df) == 0:
        continue

    # Collect categories
    for column in CATEGORICAL_FEATURES:

        values = df[column].fillna("Unknown").astype(str)

        unique_values = values.unique()

        for value in unique_values:

            if value not in category_maps[column]:

                category_maps[column][value] = (
                    len(category_maps[column])
                )


# ==================================================
# DISPLAY CATEGORY COUNTS
# ==================================================

print("\n========================================")
print("CATEGORY COUNTS")
print("========================================")

for column in CATEGORICAL_FEATURES:

    print(
        f"{column}: "
        f"{len(category_maps[column]):,}"
    )


# ==================================================
# SAVE CATEGORY MAPS
# ==================================================

with open(
    ENCODER_FILE,
    "wb"
) as file:

    pickle.dump(
        category_maps,
        file
    )


print("\nCategory maps saved:")
print(ENCODER_FILE)


# ==================================================
# STEP 2 — RANDOM SAMPLING
# ==================================================

print("\n========================================")
print("STEP 2: CREATING MODEL SAMPLES")
print("========================================")

print(
    f"Training sample:   {TRAIN_SAMPLE_SIZE:,}"
)

print(
    f"Validation sample: {VALIDATION_SAMPLE_SIZE:,}"
)

print(
    f"Testing sample:    {TEST_SAMPLE_SIZE:,}"
)


train_parts = []
validation_parts = []
test_parts = []


# ==================================================
# READ DATA AGAIN
# ==================================================

chunk_number = 0

for df in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1

    print(
        f"\nProcessing sampling chunk "
        f"{chunk_number}..."
    )

    # --------------------------------------------------
    # Convert date
    # --------------------------------------------------

    df["Arrival_Date"] = pd.to_datetime(
        df["Arrival_Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Arrival_Date", TARGET]
    )


    # --------------------------------------------------
    # Determine split
    # --------------------------------------------------

    year = df["Arrival_Date"].dt.year

    train_df = df[year <= 2024].copy()

    validation_df = df[year == 2025].copy()

    test_df = df[year >= 2026].copy()


    # ==================================================
    # FUNCTION TO ENCODE DATA
    # ==================================================

    def encode_dataframe(data):

        if len(data) == 0:
            return data

        # ----------------------------------------------
        # Encode categorical features
        # ----------------------------------------------

        for column in CATEGORICAL_FEATURES:

            data[column] = (
                data[column]
                .fillna("Unknown")
                .astype(str)
                .map(category_maps[column])
                .fillna(-1)
                .astype("int32")
            )


        # ----------------------------------------------
        # Numerical features
        # ----------------------------------------------

        selected_columns = (
            CATEGORICAL_FEATURES
            + NUMERICAL_FEATURES
            + [TARGET]
        )

        data = data[selected_columns]

        return data


    # ==================================================
    # ENCODE
    # ==================================================

    train_df = encode_dataframe(train_df)

    validation_df = encode_dataframe(validation_df)

    test_df = encode_dataframe(test_df)


    # ==================================================
    # SAMPLE EACH CHUNK
    # ==================================================

    if len(train_df) > 0:

        sample_fraction = min(
            1.0,
            TRAIN_SAMPLE_SIZE /
            13187810
        )

        sampled = train_df.sample(
            frac=sample_fraction,
            random_state=42
        )

        train_parts.append(sampled)


    if len(validation_df) > 0:

        sample_fraction = min(
            1.0,
            VALIDATION_SAMPLE_SIZE /
            4130618
        )

        sampled = validation_df.sample(
            frac=sample_fraction,
            random_state=42
        )

        validation_parts.append(sampled)


    if len(test_df) > 0:

        sample_fraction = min(
            1.0,
            TEST_SAMPLE_SIZE /
            563466
        )

        sampled = test_df.sample(
            frac=sample_fraction,
            random_state=42
        )

        test_parts.append(sampled)


# ==================================================
# COMBINE SAMPLES
# ==================================================

print("\nCombining samples...")


train_sample = pd.concat(
    train_parts,
    ignore_index=True
)

validation_sample = pd.concat(
    validation_parts,
    ignore_index=True
)

test_sample = pd.concat(
    test_parts,
    ignore_index=True
)


# ==================================================
# LIMIT TO TARGET SIZE
# ==================================================

train_sample = train_sample.sample(
    n=min(
        TRAIN_SAMPLE_SIZE,
        len(train_sample)
    ),
    random_state=42
)

validation_sample = validation_sample.sample(
    n=min(
        VALIDATION_SAMPLE_SIZE,
        len(validation_sample)
    ),
    random_state=42
)

test_sample = test_sample.sample(
    n=min(
        TEST_SAMPLE_SIZE,
        len(test_sample)
    ),
    random_state=42
)


# ==================================================
# SAVE
# ==================================================

train_file = os.path.join(
    OUTPUT_DIR,
    "train_sample.csv"
)

validation_file = os.path.join(
    OUTPUT_DIR,
    "validation_sample.csv"
)

test_file = os.path.join(
    OUTPUT_DIR,
    "test_sample.csv"
)


train_sample.to_csv(
    train_file,
    index=False
)

validation_sample.to_csv(
    validation_file,
    index=False
)

test_sample.to_csv(
    test_file,
    index=False
)


# ==================================================
# FINAL INFORMATION
# ==================================================

print("\n========================================")
print("PREPROCESSING COMPLETED")
print("========================================")

print(
    f"\nTraining sample: "
    f"{len(train_sample):,}"
)

print(
    f"Validation sample: "
    f"{len(validation_sample):,}"
)

print(
    f"Testing sample: "
    f"{len(test_sample):,}"
)

print("\nFiles created:")

print(train_file)

print(validation_file)

print(test_file)

print("\nCategory maps:")

print(ENCODER_FILE)

print("\nDone.")