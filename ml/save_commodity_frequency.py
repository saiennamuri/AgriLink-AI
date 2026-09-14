import pandas as pd
import os
import pickle


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\MAHEERA\PROJECTS\SIH-AGRILINK\ml"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "featured_market_prices.csv"
)

PREPROCESSING_FILE = os.path.join(
    BASE_DIR,
    "models",
    "final_preprocessing.pkl"
)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 500000
RANDOM_STATE = 42

TRAIN_SAMPLE_SIZE = 1500000


# ============================================================
# LOAD TRAINING DATA EXACTLY AS TRAINING SCRIPT DID
# ============================================================

print("============================================")
print("SAVING COMMODITY CODE FREQUENCY")
print("============================================")

print("\nReading dataset...")

train_parts = []

chunk_number = 0


for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1

    print(
        f"Processing chunk {chunk_number}..."
    )

    # Same target filtering
    chunk = chunk[
        chunk["Modal_Price"] > 0
    ].copy()

    # Same training-year selection
    train_chunk = chunk[
        chunk["Year"] <= 2024
    ]

    # EXACT same sampling used during training
    if len(train_chunk) > 0:

        train_parts.append(
            train_chunk.sample(
                frac=0.15,
                random_state=RANDOM_STATE
            )
        )


# ============================================================
# COMBINE
# ============================================================

train_df = pd.concat(
    train_parts,
    ignore_index=True
)


# ============================================================
# EXACT SAME FINAL SAMPLE LIMIT
# ============================================================

if len(train_df) > TRAIN_SAMPLE_SIZE:

    train_df = train_df.sample(
        n=TRAIN_SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )


print(
    "\nRecreated training sample:",
    len(train_df)
)


# ============================================================
# COMMODITY CODE FREQUENCY
# ============================================================

commodity_code_frequency = (
    train_df["Commodity_Code"]
    .value_counts()
    /
    len(train_df)
).to_dict()


print(
    "Unique Commodity Codes:",
    len(commodity_code_frequency)
)


# ============================================================
# LOAD EXISTING PREPROCESSING
# ============================================================

print("\nLoading existing preprocessing...")

with open(
    PREPROCESSING_FILE,
    "rb"
) as file:

    preprocessing = pickle.load(file)


# ============================================================
# ADD COMMODITY CODE FREQUENCY
# ============================================================

preprocessing[
    "commodity_code_frequency"
] = commodity_code_frequency


# ============================================================
# SAVE UPDATED PREPROCESSING
# ============================================================

with open(
    PREPROCESSING_FILE,
    "wb"
) as file:

    pickle.dump(
        preprocessing,
        file
    )


print("\n============================================")
print("SUCCESS")
print("============================================")

print(
    "\nSaved:",
    PREPROCESSING_FILE
)

print(
    "Commodity code frequency entries:",
    len(
        preprocessing[
            "commodity_code_frequency"
        ]
    )
)

print(
    "\nModel was NOT retrained."
)