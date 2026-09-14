import pandas as pd
import numpy as np
import os
import pickle
import time


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\MAHEERA\PROJECTS\SIH-AGRILINK\ml"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "featured_market_prices.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "historical_monthly_features.csv"
)

CATEGORY_MAP_FILE = os.path.join(
    MODEL_DIR,
    "category_maps.pkl"
)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 500000


# ============================================================
# LOAD CATEGORY MAPS
# ============================================================

print("============================================")
print("LOADING CATEGORY MAPS")
print("============================================")

with open(
    CATEGORY_MAP_FILE,
    "rb"
) as file:

    category_maps = pickle.load(file)


# Handle possible wrapper structure
if "maps" in category_maps:
    category_maps = category_maps["maps"]


print(
    "Available maps:",
    list(category_maps.keys())
)


# ============================================================
# CHECK REQUIRED MAPS
# ============================================================

if "Market" not in category_maps:

    raise ValueError(
        "Market category map not found."
    )

if "State" not in category_maps:

    raise ValueError(
        "State category map not found."
    )


market_map = category_maps["Market"]

state_map = category_maps["State"]


# ============================================================
# STORAGE FOR PARTIAL AGGREGATES
# ============================================================

market_commodity_parts = []

state_commodity_parts = []

commodity_parts = []

market_parts = []


# ============================================================
# READ DATA IN CHUNKS
# ============================================================

print("\n============================================")
print("BUILDING MONTHLY HISTORICAL DATA")
print("============================================")

start_time = time.time()

chunk_number = 0


for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1

    print(
        f"\nProcessing chunk {chunk_number}..."
    )

    # --------------------------------------------------------
    # Convert market/state names to IDs
    # --------------------------------------------------------

    chunk["Market_ID"] = (
        chunk["Market"]
        .map(market_map)
        .fillna(-1)
        .astype("int32")
    )

    chunk["State_ID"] = (
        chunk["State"]
        .map(state_map)
        .fillna(-1)
        .astype("int32")
    )


    # ========================================================
    # 1. MARKET + COMMODITY
    # ========================================================

    mc = (
        chunk
        .groupby(
            [
                "Market_ID",
                "Commodity_Code",
                "Year",
                "Month"
            ],
            as_index=False
        )
        .agg(
            Price_Sum=(
                "Modal_Price",
                "sum"
            ),

            Price_Count=(
                "Modal_Price",
                "count"
            )
        )
    )

    market_commodity_parts.append(mc)


    # ========================================================
    # 2. STATE + COMMODITY
    # ========================================================

    sc = (
        chunk
        .groupby(
            [
                "State_ID",
                "Commodity_Code",
                "Year",
                "Month"
            ],
            as_index=False
        )
        .agg(
            Price_Sum=(
                "Modal_Price",
                "sum"
            ),

            Price_Count=(
                "Modal_Price",
                "count"
            )
        )
    )

    state_commodity_parts.append(sc)


    # ========================================================
    # 3. COMMODITY OVERALL
    # ========================================================

    c = (
        chunk
        .groupby(
            [
                "Commodity_Code",
                "Year",
                "Month"
            ],
            as_index=False
        )
        .agg(
            Price_Sum=(
                "Modal_Price",
                "sum"
            ),

            Price_Count=(
                "Modal_Price",
                "count"
            )
        )
    )

    commodity_parts.append(c)


    # ========================================================
    # 4. MARKET OVERALL
    # ========================================================

    m = (
        chunk
        .groupby(
            [
                "Market_ID",
                "Year",
                "Month"
            ],
            as_index=False
        )
        .agg(
            Price_Sum=(
                "Modal_Price",
                "sum"
            ),

            Price_Count=(
                "Modal_Price",
                "count"
            )
        )
    )

    market_parts.append(m)


# ============================================================
# COMBINE PARTIAL RESULTS
# ============================================================

print("\n============================================")
print("COMBINING MONTHLY AGGREGATES")
print("============================================")


def combine_parts(parts):

    combined = pd.concat(
        parts,
        ignore_index=True
    )

    group_columns = [
        column
        for column in combined.columns
        if column not in [
            "Price_Sum",
            "Price_Count"
        ]
    ]

    combined = (
        combined
        .groupby(
            group_columns,
            as_index=False
        )
        .agg(
            Price_Sum=(
                "Price_Sum",
                "sum"
            ),

            Price_Count=(
                "Price_Count",
                "sum"
            )
        )
    )

    combined["Monthly_Avg"] = (
        combined["Price_Sum"]
        /
        combined["Price_Count"]
    )

    combined.drop(
        columns=[
            "Price_Sum",
            "Price_Count"
        ],
        inplace=True
    )

    return combined


print("Market + Commodity...")

market_commodity = combine_parts(
    market_commodity_parts
)


print("State + Commodity...")

state_commodity = combine_parts(
    state_commodity_parts
)


print("Commodity...")

commodity = combine_parts(
    commodity_parts
)


print("Market...")

market = combine_parts(
    market_parts
)


# ============================================================
# MONTH INDEX
# ============================================================

def add_month_index(df):

    df["Month_Index"] = (
        df["Year"] * 12
        +
        df["Month"]
    )

    return df


market_commodity = add_month_index(
    market_commodity
)

state_commodity = add_month_index(
    state_commodity
)

commodity = add_month_index(
    commodity
)

market = add_month_index(
    market
)


# ============================================================
# PREVIOUS MONTH FEATURE
# ============================================================

def create_previous_month_feature(
    df,
    group_columns,
    feature_name
):

    df = df.sort_values(
        group_columns +
        ["Month_Index"]
    ).copy()


    df["Previous_Index"] = (
        df
        .groupby(group_columns)["Month_Index"]
        .shift(1)
    )


    df[feature_name] = (
        df
        .groupby(group_columns)["Monthly_Avg"]
        .shift(1)
    )


    # Make sure previous record
    # is actually previous month

    invalid = (
        df["Month_Index"]
        -
        df["Previous_Index"]
        != 1
    )

    df.loc[
        invalid,
        feature_name
    ] = np.nan


    return df


# ============================================================
# MARKET + COMMODITY FEATURES
# ============================================================

print("\nCreating market-commodity history...")


market_commodity = create_previous_month_feature(

    market_commodity,

    [
        "Market_ID",
        "Commodity_Code"
    ],

    "Prev_Month_Market_Commodity_Avg"

)


# ============================================================
# PREVIOUS YEAR SAME MONTH
# ============================================================

print(
    "Creating previous-year same-month feature..."
)


lookup = market_commodity[
    [
        "Market_ID",
        "Commodity_Code",
        "Month_Index",
        "Monthly_Avg"
    ]
].copy()


lookup.rename(
    columns={
        "Month_Index":
            "Previous_Year_Index",

        "Monthly_Avg":
            "Prev_Year_Same_Month_Avg"
    },
    inplace=True
)


market_commodity["Previous_Year_Index"] = (
    market_commodity["Month_Index"]
    -
    12
)


market_commodity = market_commodity.merge(

    lookup,

    on=[
        "Market_ID",
        "Commodity_Code",
        "Previous_Year_Index"
    ],

    how="left"

)


# ============================================================
# PREVIOUS 3 AVAILABLE MONTHS
# ============================================================

market_commodity[
    "Prev_3_Available_Months_Avg"
] = (

    market_commodity
    .groupby(
        [
            "Market_ID",
            "Commodity_Code"
        ]
    )["Monthly_Avg"]
    .shift(1)
    .rolling(
        3,
        min_periods=1
    )
    .mean()
    .reset_index(
        level=[
            0,
            1
        ],
        drop=True
    )

)


# ============================================================
# STATE + COMMODITY
# ============================================================

print(
    "Creating state-commodity history..."
)


state_commodity = create_previous_month_feature(

    state_commodity,

    [
        "State_ID",
        "Commodity_Code"
    ],

    "Prev_Month_State_Commodity_Avg"

)


# ============================================================
# COMMODITY
# ============================================================

print(
    "Creating commodity history..."
)


commodity = create_previous_month_feature(

    commodity,

    [
        "Commodity_Code"
    ],

    "Prev_Month_Commodity_Avg"

)


# ============================================================
# MARKET
# ============================================================

print(
    "Creating market history..."
)


market = create_previous_month_feature(

    market,

    [
        "Market_ID"
    ],

    "Prev_Month_Market_Avg"

)


# ============================================================
# SELECT COLUMNS
# ============================================================

market_commodity_features = market_commodity[
    [
        "Market_ID",
        "Commodity_Code",
        "Year",
        "Month",
        "Prev_Month_Market_Commodity_Avg",
        "Prev_Year_Same_Month_Avg",
        "Prev_3_Available_Months_Avg"
    ]
].copy()


state_commodity_features = state_commodity[
    [
        "State_ID",
        "Commodity_Code",
        "Year",
        "Month",
        "Prev_Month_State_Commodity_Avg"
    ]
].copy()


commodity_features = commodity[
    [
        "Commodity_Code",
        "Year",
        "Month",
        "Prev_Month_Commodity_Avg"
    ]
].copy()


market_features = market[
    [
        "Market_ID",
        "Year",
        "Month",
        "Prev_Month_Market_Avg"
    ]
].copy()


# ============================================================
# MERGE ALL FEATURES
# ============================================================

print("\n============================================")
print("MERGING HISTORICAL FEATURES")
print("============================================")


historical = market_commodity_features.merge(

    state_commodity_features,

    on=[
        "State_ID",
        "Commodity_Code",
        "Year",
        "Month"
    ],

    how="outer"

)


historical = historical.merge(

    commodity_features,

    on=[
        "Commodity_Code",
        "Year",
        "Month"
    ],

    how="outer"

)


historical = historical.merge(

    market_features,

    on=[
        "Market_ID",
        "Year",
        "Month"
    ],

    how="outer"

)


# ============================================================
# SORT
# ============================================================

historical.sort_values(
    [
        "Year",
        "Month"
    ],
    inplace=True
)


# ============================================================
# SAVE
# ============================================================

print("\n============================================")
print("SAVING HISTORICAL FEATURES")
print("============================================")


historical.to_csv(
    OUTPUT_FILE,
    index=False
)


elapsed = (
    time.time()
    -
    start_time
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print("HISTORICAL FEATURE CREATION COMPLETE")
print("============================================")

print(
    "Rows:",
    len(historical)
)

print(
    "Columns:",
    len(historical.columns)
)

print(
    "File:"
)

print(
    OUTPUT_FILE
)

print(
    f"\nTotal time: "
    f"{elapsed / 60:.2f} minutes"
)

print("\nFeatures created:")

print(
    "1. Previous month market-commodity average"
)

print(
    "2. Previous year same-month average"
)

print(
    "3. Previous 3 available months average"
)

print(
    "4. Previous month state-commodity average"
)

print(
    "5. Previous month commodity average"
)

print(
    "6. Previous month market average"
)

print("\nDone.")