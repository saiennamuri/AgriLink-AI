import pandas as pd
import numpy as np
import os
import pickle
import time

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS
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

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

CHUNK_SIZE = 500000

# Large but manageable representative samples
TRAIN_SAMPLE_SIZE = 1500000
VALIDATION_SAMPLE_SIZE = 400000
TEST_SAMPLE_SIZE = 150000

RANDOM_STATE = 42


# ============================================================
# 3. COLUMNS
# ============================================================

CATEGORICAL_COLUMNS = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade",
    "Season"
]

TARGET = "Modal_Price"


# ============================================================
# 4. STORAGE
# ============================================================

train_parts = []
validation_parts = []
test_parts = []

print("============================================")
print("FINAL AGRILINK MODEL")
print("============================================")

print("\nReading dataset...")


# ============================================================
# 5. READ DATA IN CHUNKS
# ============================================================

start_time = time.time()

chunk_number = 0


for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1

    print(
        f"Processing chunk {chunk_number}..."
    )

    # --------------------------------------------------------
    # Make sure target is valid
    # --------------------------------------------------------

    chunk = chunk[
        chunk[TARGET] > 0
    ].copy()


    # --------------------------------------------------------
    # Time-based split
    # --------------------------------------------------------

    train_chunk = chunk[
        chunk["Year"] <= 2024
    ]

    validation_chunk = chunk[
        chunk["Year"] == 2025
    ]

    test_chunk = chunk[
        chunk["Year"] == 2026
    ]


    # --------------------------------------------------------
    # Sample each chunk
    # --------------------------------------------------------

    if len(train_chunk) > 0:

        train_parts.append(
            train_chunk.sample(
                frac=0.15,
                random_state=RANDOM_STATE
            )
        )


    if len(validation_chunk) > 0:

        validation_parts.append(
            validation_chunk.sample(
                frac=0.15,
                random_state=RANDOM_STATE
            )
        )


    if len(test_chunk) > 0:

        test_parts.append(
            test_chunk.sample(
                frac=0.25,
                random_state=RANDOM_STATE
            )
        )


# ============================================================
# 6. COMBINE SAMPLES
# ============================================================

print("\n============================================")
print("COMBINING SAMPLES")
print("============================================")


train_df = pd.concat(
    train_parts,
    ignore_index=True
)

validation_df = pd.concat(
    validation_parts,
    ignore_index=True
)

test_df = pd.concat(
    test_parts,
    ignore_index=True
)


# ------------------------------------------------------------
# Limit to requested sample sizes
# ------------------------------------------------------------

if len(train_df) > TRAIN_SAMPLE_SIZE:

    train_df = train_df.sample(
        n=TRAIN_SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )


if len(validation_df) > VALIDATION_SAMPLE_SIZE:

    validation_df = validation_df.sample(
        n=VALIDATION_SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )


if len(test_df) > TEST_SAMPLE_SIZE:

    test_df = test_df.sample(
        n=TEST_SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )


print(
    "Training records   :",
    len(train_df)
)

print(
    "Validation records :",
    len(validation_df)
)

print(
    "Testing records    :",
    len(test_df)
)


# ============================================================
# 7. FREQUENCY ENCODING
# ============================================================

print("\n============================================")
print("CREATING LEAKAGE-SAFE FREQUENCY ENCODING")
print("============================================")


frequency_maps = {}


for column in CATEGORICAL_COLUMNS:

    print(
        "Encoding:",
        column
    )

    counts = (
        train_df[column]
        .value_counts()
    )

    total = len(train_df)

    frequency_maps[column] = (
        counts / total
    ).to_dict()


# ============================================================
# 8. FEATURE ENGINEERING FUNCTION
# ============================================================

def create_features(
    df
):

    result = pd.DataFrame(
        index=df.index
    )


    # --------------------------------------------------------
    # Frequency encoded categorical features
    # --------------------------------------------------------

    for column in CATEGORICAL_COLUMNS:

        result[
            column + "_Frequency"
        ] = (

            df[column]
            .map(
                frequency_maps[column]
            )
            .fillna(0)

        )


    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

    result["Year"] = (
        df["Year"]
        .astype("int16")
    )

    result["Month"] = (
        df["Month"]
        .astype("int8")
    )

    result["Day"] = (
        df["Day"]
        .astype("int8")
    )

    result["DayOfWeek"] = (
        df["DayOfWeek"]
        .astype("int8")
    )

    result["DayOfYear"] = (
        df["DayOfYear"]
        .astype("int16")
    )

    result["Quarter"] = (
        df["Quarter"]
        .astype("int8")
    )


    # --------------------------------------------------------
    # Cyclical month features
    # --------------------------------------------------------

    result["Month_Sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    result["Month_Cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )


    # --------------------------------------------------------
    # Cyclical day-of-year features
    # --------------------------------------------------------

    result["DayOfYear_Sin"] = np.sin(
        2 * np.pi * df["DayOfYear"] / 365
    )

    result["DayOfYear_Cos"] = np.cos(
        2 * np.pi * df["DayOfYear"] / 365
    )


    # --------------------------------------------------------
    # Commodity code as frequency
    #
    # Do NOT treat Commodity_Code as a continuous number.
    # It is an identifier.
    # --------------------------------------------------------

    commodity_code_frequency = (
        train_df["Commodity_Code"]
        .value_counts()
        /
        len(train_df)
    ).to_dict()


    result["Commodity_Code_Frequency"] = (

        df["Commodity_Code"]
        .map(
            commodity_code_frequency
        )
        .fillna(0)

    )


    return result


# ============================================================
# 9. CREATE FEATURES
# ============================================================

print("\n============================================")
print("CREATING MODEL FEATURES")
print("============================================")


X_train = create_features(
    train_df
)

X_validation = create_features(
    validation_df
)

X_test = create_features(
    test_df
)


y_train = train_df[
    TARGET
]

y_validation = validation_df[
    TARGET
]

y_test = test_df[
    TARGET
]


print(
    "\nNumber of model features:",
    X_train.shape[1]
)


print(
    "\nFeatures:"
)

for column in X_train.columns:

    print(
        " -",
        column
    )


# ============================================================
# 10. LOG TARGET
# ============================================================

print("\n============================================")
print("LOG TRANSFORMING TARGET")
print("============================================")


y_train_log = np.log1p(
    y_train
)


# ============================================================
# 11. FINAL MODEL
# ============================================================

print("\n============================================")
print("CREATING HISTGRADIENTBOOSTING MODEL")
print("============================================")


model = HistGradientBoostingRegressor(

    max_iter=350,

    learning_rate=0.06,

    max_leaf_nodes=63,

    min_samples_leaf=30,

    l2_regularization=1.0,

    early_stopping=True,

    validation_fraction=0.1,

    n_iter_no_change=30,

    random_state=RANDOM_STATE

)


# ============================================================
# 12. TRAIN
# ============================================================

print("\n============================================")
print("TRAINING FINAL MODEL")
print("============================================")

training_start = time.time()


model.fit(
    X_train,
    y_train_log
)


training_time = (
    time.time()
    -
    training_start
)


print(
    f"\nTraining completed in "
    f"{training_time / 60:.2f} minutes"
)


# ============================================================
# 13. VALIDATION
# ============================================================

print("\n============================================")
print("2025 VALIDATION")
print("============================================")


validation_prediction = model.predict(
    X_validation
)


validation_prediction = np.expm1(
    validation_prediction
)


validation_prediction = np.maximum(
    validation_prediction,
    0
)


validation_mae = mean_absolute_error(
    y_validation,
    validation_prediction
)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        validation_prediction
    )
)

validation_r2 = r2_score(
    y_validation,
    validation_prediction
)


print(
    f"MAE  : ₹{validation_mae:,.2f}"
)

print(
    f"RMSE : ₹{validation_rmse:,.2f}"
)

print(
    f"R²   : {validation_r2:.4f}"
)


# ============================================================
# 14. 2026 FINAL TEST
# ============================================================

print("\n============================================")
print("2026 FINAL TEST")
print("============================================")


test_prediction = model.predict(
    X_test
)


test_prediction = np.expm1(
    test_prediction
)


test_prediction = np.maximum(
    test_prediction,
    0
)


test_mae = mean_absolute_error(
    y_test,
    test_prediction
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_prediction
    )
)

test_r2 = r2_score(
    y_test,
    test_prediction
)


print(
    f"MAE  : ₹{test_mae:,.2f}"
)

print(
    f"RMSE : ₹{test_rmse:,.2f}"
)

print(
    f"R²   : {test_r2:.4f}"
)


# ============================================================
# 15. NORMAL PRICE EVALUATION
# ============================================================

print("\n============================================")
print("NORMAL PRICE EVALUATION")
print("============================================")


normal_mask = (
    y_test <= 100000
)


normal_actual = (
    y_test[normal_mask]
)

normal_prediction = (
    test_prediction[normal_mask]
)


normal_mae = mean_absolute_error(
    normal_actual,
    normal_prediction
)

normal_rmse = np.sqrt(
    mean_squared_error(
        normal_actual,
        normal_prediction
    )
)

normal_r2 = r2_score(
    normal_actual,
    normal_prediction
)


print(
    "Normal records:",
    len(normal_actual)
)

print(
    f"Normal MAE  : ₹{normal_mae:,.2f}"
)

print(
    f"Normal RMSE : ₹{normal_rmse:,.2f}"
)

print(
    f"Normal R²   : {normal_r2:.4f}"
)


# ============================================================
# 16. PRICE RANGE ACCURACY
# ============================================================

print("\n============================================")
print("PRICE RANGE ACCURACY")
print("============================================")


absolute_error = np.abs(
    normal_actual.values
    -
    normal_prediction
)


total_normal = len(
    normal_actual
)


for limit in [
    500,
    1000,
    2000,
    5000
]:

    count = np.sum(
        absolute_error <= limit
    )

    percentage = (
        count
        /
        total_normal
        *
        100
    )

    print(
        f"Within ₹{limit:,}: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 17. SAVE TEST PREDICTIONS
# ============================================================

prediction_output = test_df[
    [
        "State",
        "District",
        "Market",
        "Commodity",
        "Variety",
        "Grade",
        "Arrival_Date"
    ]
].copy()


prediction_output[
    "Actual_Modal_Price"
] = y_test.values


prediction_output[
    "Predicted_Modal_Price"
] = test_prediction


prediction_output[
    "Absolute_Error"
] = np.abs(
    prediction_output["Actual_Modal_Price"]
    -
    prediction_output["Predicted_Modal_Price"]
)

prediction_file = os.path.join(
    MODEL_DIR,
    "final_test_predictions.csv"
)


prediction_output.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# 18. SAVE MODEL
# ============================================================

model_file = os.path.join(
    MODEL_DIR,
    "final_price_model.pkl"
)


with open(
    model_file,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ============================================================
# 19. SAVE PREPROCESSING INFORMATION
# ============================================================

preprocessing_file = os.path.join(
    MODEL_DIR,
    "final_preprocessing.pkl"
)


with open(
    preprocessing_file,
    "wb"
) as file:

    pickle.dump(

        {
            "frequency_maps":
                frequency_maps,

            "categorical_columns":
                CATEGORICAL_COLUMNS,

            "target":
                TARGET,

            "target_transformation":
                "log1p",

            "feature_columns":
                list(X_train.columns),

            "random_state":
                RANDOM_STATE

        },

        file

    )


# ============================================================
# 20. SAVE MODEL METADATA
# ============================================================

metadata = {

    "model":
        "HistGradientBoostingRegressor",

    "training_years":
        "2021-2024",

    "validation_year":
        2025,

    "test_year":
        2026,

    "training_records":
        len(train_df),

    "validation_records":
        len(validation_df),

    "test_records":
        len(test_df),

    "features":
        list(X_train.columns),

    "validation_mae":
        validation_mae,

    "validation_rmse":
        validation_rmse,

    "validation_r2":
        validation_r2,

    "test_mae":
        test_mae,

    "test_rmse":
        test_rmse,

    "test_r2":
        test_r2,

    "normal_mae":
        normal_mae,

    "normal_rmse":
        normal_rmse,

    "normal_r2":
        normal_r2,

    "training_time_seconds":
        training_time

}


metadata_file = os.path.join(
    MODEL_DIR,
    "final_model_metadata.pkl"
)


with open(
    metadata_file,
    "wb"
) as file:

    pickle.dump(
        metadata,
        file
    )


# ============================================================
# 21. SAVE HUMAN-READABLE RESULTS
# ============================================================

results = pd.DataFrame({

    "Metric": [

        "Validation MAE",

        "Validation RMSE",

        "Validation R2",

        "2026 Test MAE",

        "2026 Test RMSE",

        "2026 Test R2",

        "Normal Price MAE",

        "Normal Price RMSE",

        "Normal Price R2"

    ],

    "Value": [

        validation_mae,

        validation_rmse,

        validation_r2,

        test_mae,

        test_rmse,

        test_r2,

        normal_mae,

        normal_rmse,

        normal_r2

    ]

})


results_file = os.path.join(
    MODEL_DIR,
    "final_model_results.csv"
)


results.to_csv(
    results_file,
    index=False
)


# ============================================================
# 22. FINAL OUTPUT
# ============================================================

print("\n")
print("============================================")
print("FINAL AGRILINK MODEL COMPLETED")
print("============================================")


print("\nFINAL 2026 PERFORMANCE")

print(
    f"MAE  : ₹{test_mae:,.2f}"
)

print(
    f"RMSE : ₹{test_rmse:,.2f}"
)

print(
    f"R²   : {test_r2:.4f}"
)


print("\nNormal-price performance")

print(
    f"MAE  : ₹{normal_mae:,.2f}"
)

print(
    f"RMSE : ₹{normal_rmse:,.2f}"
)

print(
    f"R²   : {normal_r2:.4f}"
)


print("\nSaved files:")

print(
    "1.",
    model_file
)

print(
    "2.",
    preprocessing_file
)

print(
    "3.",
    metadata_file
)

print(
    "4.",
    results_file
)

print(
    "5.",
    prediction_file
)


print("\n============================================")
print("READY FOR PREDICTION API")
print("============================================")