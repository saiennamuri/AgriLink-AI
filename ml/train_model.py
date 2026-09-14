import pandas as pd
import numpy as np
import os
import pickle
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# PATHS
# ==================================================

DATA_DIR = r"E:\MAHEERA\PROJECTS\SIH-AGRILINK\ml\data\model_data"

MODEL_DIR = r"E:\MAHEERA\PROJECTS\SIH-AGRILINK\ml\models"


TRAIN_FILE = os.path.join(
    DATA_DIR,
    "train_sample.csv"
)

VALIDATION_FILE = os.path.join(
    DATA_DIR,
    "validation_sample.csv"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "test_sample.csv"
)


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
# LOAD DATA
# ==================================================

print("========================================")
print("LOADING DATA")
print("========================================")

train_df = pd.read_csv(TRAIN_FILE)

validation_df = pd.read_csv(VALIDATION_FILE)

test_df = pd.read_csv(TEST_FILE)


print("\nTraining:", train_df.shape)

print("Validation:", validation_df.shape)

print("Testing:", test_df.shape)


# ==================================================
# TARGET CLEANING
# ==================================================

train_df = train_df[
    train_df[TARGET] > 0
].copy()

validation_df = validation_df[
    validation_df[TARGET] > 0
].copy()

test_df = test_df[
    test_df[TARGET] > 0
].copy()


# ==================================================
# TARGET ENCODING
# ==================================================

print("\n========================================")
print("TARGET ENCODING")
print("========================================")

global_mean = train_df[TARGET].mean()

print(
    f"Training global mean: ₹{global_mean:,.2f}"
)


# --------------------------------------------------
# Create target maps
# --------------------------------------------------

target_maps = {}


for column in CATEGORICAL_FEATURES:

    print(
        f"Encoding {column}..."
    )

    statistics = train_df.groupby(
        column
    )[TARGET].agg(
        ["mean", "count"]
    )


    # Smoothing
    smoothing = 20

    statistics["encoded"] = (
        (
            statistics["count"] *
            statistics["mean"]
        )
        +
        (
            smoothing *
            global_mean
        )
    ) / (
        statistics["count"]
        +
        smoothing
    )


    target_maps[column] = (
        statistics["encoded"].to_dict()
    )


# ==================================================
# ENCODING FUNCTION
# ==================================================

def encode_data(df):

    df = df.copy()


    for column in CATEGORICAL_FEATURES:

        df[column] = (
            df[column]
            .map(target_maps[column])
            .fillna(global_mean)
        )


    return df


# ==================================================
# APPLY ENCODING
# ==================================================

print("\nEncoding training data...")

train_encoded = encode_data(
    train_df
)


print("Encoding validation data...")

validation_encoded = encode_data(
    validation_df
)


print("Encoding testing data...")

test_encoded = encode_data(
    test_df
)


# ==================================================
# CREATE X AND Y
# ==================================================

FEATURES = (
    CATEGORICAL_FEATURES
    +
    NUMERICAL_FEATURES
)


X_train = train_encoded[FEATURES]

X_validation = validation_encoded[FEATURES]

X_test = test_encoded[FEATURES]


y_train = train_df[TARGET]

y_validation = validation_df[TARGET]

y_test = test_df[TARGET]


# ==================================================
# LOG TRANSFORMATION
# ==================================================

print("\nApplying log1p transformation...")

y_train_log = np.log1p(y_train)


# ==================================================
# MODEL EVALUATION FUNCTION
# ==================================================

def evaluate_model(
    model,
    model_name
):

    print("\n========================================")

    print(
        "TRAINING:",
        model_name
    )

    print("========================================")


    start_time = time.time()


    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    model.fit(
        X_train,
        y_train_log
    )


    training_time = (
        time.time()
        -
        start_time
    )


    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )


    # --------------------------------------------------
    # Validation prediction
    # --------------------------------------------------

    print(
        "Predicting validation data..."
    )


    prediction_log = model.predict(
        X_validation
    )


    prediction = np.expm1(
        prediction_log
    )


    prediction = np.maximum(
        prediction,
        0
    )


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_validation,
        prediction
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            prediction
        )
    )


    r2 = r2_score(
        y_validation,
        prediction
    )


    print("\nValidation Results:")

    print(
        f"MAE  : ₹{mae:,.2f}"
    )

    print(
        f"RMSE : ₹{rmse:,.2f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )


    return model, mae, rmse, r2


# ==================================================
# MODEL 1 — RIDGE
# ==================================================

ridge = Ridge(
    alpha=10.0
)


ridge, ridge_mae, ridge_rmse, ridge_r2 = evaluate_model(
    ridge,
    "Ridge Regression"
)


# ==================================================
# MODEL 2 — RANDOM FOREST
# ==================================================

print("\nPreparing Random Forest...")

random_forest = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)


random_forest, rf_mae, rf_rmse, rf_r2 = evaluate_model(
    random_forest,
    "Random Forest"
)


# ==================================================
# MODEL 3 — HISTOGRAM GRADIENT BOOSTING
# ==================================================

hist_gradient = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)


hist_gradient, hgb_mae, hgb_rmse, hgb_r2 = evaluate_model(
    hist_gradient,
    "HistGradientBoosting"
)


# ==================================================
# MODEL COMPARISON
# ==================================================

results = pd.DataFrame({

    "Model": [
        "Ridge Regression",
        "Random Forest",
        "HistGradientBoosting"
    ],

    "MAE": [
        ridge_mae,
        rf_mae,
        hgb_mae
    ],

    "RMSE": [
        ridge_rmse,
        rf_rmse,
        hgb_rmse
    ],

    "R2": [
        ridge_r2,
        rf_r2,
        hgb_r2
    ]
})


print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    results.to_string(
        index=False
    )
)


# ==================================================
# SELECT BEST MODEL
# ==================================================

best_index = results["MAE"].idxmin()

best_model_name = results.loc[
    best_index,
    "Model"
]


print("\n========================================")
print("BEST MODEL")
print("========================================")

print(
    "Selected:",
    best_model_name
)


if best_model_name == "Ridge Regression":

    best_model = ridge

elif best_model_name == "Random Forest":

    best_model = random_forest

else:

    best_model = hist_gradient


# ==================================================
# FINAL TEST EVALUATION
# ==================================================

print("\n========================================")
print("FINAL TEST EVALUATION")
print("========================================")


test_prediction_log = best_model.predict(
    X_test
)


test_prediction = np.expm1(
    test_prediction_log
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
    f"Test MAE  : ₹{test_mae:,.2f}"
)

print(
    f"Test RMSE : ₹{test_rmse:,.2f}"
)

print(
    f"Test R²   : {test_r2:.4f}"
)


# ==================================================
# SAVE MODEL
# ==================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_price_model.pkl"
)


with open(
    MODEL_FILE,
    "wb"
) as file:

    pickle.dump(
        best_model,
        file
    )


# ==================================================
# SAVE TARGET ENCODING MAPS
# ==================================================

ENCODER_FILE = os.path.join(
    MODEL_DIR,
    "target_encoding_maps.pkl"
)


with open(
    ENCODER_FILE,
    "wb"
) as file:

    pickle.dump(
        {
            "maps": target_maps,
            "global_mean": global_mean,
            "categorical_features":
                CATEGORICAL_FEATURES
        },
        file
    )


# ==================================================
# SAVE MODEL METADATA
# ==================================================

metadata = {

    "model": best_model_name,

    "features": FEATURES,

    "categorical_features":
        CATEGORICAL_FEATURES,

    "numerical_features":
        NUMERICAL_FEATURES,

    "target": TARGET,

    "target_transformation":
        "log1p",

    "test_MAE":
        test_mae,

    "test_RMSE":
        test_rmse,

    "test_R2":
        test_r2
}


METADATA_FILE = os.path.join(
    MODEL_DIR,
    "model_metadata.pkl"
)


with open(
    METADATA_FILE,
    "wb"
) as file:

    pickle.dump(
        metadata,
        file
    )


# ==================================================
# SAVE COMPARISON
# ==================================================

RESULTS_FILE = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)


results.to_csv(
    RESULTS_FILE,
    index=False
)


# ==================================================
# COMPLETED
# ==================================================

print("\n========================================")
print("MODEL TRAINING COMPLETED")
print("========================================")

print("\nBest model:")
print(best_model_name)

print("\nSaved files:")

print(MODEL_FILE)

print(ENCODER_FILE)

print(METADATA_FILE)

print(RESULTS_FILE)

print("\nDone.")