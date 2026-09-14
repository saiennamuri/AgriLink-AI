import pandas as pd
import numpy as np
import os
import pickle
import time

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# PATHS
# ==================================================

BASE_DIR = r"E:\MAHEERA\PROJECTS\SIH-AGRILINK\ml"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "model_data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


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
# ORIGINAL FEATURES
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
print("MODEL V2 - LOADING DATA")
print("========================================")


train_df = pd.read_csv(
    TRAIN_FILE
)

validation_df = pd.read_csv(
    VALIDATION_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


print(
    "Training:",
    train_df.shape
)

print(
    "Validation:",
    validation_df.shape
)

print(
    "Testing:",
    test_df.shape
)


# ==================================================
# REMOVE INVALID TARGETS
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
# CREATE INTERACTION FEATURES
# ==================================================

print("\n========================================")
print("CREATING INTERACTION FEATURES")
print("========================================")


def create_features(df):

    df = df.copy()


    # ------------------------------------------------
    # Category interactions
    # ------------------------------------------------

    df["Commodity_Market"] = (
        df["Commodity"].astype(str)
        + "_"
        + df["Market"].astype(str)
    )


    df["Commodity_State"] = (
        df["Commodity"].astype(str)
        + "_"
        + df["State"].astype(str)
    )


    df["Commodity_Season"] = (
        df["Commodity"].astype(str)
        + "_"
        + df["Season"].astype(str)
    )


    df["Market_Season"] = (
        df["Market"].astype(str)
        + "_"
        + df["Season"].astype(str)
    )


    df["Commodity_Month"] = (
        df["Commodity"].astype(str)
        + "_"
        + df["Month"].astype(str)
    )


    # ------------------------------------------------
    # Date cyclic features
    # ------------------------------------------------

    df["Month_Sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )


    df["DayOfYear_Sin"] = np.sin(
        2 * np.pi * df["DayOfYear"] / 365
    )

    df["DayOfYear_Cos"] = np.cos(
        2 * np.pi * df["DayOfYear"] / 365
    )


    return df


train_df = create_features(
    train_df
)

validation_df = create_features(
    validation_df
)

test_df = create_features(
    test_df
)


# ==================================================
# V2 CATEGORICAL FEATURES
# ==================================================

INTERACTION_FEATURES = [
    "Commodity_Market",
    "Commodity_State",
    "Commodity_Season",
    "Market_Season",
    "Commodity_Month"
]


ALL_CATEGORICAL = (
    CATEGORICAL_FEATURES
    +
    INTERACTION_FEATURES
)


# ==================================================
# TARGET ENCODING
# ==================================================

print("\n========================================")
print("CREATING TARGET ENCODINGS")
print("========================================")


global_mean = train_df[TARGET].mean()


print(
    f"Training global mean: "
    f"₹{global_mean:,.2f}"
)


target_maps = {}


for column in ALL_CATEGORICAL:

    print(
        "Encoding:",
        column
    )


    statistics = (
        train_df
        .groupby(column)[TARGET]
        .agg(["mean", "count"])
    )


    # Smoothing
    smoothing = 20


    statistics["encoded"] = (

        (
            statistics["count"]
            *
            statistics["mean"]
        )

        +

        (
            smoothing
            *
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
# ENCODE DATA
# ==================================================

def encode_data(df):

    df = df.copy()


    for column in ALL_CATEGORICAL:

        df[column] = (
            df[column]
            .map(target_maps[column])
            .fillna(global_mean)
        )


    return df


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
# FINAL FEATURES
# ==================================================

FEATURES = (
    ALL_CATEGORICAL
    +
    NUMERICAL_FEATURES
    +
    [
        "Month_Sin",
        "Month_Cos",
        "DayOfYear_Sin",
        "DayOfYear_Cos"
    ]
)


print("\n========================================")
print("FINAL FEATURE SET")
print("========================================")

print(
    "Number of features:",
    len(FEATURES)
)

print(
    FEATURES
)


# ==================================================
# X AND Y
# ==================================================

X_train = train_encoded[
    FEATURES
]

X_validation = validation_encoded[
    FEATURES
]

X_test = test_encoded[
    FEATURES
]


y_train = train_df[
    TARGET
]

y_validation = validation_df[
    TARGET
]

y_test = test_df[
    TARGET
]


# ==================================================
# LOG TARGET
# ==================================================

print(
    "\nApplying log1p transformation..."
)


y_train_log = np.log1p(
    y_train
)


# ==================================================
# MODEL EVALUATION
# ==================================================

results = []


def train_and_evaluate(
    model,
    model_name
):

    print("\n========================================")
    print(
        "TRAINING:",
        model_name
    )
    print("========================================")


    start = time.time()


    model.fit(
        X_train,
        y_train_log
    )


    training_time = (
        time.time()
        -
        start
    )


    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )


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


    results.append({

        "Model": model_name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "Training_Time": training_time

    })


    return model


# ==================================================
# MODEL 1
# RANDOM FOREST
# ==================================================

random_forest = RandomForestRegressor(

    n_estimators=120,

    max_depth=25,

    min_samples_leaf=3,

    n_jobs=-1,

    random_state=42

)


random_forest = train_and_evaluate(

    random_forest,

    "Random Forest V2"

)


# ==================================================
# MODEL 2
# EXTRA TREES
# ==================================================

extra_trees = ExtraTreesRegressor(

    n_estimators=120,

    max_depth=25,

    min_samples_leaf=3,

    n_jobs=-1,

    random_state=42

)


extra_trees = train_and_evaluate(

    extra_trees,

    "Extra Trees V2"

)


# ==================================================
# MODEL 3
# HISTOGRAM GRADIENT BOOSTING
# ==================================================

hist_gradient = HistGradientBoostingRegressor(

    max_iter=250,

    learning_rate=0.07,

    max_leaf_nodes=31,

    l2_regularization=1.0,

    random_state=42

)


hist_gradient = train_and_evaluate(

    hist_gradient,

    "HistGradientBoosting V2"

)


# ==================================================
# MODEL COMPARISON
# ==================================================

results_df = pd.DataFrame(
    results
)


print("\n========================================")
print("MODEL V2 COMPARISON")
print("========================================")


print(
    results_df
    .sort_values("MAE")
    .to_string(index=False)
)


# ==================================================
# SELECT BEST MODEL
# ==================================================

best_index = (
    results_df["MAE"]
    .idxmin()
)


best_model_name = (
    results_df
    .loc[
        best_index,
        "Model"
    ]
)


print("\n========================================")
print("BEST MODEL V2")
print("========================================")


print(
    "Selected:",
    best_model_name
)


if best_model_name == "Random Forest V2":

    best_model = random_forest


elif best_model_name == "Extra Trees V2":

    best_model = extra_trees


else:

    best_model = hist_gradient


# ==================================================
# FINAL TEST EVALUATION
# ==================================================

print("\n========================================")
print("V2 FINAL TEST EVALUATION")
print("========================================")


test_prediction_log = (
    best_model.predict(X_test)
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
# SAVE V2 MODEL
# ==================================================

V2_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_price_model_v2.pkl"
)


with open(
    V2_MODEL_FILE,
    "wb"
) as file:

    pickle.dump(
        best_model,
        file
    )


# ==================================================
# SAVE V2 ENCODERS
# ==================================================

V2_ENCODER_FILE = os.path.join(
    MODEL_DIR,
    "target_encoding_maps_v2.pkl"
)


with open(
    V2_ENCODER_FILE,
    "wb"
) as file:

    pickle.dump(

        {
            "maps": target_maps,

            "global_mean":
                global_mean,

            "categorical_features":
                ALL_CATEGORICAL,

            "features":
                FEATURES

        },

        file

    )


# ==================================================
# SAVE V2 METADATA
# ==================================================

V2_METADATA_FILE = os.path.join(
    MODEL_DIR,
    "model_metadata_v2.pkl"
)


metadata = {

    "model":
        best_model_name,

    "features":
        FEATURES,

    "target":
        TARGET,

    "target_transformation":
        "log1p",

    "test_MAE":
        test_mae,

    "test_RMSE":
        test_rmse,

    "test_R2":
        test_r2

}


with open(
    V2_METADATA_FILE,
    "wb"
) as file:

    pickle.dump(
        metadata,
        file
    )


# ==================================================
# SAVE COMPARISON
# ==================================================

COMPARISON_FILE = os.path.join(
    MODEL_DIR,
    "model_comparison_v2.csv"
)


results_df.to_csv(
    COMPARISON_FILE,
    index=False
)


# ==================================================
# SAVE TEST PREDICTIONS
# ==================================================

prediction_df = pd.DataFrame({

    "Actual_Price":
        y_test.values,

    "Predicted_Price":
        test_prediction

})


prediction_df[
    "Absolute_Error"
] = abs(

    prediction_df[
        "Actual_Price"
    ]

    -

    prediction_df[
        "Predicted_Price"
    ]

)


V2_PREDICTIONS_FILE = os.path.join(
    MODEL_DIR,
    "test_predictions_v2.csv"
)


prediction_df.to_csv(
    V2_PREDICTIONS_FILE,
    index=False
)


# ==================================================
# COMPLETED
# ==================================================

print("\n========================================")
print("MODEL V2 TRAINING COMPLETED")
print("========================================")


print("\nBest model:")

print(
    best_model_name
)


print("\nFinal V2 test metrics:")

print(
    f"MAE  : ₹{test_mae:,.2f}"
)

print(
    f"RMSE : ₹{test_rmse:,.2f}"
)

print(
    f"R²   : {test_r2:.4f}"
)


print("\nSaved files:")

print(
    V2_MODEL_FILE
)

print(
    V2_ENCODER_FILE
)

print(
    V2_METADATA_FILE
)

print(
    COMPARISON_FILE
)

print(
    V2_PREDICTIONS_FILE
)

print("\nDone.")