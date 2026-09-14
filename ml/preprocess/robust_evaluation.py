import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


# ==================================================
# PATHS
# ==================================================

BASE_DIR = r"E:\MAHEERA\PROJECTS\SIH-AGRILINK\ml"

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODEL_DATA_DIR = os.path.join(
    DATA_DIR,
    "model_data"
)


MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_price_model.pkl"
)

ENCODER_FILE = os.path.join(
    MODEL_DIR,
    "target_encoding_maps.pkl"
)

TEST_FILE = os.path.join(
    MODEL_DATA_DIR,
    "test_sample.csv"
)

ORIGINAL_DATA_FILE = os.path.join(
    DATA_DIR,
    "ml_ready_prices.csv"
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

FEATURES = (
    CATEGORICAL_FEATURES
    +
    NUMERICAL_FEATURES
)

TARGET = "Modal_Price"


# ==================================================
# LOAD MODEL
# ==================================================

print("========================================")
print("LOADING MODEL")
print("========================================")


with open(
    MODEL_FILE,
    "rb"
) as file:

    model = pickle.load(file)


print("Model loaded successfully.")


# ==================================================
# LOAD ENCODING MAPS
# ==================================================

with open(
    ENCODER_FILE,
    "rb"
) as file:

    encoder_data = pickle.load(file)


target_maps = encoder_data["maps"]

global_mean = encoder_data["global_mean"]


# ==================================================
# LOAD TEST DATA
# ==================================================

print("\n========================================")
print("LOADING TEST DATA")
print("========================================")


test_df = pd.read_csv(
    TEST_FILE
)


test_df = test_df[
    test_df[TARGET] > 0
].copy()


print(
    "Test records:",
    len(test_df)
)


# ==================================================
# PREDICTION
# ==================================================

print("\n========================================")
print("GENERATING PREDICTIONS")
print("========================================")


# Keep original category IDs
# before target encoding

for column in CATEGORICAL_FEATURES:

    test_df[column + "_ID"] = test_df[column]


# Encode categorical columns

for column in CATEGORICAL_FEATURES:

    test_df[column] = (
        test_df[column]
        .map(target_maps[column])
        .fillna(global_mean)
    )


X_test = test_df[FEATURES]

y_test = test_df[TARGET]


prediction_log = model.predict(
    X_test
)


predicted_price = np.expm1(
    prediction_log
)


predicted_price = np.maximum(
    predicted_price,
    0
)


test_df["Predicted_Price"] = (
    predicted_price
)


# ==================================================
# ERROR CALCULATIONS
# ==================================================

test_df["Absolute_Error"] = (
    abs(
        test_df[TARGET]
        -
        test_df["Predicted_Price"]
    )
)


test_df["Percentage_Error"] = (
    test_df["Absolute_Error"]
    /
    test_df[TARGET]
) * 100


# ==================================================
# OVERALL METRICS
# ==================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


mae = mean_absolute_error(
    y_test,
    predicted_price
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predicted_price
    )
)


r2 = r2_score(
    y_test,
    predicted_price
)


print("\n========================================")
print("OVERALL TEST PERFORMANCE")
print("========================================")


print(
    f"MAE  : ₹{mae:,.2f}"
)

print(
    f"RMSE : ₹{rmse:,.2f}"
)

print(
    f"R²   : {r2:.4f}"
)


print(
    f"Median Absolute Error : "
    f"₹{test_df['Absolute_Error'].median():,.2f}"
)


# ==================================================
# NORMAL PRICE DATA
# ==================================================

print("\n========================================")
print("NORMAL PRICE PERFORMANCE")
print("========================================")


# Remove extremely high price records
# only for additional evaluation.

normal_df = test_df[
    test_df[TARGET] <= 100000
].copy()


normal_actual = normal_df[TARGET]

normal_predicted = (
    normal_df["Predicted_Price"]
)


normal_mae = mean_absolute_error(
    normal_actual,
    normal_predicted
)


normal_rmse = np.sqrt(
    mean_squared_error(
        normal_actual,
        normal_predicted
    )
)


normal_r2 = r2_score(
    normal_actual,
    normal_predicted
)


print(
    "Records after removing "
    "extreme prices:",
    len(normal_df)
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


print(
    "Median Error : ₹"
    f"{normal_df['Absolute_Error'].median():,.2f}"
)


# ==================================================
# PREDICTION ACCURACY RANGES
# ==================================================

print("\n========================================")
print("PREDICTION ACCURACY RANGES")
print("========================================")


total = len(normal_df)


within_500 = (
    normal_df["Absolute_Error"] <= 500
).sum()


within_1000 = (
    normal_df["Absolute_Error"] <= 1000
).sum()


within_2000 = (
    normal_df["Absolute_Error"] <= 2000
).sum()


within_5000 = (
    normal_df["Absolute_Error"] <= 5000
).sum()


print(
    f"Within ₹500  : "
    f"{within_500:,} "
    f"({within_500 / total * 100:.2f}%)"
)


print(
    f"Within ₹1,000: "
    f"{within_1000:,} "
    f"({within_1000 / total * 100:.2f}%)"
)


print(
    f"Within ₹2,000: "
    f"{within_2000:,} "
    f"({within_2000 / total * 100:.2f}%)"
)


print(
    f"Within ₹5,000: "
    f"{within_5000:,} "
    f"({within_5000 / total * 100:.2f}%)"
)


# ==================================================
# PRICE RANGE ANALYSIS
# ==================================================

print("\n========================================")
print("PRICE RANGE PERFORMANCE")
print("========================================")


def evaluate_range(
    name,
    dataframe
):

    if len(dataframe) == 0:

        return

    range_mae = mean_absolute_error(
        dataframe[TARGET],
        dataframe["Predicted_Price"]
    )

    range_rmse = np.sqrt(
        mean_squared_error(
            dataframe[TARGET],
            dataframe["Predicted_Price"]
        )
    )

    range_r2 = r2_score(
        dataframe[TARGET],
        dataframe["Predicted_Price"]
    )


    print(
        f"\n{name}"
    )

    print(
        f"Records: {len(dataframe):,}"
    )

    print(
        f"MAE: ₹{range_mae:,.2f}"
    )

    print(
        f"RMSE: ₹{range_rmse:,.2f}"
    )

    print(
        f"R²: {range_r2:.4f}"
    )


evaluate_range(
    "₹0 – ₹1,000",
    normal_df[
        normal_df[TARGET] <= 1000
    ]
)


evaluate_range(
    "₹1,001 – ₹5,000",
    normal_df[
        (normal_df[TARGET] > 1000)
        &
        (normal_df[TARGET] <= 5000)
    ]
)


evaluate_range(
    "₹5,001 – ₹10,000",
    normal_df[
        (normal_df[TARGET] > 5000)
        &
        (normal_df[TARGET] <= 10000)
    ]
)


evaluate_range(
    "₹10,001 – ₹25,000",
    normal_df[
        (normal_df[TARGET] > 10000)
        &
        (normal_df[TARGET] <= 25000)
    ]
)


evaluate_range(
    "₹25,001 – ₹100,000",
    normal_df[
        (normal_df[TARGET] > 25000)
    ]
)


# ==================================================
# SAVE ROBUST RESULTS
# ==================================================

results_file = os.path.join(
    MODEL_DIR,
    "robust_test_predictions.csv"
)


test_df.to_csv(
    results_file,
    index=False
)


print(
    "\nSaved detailed predictions:"
)

print(
    results_file
)


# ==================================================
# GRAPH 1
# NORMAL ACTUAL VS PREDICTED
# ==================================================

print(
    "\nCreating normal-price graph..."
)


plot_df = normal_df.sample(
    min(5000, len(normal_df)),
    random_state=42
)


plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    plot_df[TARGET],
    plot_df["Predicted_Price"],
    alpha=0.4
)


max_value = max(
    plot_df[TARGET].max(),
    plot_df["Predicted_Price"].max()
)


plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)


plt.xlabel(
    "Actual Modal Price"
)


plt.ylabel(
    "Predicted Modal Price"
)


plt.title(
    "Actual vs Predicted Prices - Normal Range"
)


plt.tight_layout()


graph1 = os.path.join(
    MODEL_DIR,
    "normal_actual_vs_predicted.png"
)


plt.savefig(
    graph1,
    dpi=300
)


plt.show()


# ==================================================
# GRAPH 2
# ERROR DISTRIBUTION
# ==================================================

print(
    "Creating normal error graph..."
)


plt.figure(
    figsize=(8, 6)
)


plt.hist(
    normal_df["Absolute_Error"],
    bins=50
)


plt.xlabel(
    "Absolute Prediction Error"
)


plt.ylabel(
    "Number of Records"
)


plt.title(
    "Prediction Error Distribution - Normal Prices"
)


plt.tight_layout()


graph2 = os.path.join(
    MODEL_DIR,
    "normal_error_distribution.png"
)


plt.savefig(
    graph2,
    dpi=300
)


plt.show()


# ==================================================
# SAVE SUMMARY
# ==================================================

summary = pd.DataFrame({

    "Metric": [

        "Overall MAE",
        "Overall RMSE",
        "Overall R2",

        "Normal Price MAE",
        "Normal Price RMSE",
        "Normal Price R2",

        "Median Normal Error",

        "Within 500",
        "Within 1000",
        "Within 2000",
        "Within 5000"
    ],

    "Value": [

        mae,
        rmse,
        r2,

        normal_mae,
        normal_rmse,
        normal_r2,

        normal_df[
            "Absolute_Error"
        ].median(),

        within_500 / total * 100,
        within_1000 / total * 100,
        within_2000 / total * 100,
        within_5000 / total * 100
    ]
})


summary_file = os.path.join(
    MODEL_DIR,
    "robust_evaluation_summary.csv"
)


summary.to_csv(
    summary_file,
    index=False
)


# ==================================================
# COMPLETED
# ==================================================

print("\n========================================")
print("ROBUST EVALUATION COMPLETED")
print("========================================")


print("\nGenerated files:")

print(
    results_file
)

print(
    graph1
)

print(
    graph2
)

print(
    summary_file
)

print("\nDone.")