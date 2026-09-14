import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

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


TEST_FILE = os.path.join(
    DATA_DIR,
    "test_sample.csv"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_price_model.pkl"
)

ENCODER_FILE = os.path.join(
    MODEL_DIR,
    "target_encoding_maps.pkl"
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


print(
    f"Global mean price: ₹{global_mean:,.2f}"
)


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
# ENCODE TEST DATA
# ==================================================

print("\nEncoding test data...")


for column in CATEGORICAL_FEATURES:

    test_df[column] = (
        test_df[column]
        .map(target_maps[column])
        .fillna(global_mean)
    )


# ==================================================
# CREATE X AND Y
# ==================================================

X_test = test_df[FEATURES]

y_test = test_df[TARGET]


# ==================================================
# PREDICTION
# ==================================================

print("\n========================================")
print("GENERATING PREDICTIONS")
print("========================================")


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


test_df["Predicted_Price"] = predicted_price


# ==================================================
# ERROR
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
# METRICS
# ==================================================

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
print("MODEL PERFORMANCE")
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


# ==================================================
# ERROR ANALYSIS
# ==================================================

print("\n========================================")
print("ERROR ANALYSIS")
print("========================================")


print(
    f"Mean Absolute Error: "
    f"₹{test_df['Absolute_Error'].mean():,.2f}"
)


print(
    f"Median Absolute Error: "
    f"₹{test_df['Absolute_Error'].median():,.2f}"
)


print(
    f"Maximum Error: "
    f"₹{test_df['Absolute_Error'].max():,.2f}"
)


print(
    f"Mean Percentage Error: "
    f"{test_df['Percentage_Error'].mean():.2f}%"
)


# ==================================================
# ACTUAL VS PREDICTED SAMPLE
# ==================================================

print("\n========================================")
print("SAMPLE PREDICTIONS")
print("========================================")


sample = test_df[
    [
        "Modal_Price",
        "Predicted_Price",
        "Absolute_Error"
    ]
].head(20)


print(
    sample.to_string(
        index=False
    )
)


# ==================================================
# SAVE PREDICTIONS
# ==================================================

PREDICTION_FILE = os.path.join(
    MODEL_DIR,
    "test_predictions.csv"
)


test_df.to_csv(
    PREDICTION_FILE,
    index=False
)


print(
    "\nPredictions saved to:"
)

print(
    PREDICTION_FILE
)


# ==================================================
# GRAPH 1
# ACTUAL VS PREDICTED
# ==================================================

print("\nCreating Actual vs Predicted graph...")


plt.figure(
    figsize=(8, 6)
)


# Use sample for readable graph
plot_data = test_df.sample(
    min(5000, len(test_df)),
    random_state=42
)


plt.scatter(
    plot_data[TARGET],
    plot_data["Predicted_Price"],
    alpha=0.4
)


max_value = max(
    plot_data[TARGET].max(),
    plot_data["Predicted_Price"].max()
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
    "Actual vs Predicted Market Price"
)


plt.tight_layout()


GRAPH1 = os.path.join(
    MODEL_DIR,
    "actual_vs_predicted.png"
)


plt.savefig(
    GRAPH1,
    dpi=300
)


plt.show()


# ==================================================
# GRAPH 2
# ERROR DISTRIBUTION
# ==================================================

print(
    "Creating error distribution graph..."
)


plt.figure(
    figsize=(8, 6)
)


plt.hist(
    test_df["Absolute_Error"],
    bins=50
)


plt.xlabel(
    "Absolute Prediction Error"
)


plt.ylabel(
    "Number of Records"
)


plt.title(
    "Prediction Error Distribution"
)


plt.tight_layout()


GRAPH2 = os.path.join(
    MODEL_DIR,
    "error_distribution.png"
)


plt.savefig(
    GRAPH2,
    dpi=300
)


plt.show()


# ==================================================
# GRAPH 3
# ACTUAL VS PREDICTED PRICE SAMPLE
# ==================================================

print(
    "Creating comparison graph..."
)


comparison = test_df.head(100)


plt.figure(
    figsize=(12, 6)
)


plt.plot(
    comparison[TARGET].values,
    label="Actual Price"
)


plt.plot(
    comparison["Predicted_Price"].values,
    label="Predicted Price"
)


plt.xlabel(
    "Test Record"
)


plt.ylabel(
    "Price"
)


plt.title(
    "Actual vs Predicted Prices - Sample"
)


plt.legend()


plt.tight_layout()


GRAPH3 = os.path.join(
    MODEL_DIR,
    "price_comparison_sample.png"
)


plt.savefig(
    GRAPH3,
    dpi=300
)


plt.show()


# ==================================================
# COMMODITY-WISE PERFORMANCE
# ==================================================

print(
    "\n========================================"
)

print(
    "COMMODITY-WISE PERFORMANCE"
)

print(
    "========================================"
)


commodity_results = (
    test_df
    .groupby("Commodity")
    .agg(
        Records=("Modal_Price", "count"),
        Actual_Average=("Modal_Price", "mean"),
        Predicted_Average=("Predicted_Price", "mean"),
        MAE=("Absolute_Error", "mean")
    )
    .sort_values(
        "MAE",
        ascending=True
    )
)


print(
    "\nBest performing commodities:"
)


print(
    commodity_results.head(10).to_string()
)


print(
    "\nHighest error commodities:"
)


print(
    commodity_results.tail(10).sort_values(
        "MAE",
        ascending=False
    ).to_string()
)


COMMODITY_FILE = os.path.join(
    MODEL_DIR,
    "commodity_performance.csv"
)


commodity_results.to_csv(
    COMMODITY_FILE
)


# ==================================================
# FINAL MESSAGE
# ==================================================

print("\n========================================")
print("MODEL ANALYSIS COMPLETED")
print("========================================")


print("\nGenerated files:")

print(
    PREDICTION_FILE
)

print(
    GRAPH1
)

print(
    GRAPH2
)

print(
    GRAPH3
)

print(
    COMMODITY_FILE
)

print("\nDone.")