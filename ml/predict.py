import os
import pickle
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_price_model.pkl"
)

PREPROCESSING_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_preprocessing.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

with open(
    MODEL_PATH,
    "rb"
) as file:

    model = pickle.load(file)


# ============================================================
# LOAD PREPROCESSING
# ============================================================

with open(
    PREPROCESSING_PATH,
    "rb"
) as file:

    preprocessing = pickle.load(file)


# ============================================================
# LOAD ENCODING INFORMATION
# ============================================================

frequency_maps = preprocessing[
    "frequency_maps"
]

categorical_columns = preprocessing[
    "categorical_columns"
]

commodity_code_frequency = preprocessing[
    "commodity_code_frequency"
]

feature_columns = preprocessing[
    "feature_columns"
]


# ============================================================
# SEASON
# ============================================================

def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Summer"

    elif month in [6, 7, 8, 9]:
        return "Monsoon"

    else:
        return "Post-Monsoon"


# ============================================================
# CASE-INSENSITIVE FREQUENCY LOOKUP
# ============================================================

def frequency_encode(column, value):

    mapping = frequency_maps[column]

    # Handle missing values
    if value is None:
        return 0.0

    value = str(value).strip()

    # Empty value
    if value == "":
        return 0.0

    # Exact match
    if value in mapping:

        return float(
            mapping[value]
        )

    # Case-insensitive match
    value_lower = value.lower()

    for category, frequency in mapping.items():

        if str(category).strip().lower() == value_lower:

            return float(frequency)

    # Unknown category
    return 0.0


# ============================================================
# COMMODITY CODE FREQUENCY
# ============================================================

def get_commodity_code_frequency(
    commodity_code
):

    if commodity_code is None:
        return 0.0

    # Exact numeric key
    if commodity_code in commodity_code_frequency:

        return float(
            commodity_code_frequency[
                commodity_code
            ]
        )

    # Handle NumPy integer / float differences
    for code, frequency in commodity_code_frequency.items():

        try:

            if int(code) == int(commodity_code):

                return float(frequency)

        except (ValueError, TypeError):

            continue

    return 0.0


# ============================================================
# COMMODITY CODE
# ============================================================

def get_commodity_code(
    commodity
):

    # The model does not use raw Commodity_Code.
    # We only need the code to obtain its frequency.

    # Search the original category/code relationship
    # stored separately by the project.

    commodity_code_path = os.path.join(
        BASE_DIR,
        "models",
        "commodity_code_map.pkl"
    )

    if not os.path.exists(
        commodity_code_path
    ):

        raise FileNotFoundError(
            "commodity_code_map.pkl not found."
        )

    with open(
        commodity_code_path,
        "rb"
    ) as file:

        commodity_code_map = pickle.load(
            file
        )

    # Exact match
    if commodity in commodity_code_map:

        return int(
            commodity_code_map[
                commodity
            ]
        )

    # Case-insensitive match
    commodity_lower = str(
        commodity
    ).strip().lower()

    for name, code in commodity_code_map.items():

        if str(name).strip().lower() == commodity_lower:

            return int(code)

    return None


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_price(
    state,
    district,
    market,
    commodity,
    variety,
    grade,
    date
):

    # --------------------------------------------------------
    # Missing Variety / Grade
    # --------------------------------------------------------

    if variety is None:
        variety = ""

    if grade is None:
        grade = ""

    # --------------------------------------------------------
    # Parse date
    # --------------------------------------------------------

    date = pd.to_datetime(
        date
    )

    year = date.year
    month = date.month
    day = date.day
    day_of_week = date.dayofweek
    day_of_year = date.dayofyear
    quarter = date.quarter

    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    season = get_season(
        month
    )

    # --------------------------------------------------------
    # Frequency encoded features
    # --------------------------------------------------------

    state_frequency = frequency_encode(
        "State",
        state
    )

    district_frequency = frequency_encode(
        "District",
        district
    )

    market_frequency = frequency_encode(
        "Market",
        market
    )

    commodity_frequency = frequency_encode(
        "Commodity",
        commodity
    )

    variety_frequency = frequency_encode(
        "Variety",
        variety
    )

    grade_frequency = frequency_encode(
        "Grade",
        grade
    )

    season_frequency = frequency_encode(
        "Season",
        season
    )

    # --------------------------------------------------------
    # Commodity code frequency
    # --------------------------------------------------------

    commodity_code = get_commodity_code(
        commodity
    )

    if commodity_code is None:

        raise ValueError(
            f"Commodity '{commodity}' "
            f"was not found."
        )

    commodity_code_freq = (
        get_commodity_code_frequency(
            commodity_code
        )
    )

    # --------------------------------------------------------
    # Cyclical features
    # EXACT SAME FORMULAS AS TRAINING
    # --------------------------------------------------------

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    day_of_year_sin = np.sin(
        2 * np.pi * day_of_year / 365
    )

    day_of_year_cos = np.cos(
        2 * np.pi * day_of_year / 365
    )

    # --------------------------------------------------------
    # EXACT 18 FEATURES
    # --------------------------------------------------------

    features = {

        "State_Frequency":
            state_frequency,

        "District_Frequency":
            district_frequency,

        "Market_Frequency":
            market_frequency,

        "Commodity_Frequency":
            commodity_frequency,

        "Variety_Frequency":
            variety_frequency,

        "Grade_Frequency":
            grade_frequency,

        "Season_Frequency":
            season_frequency,

        "Year":
            year,

        "Month":
            month,

        "Day":
            day,

        "DayOfWeek":
            day_of_week,

        "DayOfYear":
            day_of_year,

        "Quarter":
            quarter,

        "Month_Sin":
            month_sin,

        "Month_Cos":
            month_cos,

        "DayOfYear_Sin":
            day_of_year_sin,

        "DayOfYear_Cos":
            day_of_year_cos,

        "Commodity_Code_Frequency":
            commodity_code_freq
    }

    # --------------------------------------------------------
    # DataFrame in EXACT training order
    # --------------------------------------------------------

    X = pd.DataFrame(
        [features]
    )

    X = X[
        feature_columns
    ]

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction_log = model.predict(
        X
    )[0]

    # --------------------------------------------------------
    # REVERSE log1p
    # --------------------------------------------------------

    predicted_price = np.expm1(
        prediction_log
    )

    predicted_price = max(
        0,
        float(predicted_price)
    )

    return predicted_price


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    print("\n==========================================")
    print("       AgriLink AI Price Prediction")
    print("==========================================\n")

    state = input(
        "Enter State: "
    ).strip()

    district = input(
        "Enter District: "
    ).strip()

    market = input(
        "Enter Market: "
    ).strip()

    commodity = input(
        "Enter Commodity: "
    ).strip()

    variety = input(
        "Enter Variety (optional): "
    ).strip()

    grade = input(
        "Enter Grade (optional): "
    ).strip()

    date = input(
        "Enter Date (YYYY-MM-DD): "
    ).strip()

    try:

        price = predict_price(
            state,
            district,
            market,
            commodity,
            variety,
            grade,
            date
        )

        print(
            "\n------------------------------------------"
        )

        print(
            f"Expected Price : "
            f"₹{price:,.2f} / quintal"
        )

        print(
            "------------------------------------------"
        )

    except Exception as e:

        print(
            "\nPrediction failed."
        )

        print(
            "Error:",
            e
        )