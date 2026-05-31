from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, render_template, request
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# Basic setup
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "AmesHousingSpatial.csv"
TARGET = "Sale_Price"

app = Flask(__name__)


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET, "Latitude", "Longitude"])
    return df


DF = load_data()


# --------------------------------------------------
# Feature selection
# --------------------------------------------------

NUM_COLS = DF.select_dtypes(include=np.number).columns.tolist()
NUM_COLS.remove(TARGET)

# --------------------------------------------------
# Train/Test split
# --------------------------------------------------

X = DF.drop(columns=[TARGET])
y = DF[TARGET]

# first split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# second split (for conformal calibration)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


# --------------------------------------------------
# Random Forest model
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train[NUM_COLS], y_train)

rf_pred = rf_model.predict(X_test[NUM_COLS])

# --------------------------------------------------
# Conformal prediction (95% interval)
# --------------------------------------------------

# predictions on calibration set
cal_preds = rf_model.predict(X_cal[NUM_COLS])

# absolute residuals
cal_residuals = np.abs(y_cal - cal_preds)

# confidence level
alpha = 0.05

# quantile (conformal radius)
q = np.quantile(cal_residuals, 1 - alpha)

# --------------------------------------------------
# Linear Regression model
# --------------------------------------------------

lin_model = LinearRegression()

lin_model.fit(X_train[NUM_COLS], y_train)

lin_pred = lin_model.predict(X_test[NUM_COLS])

# --------------------------------------------------
# Model metrics
# --------------------------------------------------

MODEL_METRICS = {
    "random_forest": {
        "r2": round(r2_score(y_test, rf_pred), 4),
        "rmse": round(root_mean_squared_error(y_test, rf_pred), 2),
        "mae": round(mean_absolute_error(y_test, rf_pred), 2),
    },
    "linear_regression": {
        "r2": round(r2_score(y_test, lin_pred), 4),
        "rmse": round(root_mean_squared_error(y_test, lin_pred), 2),
        "mae": round(mean_absolute_error(y_test, lin_pred), 2),
    },
}


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

importance = rf_model.feature_importances_

FEATURE_IMPORTANCE_DF = (
    pd.DataFrame({"feature": NUM_COLS, "importance": importance})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)


# --------------------------------------------------
# Predicted price + residual
# --------------------------------------------------

DF["Predicted_Price"] = rf_model.predict(DF[NUM_COLS])
DF["Residual"] = DF[TARGET] - DF["Predicted_Price"]

DF["PI_Lower"] = DF["Predicted_Price"] - q
DF["PI_Upper"] = DF["Predicted_Price"] + q

DF["PI_Width"] = DF["PI_Upper"] - DF["PI_Lower"]

print("\nUpdated DF with rf predicted prices and residuals:")
print(DF.loc[:5, ["PI_Lower", TARGET, "PI_Upper"]])
# --------------------------------------------------
# Training set medians (used to fill missing features in custom predictions)
# --------------------------------------------------

TRAIN_MEDIANS = X_train[NUM_COLS].median().to_dict()


# --------------------------------------------------
# 10-feature prediction model (with categorical encoding)
# --------------------------------------------------

# Ordinal encoding for Overall_Cond
OVERALL_COND_MAP = {
    "Very_Poor": 1, "Poor": 2, "Below_Average": 3, "Average": 4,
    "Above_Average": 5, "Good": 6, "Very_Good": 7, "Excellent": 8,
}

# Mean target encoding for Neighborhood and House_Style (fit on training set only)
_train_with_y = X_train.copy()
_train_with_y[TARGET] = y_train.values
NEIGHBORHOOD_ENC  = _train_with_y.groupby("Neighborhood")[TARGET].mean().to_dict()
HOUSE_STYLE_ENC   = _train_with_y.groupby("House_Style")[TARGET].mean().to_dict()
GLOBAL_MEAN_PRICE = float(y_train.mean())

PRED_COLS = [
    "Overall_Cond_Enc", "Gr_Liv_Area", "Neighborhood_Enc", "Year_Built",
    "Total_Bsmt_SF", "Garage_Cars", "Full_Bath", "Bedroom_AbvGr",
    "House_Style_Enc", "Central_Air_Enc",
]

PRED_LABELS = {
    "Overall_Cond_Enc": "Overall Condition",
    "Gr_Liv_Area":      "Living Area (sqft)",
    "Neighborhood_Enc": "Neighborhood",
    "Year_Built":       "Year Built",
    "Total_Bsmt_SF":    "Basement SF",
    "Garage_Cars":      "Garage Cars",
    "Full_Bath":        "Full Bathrooms",
    "Bedroom_AbvGr":    "Bedrooms",
    "House_Style_Enc":  "House Style",
    "Central_Air_Enc":  "Central Air",
}

def encode_pred_df(df: pd.DataFrame) -> pd.DataFrame:
    enc = pd.DataFrame(index=df.index)
    enc["Overall_Cond_Enc"] = df["Overall_Cond"].map(OVERALL_COND_MAP).fillna(4.0)
    enc["Gr_Liv_Area"]      = pd.to_numeric(df["Gr_Liv_Area"], errors="coerce")
    enc["Neighborhood_Enc"] = df["Neighborhood"].map(NEIGHBORHOOD_ENC).fillna(GLOBAL_MEAN_PRICE)
    enc["Year_Built"]       = pd.to_numeric(df["Year_Built"], errors="coerce")
    enc["Total_Bsmt_SF"]    = pd.to_numeric(df["Total_Bsmt_SF"], errors="coerce")
    enc["Garage_Cars"]      = pd.to_numeric(df["Garage_Cars"], errors="coerce")
    enc["Full_Bath"]        = pd.to_numeric(df["Full_Bath"], errors="coerce")
    enc["Bedroom_AbvGr"]    = pd.to_numeric(df["Bedroom_AbvGr"], errors="coerce")
    enc["House_Style_Enc"]  = df["House_Style"].map(HOUSE_STYLE_ENC).fillna(GLOBAL_MEAN_PRICE)
    enc["Central_Air_Enc"]  = (df["Central_Air"] == "Y").astype(float)
    return enc[PRED_COLS]

X_pred_train = encode_pred_df(X_train)
X_pred_cal   = encode_pred_df(X_cal)

pred_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
pred_model.fit(X_pred_train, y_train)

# Conformal interval for pred_model
pred_cal_residuals = np.abs(y_cal - pred_model.predict(X_pred_cal))
q_pred = float(np.quantile(pred_cal_residuals, 1 - alpha))

# SHAP explainer for pred_model
pred_shap_explainer = shap.TreeExplainer(pred_model)
_ev_pred = pred_shap_explainer.expected_value
PRED_SHAP_BASE = float(_ev_pred[0] if hasattr(_ev_pred, '__len__') else _ev_pred)
print(f"\n10-feature pred_model ready. q_pred={q_pred:.0f}, SHAP base={PRED_SHAP_BASE:.0f}")


# --------------------------------------------------
# Clustering
# --------------------------------------------------

cluster_cols = [
    "Gr_Liv_Area",
    "Garage_Cars",
    "Total_Bsmt_SF",
    "Year_Built",
    #"Sale_Price",
]

cluster_data = DF[cluster_cols].fillna(DF[cluster_cols].median())

cluster_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=4, random_state=42))
])

DF["Cluster"] = cluster_pipeline.fit_predict(cluster_data)

# --------------------------------------------------
# Filtering function
# --------------------------------------------------

def apply_filters(df: pd.DataFrame, args: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()

    neighborhood = args.get("neighborhood")
    if neighborhood and neighborhood != "All":
        result = result[result["Neighborhood"] == neighborhood]

    min_price = args.get("min_price")
    if min_price not in (None, ""):
        result = result[result[TARGET] >= float(min_price)]

    max_price = args.get("max_price")
    if max_price not in (None, ""):
        result = result[result[TARGET] <= float(max_price)]

    min_year = args.get("min_year")
    if min_year not in (None, ""):
        result = result[result["Year_Built"] >= float(min_year)]

    max_year = args.get("max_year")
    if max_year not in (None, ""):
        result = result[result["Year_Built"] <= float(max_year)]

    min_qual = args.get("min_qual")
    if min_qual not in (None, ""):
        result = result[result["Overall_Cond_Num"] >= float(min_qual)]

    return result


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def index():

    neighborhoods = sorted(DF["Neighborhood"].dropna().unique())
    return render_template("index.html", neighborhoods=neighborhoods)


@app.route("/api/summary")
def summary():
    quality_map = {
        "Very_Poor": 1,
        "Poor": 2,
        "Below_Average": 3,
        "Average": 4,
        "Above_Average": 5,
        "Good": 6,
        "Very_Good": 7,
        "Excellent": 8
    }

    DF["Overall_Cond_Num"] = DF["Overall_Cond"].map(quality_map)
    return jsonify(
        {
            "row_count": int(len(DF)),
            "column_count": int(DF.shape[1]),
            "price_min": float(DF[TARGET].min()),
            "price_max": float(DF[TARGET].max()),
            "price_mean": float(DF[TARGET].mean()),
            "neighborhoods": sorted(DF["Neighborhood"].dropna().unique().tolist()),
            "year_min": int(DF["Year_Built"].min()),
            "year_max": int(DF["Year_Built"].max()),
            "quality_min": int(DF["Overall_Cond_Num"].min()),
            "quality_max": int(DF["Overall_Cond_Num"].max()),
            "model_metrics": MODEL_METRICS,
        }
    )


@app.route("/api/houses")
def houses():
    filtered = apply_filters(DF, request.args)

    metric = request.args.get("metric", "Sale_Price")

    cols = [
        "Latitude",
        "Longitude",
        "Neighborhood",
        "Sale_Price",
        "Predicted_Price",
        "Residual",
        "Cluster",
        "PI_Lower",
        "PI_Upper",
        "PI_Width",
        "Overall_Cond",
        "Gr_Liv_Area",
        "Lot_Area",
        "Year_Built",
    ]

    data = filtered[cols].copy()
    data["metric"] = data[metric]

    return jsonify(data.to_dict(orient="records"))


@app.route("/api/neighborhoods")
def neighborhoods():
    filtered = apply_filters(DF, request.args)

    grouped = (
        filtered.groupby("Neighborhood")
        .agg(
            avg_price=(TARGET, "mean"),
            median_price=(TARGET, "median"),
            n_houses=(TARGET, "size"),
        )
        .reset_index()
        .sort_values("median_price", ascending=False)
    )

    return jsonify(grouped.to_dict(orient="records"))


@app.route("/api/price_histogram")
def price_histogram():
    filtered = apply_filters(DF, request.args)

    counts, bins = np.histogram(filtered[TARGET], bins=20)

    hist = [
        {
            "bin_start": float(bins[i]),
            "bin_end": float(bins[i + 1]),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]

    return jsonify(hist)


@app.route("/api/scatter")
def scatter():
    filtered = apply_filters(DF, request.args)

    x_var = request.args.get("x", "Gr_Liv_Area")
    y_var = request.args.get("y", TARGET)

    cols = [x_var, y_var, "Neighborhood", TARGET, "Latitude", "Longitude"]

    plot_df = filtered[cols].dropna().copy()

    plot_df.columns = ["x", "y", "Neighborhood", "Sale_Price", "Latitude", "Longitude"]

    return jsonify(plot_df.to_dict(orient="records"))


@app.route("/api/feature_importance")
def feature_importance():
    top_n = int(request.args.get("top_n", 20))
    result = FEATURE_IMPORTANCE_DF.head(top_n)
    return jsonify(result.to_dict(orient="records"))


@app.route("/api/clusters")
def clusters():
    filtered = apply_filters(DF, request.args)

    grouped = (
        filtered.groupby("Cluster")
        .agg(
            avg_price=(TARGET, "mean"),
            avg_area=("Gr_Liv_Area", "mean"),
            avg_quality=("Overall_Cond_Num", "mean"),
            n_houses=(TARGET, "size"),
        )
        .reset_index()
        .sort_values("Cluster")
    )

    return jsonify(grouped.to_dict(orient="records"))


# --------------------------------------------------
# Custom prediction endpoint
# --------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)

    # Build raw feature row for all 10 inputs
    raw = {
        "Overall_Cond": body.get("overall_cond", "Average"),
        "Gr_Liv_Area":  body.get("gr_liv_area") or TRAIN_MEDIANS["Gr_Liv_Area"],
        "Neighborhood": body.get("neighborhood", ""),
        "Year_Built":   body.get("year_built") or TRAIN_MEDIANS["Year_Built"],
        "Total_Bsmt_SF": body.get("total_bsmt_sf") or TRAIN_MEDIANS["Total_Bsmt_SF"],
        "Garage_Cars":  body.get("garage_cars", 2),
        "Full_Bath":    body.get("full_bath", 2),
        "Bedroom_AbvGr": body.get("bedroom_abvgr", 3),
        "House_Style":  body.get("house_style", "One_Story"),
        "Central_Air":  body.get("central_air", "Y"),
    }

    X_input = encode_pred_df(pd.DataFrame([raw]))

    print(f"[predict] encoded input:\n{X_input.to_dict(orient='records')}")
    predicted = float(pred_model.predict(X_input)[0])
    lower = predicted - q_pred
    upper = predicted + q_pred

    # SHAP values for all 10 features, sorted by absolute magnitude
    sv = pred_shap_explainer.shap_values(X_input)[0]
    shap_values = sorted(
        [{"feature": PRED_LABELS[col], "value": round(float(v), 2)}
         for col, v in zip(PRED_COLS, sv)],
        key=lambda x: abs(x["value"]),
        reverse=True,
    )

    return jsonify({
        "predicted_price": round(predicted, 0),
        "lower": round(lower, 0),
        "upper": round(upper, 0),
        "interval_width": round(upper - lower, 0),
        "shap_base_value": round(PRED_SHAP_BASE, 0),
        "shap_values": shap_values,
    })


# --------------------------------------------------
# Run server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 10000)
