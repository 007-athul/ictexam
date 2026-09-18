import os

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Revenue Predictor", page_icon="$", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetricValue"] {color: #087f5b;}
    .hero {padding: 1.5rem 0 1rem;}
    .hero h1 {font-size: 3rem; margin-bottom: .25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if os.path.exists("G3.csv"):
        return pd.read_csv("G3.csv")
    return None


def available_features(data):
    preferred_numeric = [
        "employees", "employees_year", "revenue_year", "net_income",
        "net_income_year", "market_cap", "legal_units_count",
        "direct_subsidiaries_count", "max_hierarchy_depth",
    ]
    preferred_categorical = [
        "enterprise_group_jurisdiction", "company_type", "revenue_currency",
        "net_income_currency", "market_cap_currency", "market_cap_source",
    ]
    numeric = [column for column in preferred_numeric if column in data.columns]
    categorical = [column for column in preferred_categorical if column in data.columns]
    if not numeric:
        numeric = [column for column in data.select_dtypes(include="number").columns if column != "revenue"]
    return numeric, categorical


def build_pipeline(numeric_columns, categorical_columns, model_name):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ], remainder="drop")
    models = {
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Linear Regression": LinearRegression(),
    }
    return Pipeline([("preprocessor", preprocessor), ("model", models[model_name])])


def label(column):
    return column.replace("_", " ").title()


with st.sidebar:
    st.title("Revenue Predictor")
    page = st.radio("Navigate", ["Home", "Predict", "Results"], label_visibility="collapsed")
    st.divider()
    st.subheader("Model settings")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    model_name = st.selectbox("Regression model", ["Gradient Boosting", "Random Forest", "Linear Regression"])
    test_size = st.slider("Test data size", 0.1, 0.4, 0.2, 0.05)

data = load_data(uploaded_file)
if data is None:
    st.info("Place G3.csv beside this app or upload a CSV file to begin.")
    st.stop()
if "revenue" not in data.columns:
    st.error("The CSV must contain a `revenue` column.")
    st.stop()

data = data.copy()
data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce")
data = data.dropna(subset=["revenue"]).drop_duplicates()
numeric_columns, categorical_columns = available_features(data)
feature_columns = numeric_columns + categorical_columns
if not feature_columns:
    st.error("No usable prediction features were found in the CSV.")
    st.stop()
if len(data) < 5:
    st.error("At least 5 rows with a valid revenue value are required for training.")
    st.stop()

X = data[feature_columns]
y = data["revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
pipeline = build_pipeline(numeric_columns, categorical_columns, model_name)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

if page == "Home":
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.title("Revenue Predictor")
    st.write("Estimate a company's revenue using a machine-learning regression model trained on the supplied company dataset.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.subheader("Project overview")
    overview_col, workflow_col = st.columns(2)
    with overview_col:
        st.write("This project analyzes company information such as employees, net income, market capitalization, company type, and currency to predict revenue.")
        st.write("The app automatically cleans missing values, encodes categorical fields, scales numeric fields, and trains the selected model.")
    with workflow_col:
        st.markdown("**How to use the app**")
        st.markdown("1. Open **Predict** and enter company information.\n2. Select **Predict revenue**.\n3. Open **Results** to review the estimate and input summary.")
    st.subheader("Dataset and model snapshot")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Companies", f"{len(data):,}")
    metric_col2.metric("Input features", len(feature_columns))
    metric_col3.metric("Median revenue", f"{data['revenue'].median():,.0f}")
    metric_col4.metric("Model R²", f"{r2_score(y_test, predictions):.3f}")
    st.subheader("Revenue distribution")
    st.bar_chart(data["revenue"].value_counts(bins=12).sort_index())

elif page == "Predict":
    st.title("Predict revenue")
    st.caption("Enter the company information available to you. The model handles missing values automatically.")
    input_values = {}
    input_columns = st.columns(2)
    for index, column in enumerate(numeric_columns):
        default_value = float(data[column].median()) if data[column].notna().any() else 0.0
        input_values[column] = input_columns[index % 2].number_input(label(column), value=default_value, format="%.2f")
    for index, column in enumerate(categorical_columns):
        values = sorted(data[column].dropna().astype(str).unique().tolist())
        input_values[column] = input_columns[index % 2].selectbox(label(column), ["Unknown"] + values)
    if st.button("Predict revenue", type="primary", use_container_width=True):
        input_frame = pd.DataFrame([input_values])
        for column in categorical_columns:
            if input_frame.at[0, column] == "Unknown":
                input_frame.at[0, column] = None
        prediction = float(pipeline.predict(input_frame)[0])
        st.session_state["prediction"] = prediction
        st.session_state["input_values"] = input_values
        st.session_state["model_name"] = model_name
        st.success("Prediction ready. Open the Results page to view it.")

else:
    st.title("Prediction results")
    if "prediction" not in st.session_state:
        st.info("No prediction is available yet. Open the Predict page and submit company information first.")
        st.stop()
    prediction = st.session_state["prediction"]
    st.success("Revenue prediction generated successfully.")
    result_col1, result_col2 = st.columns(2)
    result_col1.metric("Predicted revenue", f"{prediction:,.2f}")
    result_col2.metric("Model used", st.session_state.get("model_name", model_name))
    st.subheader("Submitted company information")
    submitted = st.session_state.get("input_values", {})
    display_values = {label(key): ("Not provided" if value == "Unknown" else value) for key, value in submitted.items()}
    st.dataframe(pd.DataFrame([display_values]), use_container_width=True, hide_index=True)
    st.caption("This estimate is based on the selected model and the currently loaded dataset.")

with st.expander("Model evaluation"):
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("R² score", f"{r2_score(y_test, predictions):.3f}")
    metric_col2.metric("Mean absolute error", f"{mean_absolute_error(y_test, predictions):,.2f}")
    metric_col3.metric("RMSE", f"{mean_squared_error(y_test, predictions) ** 0.5:,.2f}")