
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Battery SoH Prediction",
    page_icon="🔋",
    layout="wide"
)
 
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.markdown("""
<div class="main-header">
    <h1>🔋 Battery State of Health Prediction</h1>
    <p>Machine Learning Based Lithium-Ion Battery Health Monitoring</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("🔋 Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Home",
        "Dataset",
        "Data Visualization",
        "Model Performance",
        "SoH Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Battery SoH Prediction using Machine Learning")

@st.cache_data
def load_default_dataset():
    file_path = "Battery_SoH_Dataset.xlsx"
    workbook = pd.ExcelFile(file_path)
    preferred_sheets = ["Battery Dataset", "Dataset"]
    sheet_name = next(
        (name for name in preferred_sheets if name in workbook.sheet_names),
        workbook.sheet_names[0]
    )
    return pd.read_excel(file_path, sheet_name=sheet_name)


try:
    df = load_default_dataset()
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()

FEATURE_COLS = [
    "cycle_number",
    "ambient_temp_C",
    "avg_voltage_V",
    "avg_current_A",
    "max_temp_C",
    "internal_resistance_mO",
    "charge_time_min",
    "discharge_time_min",
    "voltage_drop_V",
    "cc_charge_ratio"
]

TARGET_COL = "SoH"

required_cols = FEATURE_COLS + [TARGET_COL, "cell_id"]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

df = df.dropna(subset=required_cols).copy()

if df["cell_id"].nunique() < 2:
    st.error("At least two battery cells are required.")
    st.stop()

@st.cache_resource
def train_models(data):

    X = data[FEATURE_COLS]
    y = data[TARGET_COL]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=data["cell_id"])
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42
        ),

        "SVR": SVR(
            kernel="rbf",
            C=50,
            epsilon=0.15,
            gamma="scale"
        )
    }

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = {}
    predictions = {}

    for name, model in models.items():

        scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=cv,
            scoring="r2",
            n_jobs=-1
        )

        model.fit(X_train_scaled, y_train)

        pred = model.predict(X_test_scaled)

        predictions[name] = pred

        results[name] = {
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
            "CV_R2_mean": scores.mean(),
            "CV_R2_std": scores.std()
        }

    results_df = pd.DataFrame(results).T

    return (
        models, scaler, X_test, y_test,
        predictions, results_df, train_idx, test_idx
    )

with st.spinner("Training machine learning models..."):

    (
        models,
        scaler,
        X_test,
        y_test,
        predictions,
        results_df,
        train_idx,
        test_idx
    ) = train_models(df)

best_model_name = results_df["R2"].idxmax()
best_model = models[best_model_name]

if page == "Home":

    st.subheader("Project Overview")

    st.write(
        "This project predicts lithium-ion battery State of Health "
        "using machine learning algorithms and battery operating parameters."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Battery Cells", df["cell_id"].nunique())
    c3.metric("Features", len(FEATURE_COLS))
    c4.metric("Best Model", best_model_name)

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Machine Learning Algorithms")

    col1, col2, col3 = st.columns(3)

    col1.info("Random Forest Regression")
    col2.success("Gradient Boosting Regression")
    col3.warning("Support Vector Regression")

    st.markdown("### Best Model Accuracy")
    st.metric(
        "Test R² Score",
        f"{results_df.loc[best_model_name, 'R2']:.4f}"
    )

elif page == "Dataset":

    st.subheader("Battery Dataset")

    st.write("Complete battery dataset used for model training.")

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download Dataset CSV",
        df.to_csv(index=False).encode("utf-8"),
        "battery_soh_dataset.csv",
        "text/csv"
    )

    st.markdown("### Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

elif page == "Data Visualization":

    st.subheader("Battery Data Visualization")

    fig, ax = plt.subplots(figsize=(10, 5))

    for cell_id, group in df.groupby("cell_id"):
        group = group.sort_values("cycle_number")
        ax.plot(
            group["cycle_number"],
            group["SoH"],
            label=cell_id
        )

    ax.axhline(
        80,
        color="red",
        linestyle="--",
        label="80% EOL Threshold"
    )

    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("State of Health (%)")
    ax.set_title("Battery Degradation Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### SoH Distribution")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df["SoH"], bins=30, edgecolor="black")
    ax.set_xlabel("SoH (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Battery SoH Distribution")

    st.pyplot(fig)
    plt.close(fig)

elif page == "Model Performance":

    st.subheader("Machine Learning Model Comparison")

    st.dataframe(
        results_df.round(4),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    for col, metric in zip(
        [col1, col2, col3],
        ["MAE", "RMSE", "R2"]
    ):
        col.metric(
            f"Best {metric}",
            f"{results_df[metric].min() if metric != 'R2' else results_df[metric].max():.4f}"
        )

    st.markdown("### Actual vs Predicted SoH")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, pred) in zip(axes, predictions.items()):

        ax.scatter(y_test, pred, alpha=0.4)

        low = min(y_test.min(), pred.min())
        high = max(y_test.max(), pred.max())

        ax.plot(
            [low, high],
            [low, high],
            "r--"
        )

        ax.set_title(name)
        ax.set_xlabel("Actual SoH")
        ax.set_ylabel("Predicted SoH")
        ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Model Performance Comparison")

    fig, ax = plt.subplots(figsize=(9, 4))

    results_df[["MAE", "RMSE"]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Error")
    ax.set_title("MAE and RMSE Comparison")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Feature Importance")

    for name in ["Random Forest", "Gradient Boosting"]:

        model = models[name]
        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": importance
        }).sort_values("Importance")

        st.write(f"**{name}**")

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )

        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    st.download_button(
        "Download Model Results",
        results_df.to_csv().encode("utf-8"),
        "model_comparison_results.csv",
        "text/csv"
    )

elif page == "SoH Prediction":

    st.subheader("Predict Battery State of Health")

    st.write(
        "Enter the battery operating parameters to predict its State of Health."
    )

    col1, col2 = st.columns(2)

    with col1:

        cycle = st.number_input(
            "Cycle Number",
            min_value=1,
            value=150
        )

        ambient = st.number_input(
            "Ambient Temperature (°C)",
            value=25.0
        )

        voltage = st.number_input(
            "Average Voltage (V)",
            value=3.62
        )

        current = st.number_input(
            "Average Current (A)",
            value=1.7
        )

        max_temp = st.number_input(
            "Maximum Temperature (°C)",
            value=32.0
        )

    with col2:

        resistance = st.number_input(
            "Internal Resistance (mΩ)",
            value=58.0
        )

        charge = st.number_input(
            "Charge Time (min)",
            value=66.0
        )

        discharge = st.number_input(
            "Discharge Time (min)",
            value=58.0
        )

        drop = st.number_input(
            "Voltage Drop (V)",
            value=0.17
        )

        ratio = st.number_input(
            "CC Charge Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.48
        )

    if st.button("Predict Battery SoH", type="primary"):

        new_reading = pd.DataFrame([{
            "cycle_number": cycle,
            "ambient_temp_C": ambient,
            "avg_voltage_V": voltage,
            "avg_current_A": current,
            "max_temp_C": max_temp,
            "internal_resistance_mO": resistance,
            "charge_time_min": charge,
            "discharge_time_min": discharge,
            "voltage_drop_V": drop,
            "cc_charge_ratio": ratio
        }])

        scaled = scaler.transform(new_reading[FEATURE_COLS])

        predicted_soh = best_model.predict(scaled)[0]

        st.markdown("### Prediction Result")

        st.metric(
            "Predicted Battery SoH",
            f"{predicted_soh:.2f}%"
        )

        if predicted_soh >= 80:
            st.success("Battery SoH is at or above the 80% threshold.")
        else:
            st.warning("Battery SoH is below the 80% threshold.")

        st.progress(
            int(np.clip(predicted_soh, 0, 100))
        )

        st.caption(
            f"Prediction generated using {best_model_name}."
        )

st.markdown("""
<div class="footer">
    Battery State of Health Prediction | Machine Learning Project
</div>
""", unsafe_allow_html=True)