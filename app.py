import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# Clean blue/purple theme — no orange
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 750;
    color: #26324a;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #667085;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #26324a;
    margin-top: 28px;
    margin-bottom: 12px;
}

.info-box {
    padding: 16px;
    border-radius: 12px;
    background-color: #f4f7ff;
    border-left: 5px solid #667eea;
    margin-bottom: 20px;
}

.metric-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dbe4f0;
    background-color: #f8faff;
    text-align: center;
}

.metric-label {
    font-size: 14px;
    color: #667085;
}

.metric-value {
    font-size: 28px;
    font-weight: 750;
    color: #4256c5;
}

.comparison-note {
    padding: 15px;
    border-radius: 10px;
    background-color: #f5f7fb;
    border-left: 4px solid #667eea;
    margin-bottom: 18px;
}

.footer {
    color: #667085;
    text-align: center;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📊 Bank Marketing Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning classification of whether a bank customer '
    'subscribes to a term deposit.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL FILE NAMES
# ============================================================

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest"
]

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}


# ============================================================
# FIND FILES ROBUSTLY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def find_file(filename):
    """
    Search for a required file in:
    1. app.py directory
    2. common subdirectories
    3. repository recursively

    This avoids problems when files are stored in root/model/models/etc.
    """

    possible_locations = [
        BASE_DIR / filename,
        BASE_DIR / "model" / filename,
        BASE_DIR / "models" / filename,
        BASE_DIR / "artifacts" / filename
    ]

    for path in possible_locations:
        if path.exists():
            return str(path)

    # Recursive search
    matches = list(BASE_DIR.rglob(filename))

    if matches:
        return str(matches[0])

    return None


# ============================================================
# RESOLVE MODEL FILES
# ============================================================

MODEL_FILES = {}

for model_name, filename in MODEL_FILENAMES.items():

    found_path = find_file(filename)

    if found_path is not None:
        MODEL_FILES[model_name] = found_path


# ============================================================
# LOAD PREPROCESSOR / SCALER IF AVAILABLE
# ============================================================

PREPROCESSOR_FILE = find_file("preprocessor.pkl")
SCALER_FILE = find_file("scaler.pkl")


@st.cache_resource
def load_pickle(file_path):
    return joblib.load(file_path)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Controls")

st.sidebar.markdown(
    """
    **Application workflow**

    1. Upload test data
    2. Select a machine learning model
    3. View evaluation metrics
    4. Inspect confusion matrix
    5. View classification report
    6. Compare all trained models
    """
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Test Data (CSV)",
    type=["csv"]
)

available_models = [
    model_name
    for model_name in MODEL_NAMES
    if model_name in MODEL_FILES
]

if available_models:

    selected_model_name = st.sidebar.selectbox(
        "🤖 Select Machine Learning Model",
        available_models
    )

else:

    st.sidebar.error(
        "No trained model files were found in the repository."
    )

    selected_model_name = None


st.sidebar.markdown("---")

st.sidebar.info(
    "Upload the test_data.csv generated during the assignment. "
    "The target column should be named 'y'."
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

missing_models = [
    model_name
    for model_name in MODEL_NAMES
    if model_name not in MODEL_FILES
]


if missing_models:

    st.warning(
        "Some trained model files were not found: "
        + ", ".join(missing_models)
    )


# ============================================================
# IF NO TEST DATA
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="info-box">

        ### 👋 Welcome

        This application evaluates five classification models trained
        on the **Bank Marketing dataset**.

        The application provides:

        - Accuracy
        - AUC
        - Precision
        - Recall
        - F1 Score
        - Matthews Correlation Coefficient (MCC)
        - Confusion Matrix
        - Classification Report
        - Comparison of all five models

        Please upload the **test_data.csv** file from the sidebar
        to begin evaluation.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🤖 Models Used</div>',
        unsafe_allow_html=True
    )

    models_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "kNN",
            "Naive Bayes",
            "Random Forest"
        ],
        "Model Type": [
            "Linear Classifier",
            "Tree-Based Classifier",
            "Instance-Based Classifier",
            "Probabilistic Classifier",
            "Ensemble Classifier"
        ]
    })

    st.dataframe(
        models_df,
        use_container_width=True,
        hide_index=True
    )

    st.stop()


# ============================================================
# READ TEST DATA
# ============================================================

try:

    data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error("Unable to read the uploaded CSV file.")

    st.exception(e)

    st.stop()


# ============================================================
# VALIDATE TARGET COLUMN
# ============================================================

if "y" not in data.columns:

    st.error(
        "The uploaded CSV does not contain the required target column 'y'. "
        "Please upload the test_data.csv generated during the assignment."
    )

    st.stop()


# ============================================================
# DISPLAY TEST DATA
# ============================================================

st.markdown(
    '<div class="section-title">📁 Uploaded Test Data</div>',
    unsafe_allow_html=True
)

st.write(
    f"Dataset contains **{data.shape[0]} rows** and "
    f"**{data.shape[1]} columns**."
)

st.dataframe(
    data.head(10),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PREPARE X AND y
# ============================================================

X = data.drop(columns=["y"]).copy()

y_raw = data["y"].copy()


# ============================================================
# ROBUST TARGET CONVERSION
# Supports:
# yes/no
# 1/0
# True/False
# ============================================================

def convert_target(series):

    # If already numeric
    if pd.api.types.is_numeric_dtype(series):

        numeric_values = pd.to_numeric(
            series,
            errors="coerce"
        )

        return numeric_values

    # Convert safely to string
    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    target_mapping = {
        "no": 0,
        "yes": 1,
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "n": 0,
        "y": 1
    }

    converted = normalized.map(target_mapping)

    # If some values were not mapped, try numeric conversion
    missing_mask = converted.isna()

    if missing_mask.any():

        numeric_values = pd.to_numeric(
            normalized[missing_mask],
            errors="coerce"
        )

        converted.loc[missing_mask] = numeric_values

    return converted


y = convert_target(y_raw)


# ============================================================
# CHECK TARGET CONVERSION
# ============================================================

if y.isna().any():

    invalid_values = y_raw[y.isna()].unique()

    st.error(
        "The target column 'y' contains values that could not be "
        "converted to binary 0/1."
    )

    st.write("Unrecognized target values:")

    st.write(invalid_values)

    st.stop()


y = y.astype(int)


# ============================================================
# PREPROCESS TEST DATA
# ============================================================

try:

    if PREPROCESSOR_FILE is not None:

        preprocessor = load_pickle(PREPROCESSOR_FILE)

        X_processed = preprocessor.transform(X)

    else:

        # If no preprocessor is present, assume the uploaded
        # dataset is already in the trained feature representation.
        X_processed = X


    if SCALER_FILE is not None:

        scaler = load_pickle(SCALER_FILE)

        X_final = scaler.transform(X_processed)

    else:

        X_final = X_processed


except Exception as e:

    st.error(
        "Error while preprocessing the uploaded test data."
    )

    st.write(
        "Please ensure that the uploaded test dataset has the "
        "same feature columns used during model training."
    )

    st.exception(e)

    st.stop()


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name):

    model_file = MODEL_FILES.get(model_name)

    if model_file is None:

        raise FileNotFoundError(
            f"Model file for {model_name} was not found."
        )

    model = load_pickle(model_file)

    predictions = model.predict(X_final)

    # --------------------------------------------------------
    # Probability / decision score
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_final)[:, 1]

    elif hasattr(model, "decision_function"):

        probabilities = model.decision_function(X_final)

    else:

        probabilities = predictions

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    try:

        auc = roc_auc_score(
            y,
            probabilities
        )

    except Exception:

        auc = np.nan

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y,
        predictions
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["No", "Yes"],
        output_dict=True,
        zero_division=0
    )

    return {
        "model": model_name,
        "predictions": predictions,
        "probabilities": probabilities,
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "confusion_matrix": cm,
        "classification_report": report
    }


# ============================================================
# SELECTED MODEL EVALUATION
# ============================================================

if selected_model_name is None:

    st.error(
        "No trained model is available. "
        "Please check that the .pkl files are committed to GitHub."
    )

    st.stop()


try:

    selected_result = evaluate_model(
        selected_model_name
    )

except Exception as e:

    st.error(
        f"Unable to evaluate {selected_model_name}."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SUCCESS MESSAGE
# ============================================================

st.success(
    f"✅ {selected_model_name} evaluation completed successfully."
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📈 Model Performance</div>',
    unsafe_allow_html=True
)

st.caption(
    "The following metrics are calculated on the uploaded test dataset."
)


metric_columns = st.columns(6)

metrics = [
    ("Accuracy", selected_result["accuracy"]),
    ("AUC", selected_result["auc"]),
    ("Precision", selected_result["precision"]),
    ("Recall", selected_result["recall"]),
    ("F1 Score", selected_result["f1"]),
    ("MCC", selected_result["mcc"])
]


for column, (label, value) in zip(
    metric_columns,
    metrics
):

    with column:

        if pd.isna(value):

            display_value = "N/A"

        else:

            display_value = f"{value:.4f}"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{display_value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Confusion Matrix</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="comparison-note">

    The confusion matrix shows the number of correct and incorrect
    predictions for the **No** and **Yes** classes.

    - **True Negative (TN):** correctly predicted No
    - **False Positive (FP):** predicted Yes but actual No
    - **False Negative (FN):** predicted No but actual Yes
    - **True Positive (TP):** correctly predicted Yes

    </div>
    """,
    unsafe_allow_html=True
)


cm = selected_result["confusion_matrix"]


fig, ax = plt.subplots(
    figsize=(6.5, 5.5)
)

image = ax.imshow(
    cm,
    cmap="Blues"
)

ax.set_title(
    f"{selected_model_name} — Confusion Matrix",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "True Label"
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    ["No", "Yes"]
)

ax.set_yticklabels(
    ["No", "Yes"]
)


for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold"
        )


fig.colorbar(
    image,
    ax=ax
)

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=False
)

plt.close(fig)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.markdown(
    '<div class="section-title">📋 Classification Report</div>',
    unsafe_allow_html=True
)

st.caption(
    f"Detailed precision, recall and F1-score results for "
    f"{selected_model_name}."
)


report_df = pd.DataFrame(
    selected_result["classification_report"]
).transpose()


st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ============================================================
# COMPARISON OF ALL MODELS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Comparison of All Models</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="comparison-note">

    This table compares the performance of all five trained
    classification models on the **same uploaded test dataset**.

    Higher values generally indicate better performance for these
    metrics. However, different metrics emphasize different aspects
    of classification performance.

    Therefore, this table is presented for **model comparison**
    rather than declaring a single overall winner.

    </div>
    """,
    unsafe_allow_html=True
)


comparison_results = []


for model_name in MODEL_NAMES:

    if model_name not in MODEL_FILES:

        comparison_results.append({
            "ML Model": model_name,
            "Accuracy": np.nan,
            "AUC": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1 Score": np.nan,
            "MCC": np.nan
        })

        continue


    try:

        result = evaluate_model(
            model_name
        )

        comparison_results.append({
            "ML Model": model_name,
            "Accuracy": result["accuracy"],
            "AUC": result["auc"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1 Score": result["f1"],
            "MCC": result["mcc"]
        })

    except Exception:

        comparison_results.append({
            "ML Model": model_name,
            "Accuracy": np.nan,
            "AUC": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1 Score": np.nan,
            "MCC": np.nan
        })


comparison_df = pd.DataFrame(
    comparison_results
)


st.dataframe(
    comparison_df.round(4),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# COMPARISON INTERPRETATION
# NO "WINNER" DECLARATION
# ============================================================

st.markdown(
    '<div class="section-title">💡 Comparison Summary</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    The models show different strengths across the evaluation metrics.

    **Accuracy** represents the overall proportion of correctly
    classified observations, while **AUC** measures the ability of
    the model to distinguish between the two classes.

    **Precision** indicates the proportion of predicted positive
    observations that are actually positive, whereas **Recall**
    measures how many actual positive observations are correctly
    identified.

    **F1 Score** provides a balance between precision and recall,
    while **MCC** evaluates binary classification performance using
    all four categories of the confusion matrix.

    Hence, the appropriate model depends on the evaluation metric
    and the objective of the classification problem. No single
    model is declared as an overall winner here.
    """,
    unsafe_allow_html=True
)


# ============================================================
# SELECTED MODEL SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📝 Selected Model Summary</div>',
    unsafe_allow_html=True
)

st.info(
    f"The currently selected model is **{selected_model_name}**. "
    "Its metrics, confusion matrix and classification report above "
    "are calculated using the uploaded test dataset."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    'Machine Learning Assignment 2 | Bank Marketing Classification'
    '</div>',
    unsafe_allow_html=True
)
