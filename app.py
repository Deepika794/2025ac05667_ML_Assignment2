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
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #555555;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    margin-top: 25px;
    margin-bottom: 12px;
}

.metric-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #d9e2f3;
    background-color: #f8fbff;
    text-align: center;
}

.metric-label {
    font-size: 14px;
    color: #555555;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #1f5fbf;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #f4f7fb;
    border-left: 5px solid #4a90e2;
    margin-bottom: 20px;
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
    'Machine Learning classification of whether a bank customer subscribes '
    'to a term deposit.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}


# ============================================================
# LOAD PREPROCESSOR AND SCALER
# ============================================================

@st.cache_resource
def load_preprocessor():
    return joblib.load("model/preprocessor.pkl")


@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.pkl")


@st.cache_resource
def load_model(model_file):
    return joblib.load(model_file)


preprocessor = load_preprocessor()
scaler = load_scaler()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Controls")

st.sidebar.markdown(
    """
    **Workflow**

    1. Upload test data
    2. Select a machine learning model
    3. View evaluation metrics
    4. Inspect confusion matrix
    5. Compare all models
    """
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=["csv"]
)

selected_model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(MODEL_FILES.keys())
)

st.sidebar.markdown("---")

st.sidebar.info(
    "The uploaded CSV should contain the same feature columns used "
    "during model training. The target column is expected to be 'y'."
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Please upload the test_data.csv file from the sidebar "
        "to start the evaluation."
    )

    st.markdown("### About this application")

    st.write(
        "This application evaluates five classification models trained "
        "on the Bank Marketing dataset."
    )

    st.write(
        "The application uses the saved preprocessing objects and trained "
        "models to generate predictions on the uploaded test data."
    )

    st.markdown("### Models implemented")

    models_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "kNN",
            "Naive Bayes",
            "Random Forest"
        ],
        "Type": [
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
# READ UPLOADED DATA
# ============================================================

try:

    data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"Unable to read the uploaded CSV file: {e}")
    st.stop()


# ============================================================
# DATA VALIDATION
# ============================================================

if "y" not in data.columns:

    st.error(
        "The uploaded CSV must contain the target column 'y'. "
        "Please upload the test_data.csv generated during the assignment."
    )

    st.stop()


# ============================================================
# DISPLAY DATA
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
    use_container_width=True
)


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

X = data.drop(columns=["y"])
y = data["y"].copy()


# Convert target to numerical representation
# no = 0, yes = 1

if y.dtype == "object":

    y = (
        y.astype(str)
        .str.lower()
        .map({
            "no": 0,
            "yes": 1
        })
    )

y = y.astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})


# ============================================================
# PREPROCESS TEST DATA
# ============================================================

try:

    X_processed = preprocessor.transform(X)

    X_scaled = scaler.transform(X_processed)

except Exception as e:

    st.error(
        "Error while preprocessing the uploaded data. "
        "Please make sure the CSV has the same feature columns "
        "as the training dataset."
    )

    st.exception(e)

    st.stop()


# ============================================================
# FUNCTION TO GET PREDICTIONS AND PROBABILITIES
# ============================================================

def evaluate_model(model_name):

    model = load_model(MODEL_FILES[model_name])

    predictions = model.predict(X_scaled)

    # Probability for positive class
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_scaled)[:, 1]

    elif hasattr(model, "decision_function"):

        probabilities = model.decision_function(X_scaled)

    else:

        probabilities = predictions

    accuracy = accuracy_score(y, predictions)

    try:
        auc = roc_auc_score(y, probabilities)
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
        predictions
    )

    report = classification_report(
        y,
        predictions,
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
# EVALUATE SELECTED MODEL
# ============================================================

try:

    selected_result = evaluate_model(
        selected_model_name
    )

except Exception as e:

    st.error(
        "The selected model could not be evaluated."
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

for column, (label, value) in zip(metric_columns, metrics):

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

st.write(
    "The confusion matrix shows the number of correct and incorrect "
    "predictions for the two classes."
)

cm = selected_result["confusion_matrix"]

fig, ax = plt.subplots(figsize=(6, 5))

image = ax.imshow(cm)

ax.set_title(
    f"{selected_model_name} — Confusion Matrix",
    fontsize=15
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["No", "Yes"])
ax.set_yticklabels(["No", "Yes"])

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=16
        )

fig.colorbar(image, ax=ax)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.markdown(
    '<div class="section-title">📋 Classification Report</div>',
    unsafe_allow_html=True
)

report_df = pd.DataFrame(
    selected_result["classification_report"]
).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">🏆 Comparison of All Models</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    This table compares the performance of all five trained
    classification models on the **same uploaded test dataset**.

    A higher value is generally better for all six metrics.
    The model with the strongest overall performance can therefore
    be considered the overall winner for this test dataset.
    """
)

comparison_results = []

for model_name in MODEL_FILES.keys():

    try:

        result = evaluate_model(model_name)

        comparison_results.append({
            "ML Model": model_name,
            "Accuracy": result["accuracy"],
            "AUC": result["auc"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1 Score": result["f1"],
            "MCC": result["mcc"]
        })

    except Exception as e:

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
# OVERALL WINNER
# ============================================================

if not comparison_df.empty:

    winner_index = comparison_df["F1 Score"].idxmax()

    winner_name = comparison_df.loc[
        winner_index,
        "ML Model"
    ]

    winner_f1 = comparison_df.loc[
        winner_index,
        "F1 Score"
    ]

    st.success(
        f"🏆 Overall winner based on F1 Score: "
        f"**{winner_name}** "
        f"(F1 Score = {winner_f1:.4f})"
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-title">💡 Interpretation</div>',
    unsafe_allow_html=True
)

st.write(
    f"""
    The selected model is **{selected_model_name}**.

    - **Accuracy** indicates the overall proportion of correct predictions.
    - **AUC** measures the model's ability to distinguish between the two classes.
    - **Precision** indicates how many predicted positive cases were actually positive.
    - **Recall** indicates how many actual positive cases were correctly identified.
    - **F1 Score** provides a balance between precision and recall.
    - **MCC** measures the quality of binary classification while considering
      all four confusion-matrix categories.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Machine Learning Assignment 2 | Bank Marketing Classification | "
    "Logistic Regression, Decision Tree, kNN, Naive Bayes and Random Forest"
)
