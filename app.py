import streamlit as st
import pandas as pd
import joblib

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

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Title and description
# --------------------------------------------------

st.title("Bank Marketing Classification App")

st.write(
    "This application uses trained machine learning models "
    "to predict whether a customer will subscribe to a term deposit."
)

# --------------------------------------------------
# Load preprocessing objects
# --------------------------------------------------

preprocessor = joblib.load("model/preprocessor.pkl")
scaler = joblib.load("model/scaler.pkl")

# --------------------------------------------------
# Load trained models
# --------------------------------------------------

models = {
    "Logistic Regression": joblib.load(
        "model/logistic_regression.pkl"
    ),
    "Decision Tree": joblib.load(
        "model/decision_tree.pkl"
    ),
    "KNN": joblib.load(
        "model/knn.pkl"
    ),
    "Naive Bayes": joblib.load(
        "model/naive_bayes.pkl"
    ),
    "Random Forest": joblib.load(
        "model/random_forest.pkl"
    )
}

# --------------------------------------------------
# File upload
# --------------------------------------------------

st.header("Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

# --------------------------------------------------
# Model selection
# --------------------------------------------------

selected_model = st.selectbox(
    "Select a Machine Learning Model",
    list(models.keys())
)

# --------------------------------------------------
# Prediction and evaluation
# --------------------------------------------------

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    # Check whether target column is present
    if "y" not in data.columns:

        st.error(
            "The uploaded CSV must contain the target column 'y' "
            "for evaluation."
        )

    else:

        # Separate features and target
        X = data.drop("y", axis=1)
        y = data["y"]

        # Convert target to numeric
        y = y.map({"no": 0, "yes": 1})

        # Apply the same preprocessing used during training
        X_processed = preprocessor.transform(X)

        # Apply the same scaling used during training
        X_scaled = scaler.transform(X_processed)

        # Get selected trained model
        model = models[selected_model]

        # Make predictions
        y_pred = model.predict(X_scaled)

        # Get probabilities for AUC
        y_prob = model.predict_proba(X_scaled)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y, y_pred)

        # --------------------------------------------------
        # Display metrics
        # --------------------------------------------------

        st.header("Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------

        st.header("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"]
        )

        st.dataframe(cm_df)

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------

        st.header("Classification Report")

        report = classification_report(
            y,
            y_pred,
            target_names=["No", "Yes"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df.round(4))

else:

    st.info(
        "Please upload the test_data.csv file to view predictions "
        "and model performance."
    )