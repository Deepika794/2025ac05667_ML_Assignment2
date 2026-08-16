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


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM STYLING
# ==========================================================

st.markdown(
    """
    <style>

    /* Main page background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #eef3ff;
        border-right: 1px solid #d9e2f3;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(90deg, #3157c7, #6b5bd6);
        padding: 28px 32px;
        border-radius: 14px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 4px 12px rgba(49, 87, 199, 0.18);
    }

    .main-header h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 700;
    }

    .main-header p {
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 15px;
        opacity: 0.95;
    }

    /* Section headings */
    .section-title {
        color: #263859;
        font-size: 22px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Metric cards */
    .metric-card {
        background-color: white;
        border: 1px solid #dce4f2;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(40, 60, 100, 0.08);
        margin-bottom: 15px;
    }

    .metric-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #3157c7;
        font-size: 28px;
        font-weight: 700;
    }

    /* Info box */
    .info-box {
        background-color: #eef3ff;
        border-left: 5px solid #3157c7;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 18px;
        color: #263859;
    }

    /* Success box */
    .success-box {
        background-color: #eefaf4;
        border-left: 5px solid #2e9d67;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 18px;
        color: #245c43;
    }

    /* Sidebar heading */
    .sidebar-title {
        color: #3157c7;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #7b8798;
        font-size: 13px;
        margin-top: 35px;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# LOAD PREPROCESSING OBJECTS
# ==========================================================

preprocessor = joblib.load("model/preprocessor.pkl")
scaler = joblib.load("model/scaler.pkl")


# ==========================================================
# LOAD TRAINED MODELS
# ==========================================================

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


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🏦 Bank Marketing ML</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Prediction and evaluation using trained machine learning models."
    )

    st.divider()

    st.subheader("📂 Test Data")

    uploaded_file = st.file_uploader(
        "Upload test CSV file",
        type=["csv"]
    )

    st.divider()

    st.subheader("🤖 Select Model")

    selected_model = st.selectbox(
        "Choose a machine learning model",
        list(models.keys())
    )

    st.divider()

    st.caption(
        "Five trained classification models are available."
    )


# ==========================================================
# MAIN HEADER
# ==========================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🏦 Bank Marketing Classification</h1>
        <p>
            Predict whether a customer will subscribe to a term deposit
            using trained machine learning models.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# APPLICATION INFORMATION
# ==========================================================

st.markdown(
    f"""
    <div class="info-box">
        <b>Selected Model:</b> {selected_model}<br>
        Upload the test dataset from the sidebar to generate predictions
        and evaluate model performance.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# WAIT FOR FILE UPLOAD
# ==========================================================

if uploaded_file is None:

    st.info(
        "👈 Please upload the test_data.csv file using the sidebar."
    )

    st.markdown(
        """
        ### How to use this application

        1. Upload `test_data.csv` from the sidebar.
        2. Select one of the five machine learning models.
        3. View predictions and model performance.
        4. Compare Accuracy, AUC, Precision, Recall, F1 Score and MCC.
        """,
    )

else:

    # ======================================================
    # READ DATA
    # ======================================================

    data = pd.read_csv(uploaded_file)

    st.markdown(
        '<div class="section-title">📋 Uploaded Test Data</div>',
        unsafe_allow_html=True
    )

    st.success(
        f"Test data uploaded successfully — {data.shape[0]} rows and "
        f"{data.shape[1]} columns."
    )

    with st.expander("View uploaded data", expanded=False):
        st.dataframe(
            data.head(10),
            use_container_width=True
        )


    # ======================================================
    # CHECK TARGET COLUMN
    # ======================================================

    if "y" not in data.columns:

        st.error(
            "The uploaded CSV must contain the target column 'y' "
            "for model evaluation."
        )

    else:

        # ==================================================
        # SEPARATE FEATURES AND TARGET
        # ==================================================

        X = data.drop("y", axis=1)
        y = data["y"]

        # Convert target labels to numeric
        y = y.map({
            "no": 0,
            "yes": 1
        })


        # ==================================================
        # APPLY SAME PREPROCESSING USED DURING TRAINING
        # ==================================================

        X_processed = preprocessor.transform(X)

        X_scaled = scaler.transform(X_processed)


        # ==================================================
        # SELECT MODEL
        # ==================================================

        model = models[selected_model]


        # ==================================================
        # MAKE PREDICTIONS
        # ==================================================

        y_pred = model.predict(X_scaled)

        y_prob = model.predict_proba(X_scaled)[:, 1]


        # ==================================================
        # CALCULATE METRICS
        # ==================================================

        accuracy = accuracy_score(y, y_pred)

        auc = roc_auc_score(
            y,
            y_prob
        )

        precision = precision_score(
            y,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y,
            y_pred
        )


        # ==================================================
        # MODEL PERFORMANCE
        # ==================================================

        st.markdown(
            '<div class="section-title">📊 Model Performance</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="success-box">
                <b>{selected_model}</b> predictions completed successfully.
            </div>
            """,
            unsafe_allow_html=True
        )


        # First row of metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Accuracy</div>
                    <div class="metric-value">{accuracy:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">AUC</div>
                    <div class="metric-value">{auc:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Precision</div>
                    <div class="metric-value">{precision:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # Second row of metrics
        col4, col5, col6 = st.columns(3)

        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Recall</div>
                    <div class="metric-value">{recall:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">F1 Score</div>
                    <div class="metric-value">{f1:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col6:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">MCC</div>
                    <div class="metric-value">{mcc:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ==================================================
        # DETAILED EVALUATION
        # ==================================================

        st.markdown(
            '<div class="section-title">📈 Detailed Evaluation</div>',
            unsafe_allow_html=True
        )

        tab1, tab2 = st.tabs(
            [
                "🔲 Confusion Matrix",
                "📋 Classification Report"
            ]
        )


        # ==================================================
        # CONFUSION MATRIX
        # ==================================================

        with tab1:

            cm = confusion_matrix(
                y,
                y_pred
            )

            cm_df = pd.DataFrame(
                cm,
                index=[
                    "Actual No",
                    "Actual Yes"
                ],
                columns=[
                    "Predicted No",
                    "Predicted Yes"
                ]
            )

            st.dataframe(
                cm_df,
                use_container_width=True
            )


        # ==================================================
        # CLASSIFICATION REPORT
        # ==================================================

        with tab2:

            report = classification_report(
                y,
                y_pred,
                target_names=[
                    "No",
                    "Yes"
                ],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(
                report_df.round(4),
                use_container_width=True
            )


        # ==================================================
        # FOOTER
        # ==================================================

        st.markdown(
            """
            <div class="footer">
                Bank Marketing Classification | Machine Learning Assignment
            </div>
            """,
            unsafe_allow_html=True
        )
