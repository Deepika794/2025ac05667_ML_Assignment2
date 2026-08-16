# Bank Marketing Classification

## Project Overview

This project implements a machine learning classification system using the Bank Marketing dataset. The objective is to predict whether a customer will subscribe to a term deposit.

## Project Links

- **GitHub Repository:** https://github.com/Deepika794/2025ac05667_ML_Assignment2/tree/main
- **Live Streamlit Application:** https://2025ac05667mlassignment2-gftjhaiv5subzbaw8xq82v.streamlit.app/

## Dataset

The dataset contains customer and campaign-related information such as age, job, marital status, education, balance, housing loan, personal loan, contact type, campaign information and previous campaign outcomes.

The dataset contains 4521 instances and 16 input features.

## Machine Learning Models

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest

## Preprocessing

Categorical features were encoded using a ColumnTransformer. Numerical features were scaled using StandardScaler.
The data was divided into training and testing sets using stratified sampling to maintain the target class distribution.
The fitted preprocessing objects were saved and reused during model evaluation and Streamlit deployment to ensure that the test data receives the same transformations as the training data.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Confusion matrices and classification reports were also generated.

## Model Comparison
The five classification models were evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8917 | 0.8905 | 0.5536 | 0.2981 | 0.3875 | 0.3532 |
| Decision Tree | 0.8586 | 0.6942 | 0.4032 | 0.4808 | 0.4386 | 0.3602 |
| KNN | 0.8751 | 0.7677 | 0.3902 | 0.1538 | 0.2207 | 0.1881 |
| Naive Bayes | 0.8210 | 0.7885 | 0.3092 | 0.4519 | 0.3672 | 0.2737 |
| Random Forest | 0.8884 | 0.8939 | 0.5306 | 0.2500 | 0.3399 | 0.3119 |

### Model-wise Observations

- **Logistic Regression:** Achieved the highest Accuracy (0.8917) and Precision (0.5536), along with a strong AUC (0.8905). Its Recall (0.2981) was comparatively lower.
- **Decision Tree:** Achieved the highest Recall (0.4808), F1 Score (0.4386) and MCC (0.3602). This indicates a better balance between Precision and Recall among the evaluated models, although its Accuracy and AUC were comparatively lower.
- **KNN:** Achieved an Accuracy of 0.8751, but its Recall (0.1538) and F1 Score (0.2207) were comparatively low, indicating weaker identification of positive cases.
- **Naive Bayes:** Achieved relatively high Recall (0.4519), but its Accuracy (0.8210), Precision (0.3092), F1 Score (0.3672) and MCC (0.2737) were comparatively lower.
- **Random Forest:** Achieved the highest AUC (0.8939) and strong Accuracy (0.8884) and Precision (0.5306), indicating good discrimination performance. However, its Recall (0.2500) and F1 Score (0.3399) were lower than those of Decision Tree.

### Overall Winner

Decision Tree can be considered the preferred model when F1 Score and Recall are given higher importance, as it achieved the highest F1 Score (0.4386) and Recall (0.4808) among the evaluated models.
However, no single model performs best across all evaluation metrics. Random Forest achieved the highest AUC (0.8939), while Logistic Regression achieved the highest Accuracy (0.8917) and Precision (0.5536).
Therefore, the preferred model depends on the evaluation metric and the objective of the classification problem.

## Streamlit Application
The Streamlit application allows users to:
- Upload a test CSV file
- Select a machine learning model
- Generate predictions
- View evaluation metrics
- View the confusion matrix
- View the classification report
- Compare the performance of all five machine learning models

## Project Structure

2025ac05667_ML_Assignment2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── preprocessor.pkl
    ├── scaler.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl

## How to Run

Install the required libraries:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py
