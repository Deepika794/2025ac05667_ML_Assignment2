# Bank Marketing Classification

## Project Overview

This project implements a machine learning classification system using the Bank Marketing dataset. The objective is to predict whether a customer will subscribe to a term deposit.

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

The models were compared based on all six evaluation metrics. Logistic Regression showed strong overall performance, while Decision Tree achieved stronger Recall, F1 Score and MCC. Random Forest achieved the highest AUC.

## Streamlit Application

The Streamlit application allows users to:

- Upload a test CSV file
- Select a machine learning model
- Generate predictions
- View evaluation metrics
- View the confusion matrix
- View the classification report

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