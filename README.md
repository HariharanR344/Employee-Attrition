# Employee-Attrition

Employee Attrition & Performance Prediction System
This project is an end-to-end Machine Learning + Streamlit application designed to help organizations analyze employee-related data and predict:
Employee Attrition (Whether an employee is likely to leave the company)
Employee Performance Rating (Predicting performance score based on HR metrics)
The system includes data preprocessing, SQL integration, feature engineering, model development, evaluation, and deployment via Streamlit.

Project Workflow
1. Data Loading & Exploration
Imported dataset (CSV) using pandas.
Verified dataset structure (shape, columns).
Checked for missing values (found none).
Explored data types and statistical summaries.

2. SQL Database Integration
Connected to MySQL using Python.
Created a new database Employee_Attrition.
Created table Details and inserted all rows.
Verified insertion using SELECT queries.

3. Data Preprocessing
 Outlier Detection & Handling
Used IQR (Interquartile Range) technique.
Capped extreme values to improve model performance.

Label Encoding
converted categorical columns (Gender, OverTime, etc.) into numerical format using LabelEncoder.
Handling Imbalanced Data
Used SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset for attrition prediction.

Machine Learning Models
1. Employee Attrition Prediction
Selected relevant HR-related features that influence attrition.
Performed:
Train–test split
StandardScaler
Trained classification models (Logistic Regression / Decision Tree / Random Forest).
Evaluated using:
Accuracy
Confusion Matrix
Classification Report

2. Employee Performance Rating Prediction
Selected 15 key features (Age, JobLevel, JobSatisfaction, etc.).
Scaled features using StandardScaler.
Trained model to predict performance rating.
Evaluated using:
Accuracy
Precision, Recall, F1-Score

Streamlit Web Application
A user-friendly Streamlit app (Python .py file) was developed to allow HR teams to perform predictions interactively.
Features:
Sidebar for navigation:
Employee Attrition Prediction
Employee Performance Rating Prediction
Input form for 15+ HR attributes.
Real-time prediction output.
Models loaded from pickle files.
Clean and responsive UI.
