import streamlit as st
import pickle
import numpy as np

# -------------------------------------------------------
# LOAD MODELS & SCALER
# -------------------------------------------------------
with open("attrition_model.pkl", "rb") as f:
    attrition_model = pickle.load(f)

with open("performance_model.pkl", "rb") as f:
    performance_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------------------------------------
# TRAINED FEATURE ORDER (VERY IMPORTANT)
# -------------------------------------------------------
feature_order = [
    'Age', 'Gender_Encoded', 'MonthlyIncome', 'JobLevel', 'JobSatisfaction',
    'YearsAtCompany', 'OverTime_Encoded', 'DistanceFromHome', 'TotalWorkingYears',
    'YearsInCurrentRole', 'EnvironmentSatisfaction', 'WorkLifeBalance',
    'RelationshipSatisfaction', 'JobInvolvement', 'StockOptionLevel'
]

# -------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    [ "🔮 Attrition Prediction", "📊 Performance Prediction"]
)




# -------------------------------------------------------
# ATTRITION PREDICTION PAGE
# -------------------------------------------------------
if page == "🔮 Attrition Prediction":
    st.title("🔮 Employee Attrition Prediction")

    st.markdown("Enter employee data below:")

    # Collect Inputs
    Age = st.number_input("Age", 18, 60, 30)
    Gender_Encoded = st.selectbox("Gender", ["Male", "Female"])
    Gender_Encoded = 1 if Gender_Encoded == "Male" else 0

    MonthlyIncome = st.number_input("Monthly Income", 1000, 500000, 25000)
    JobLevel = st.slider("Job Level", 1, 5, 2)
    JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    YearsAtCompany = st.number_input("Years at Company", 0, 40, 3)

    OverTime_Encoded = st.selectbox("OverTime", ["Yes", "No"])
    OverTime_Encoded = 1 if OverTime_Encoded == "Yes" else 0

    DistanceFromHome = st.number_input("Distance From Home", 1, 50, 10)
    TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 5)
    YearsInCurrentRole = st.number_input("Years in Current Role", 0, 18, 3)
    EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    WorkLifeBalance = st.slider("Work Life Balance", 1, 4, 3)
    RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
    JobInvolvement = st.slider("Job Involvement", 1, 4, 3)
    StockOptionLevel = st.slider("Stock Option Level", 0, 3, 1)

    # Build Feature Vector
    user_data = [
        Age, Gender_Encoded, MonthlyIncome, JobLevel, JobSatisfaction,
        YearsAtCompany, OverTime_Encoded, DistanceFromHome, TotalWorkingYears,
        YearsInCurrentRole, EnvironmentSatisfaction, WorkLifeBalance,
        RelationshipSatisfaction, JobInvolvement, StockOptionLevel
    ]

    # Convert to numpy and reshape
    user_data_array = np.array(user_data).reshape(1, -1)

    # Scale with previously fitted scaler
    scaled_data = scaler.transform(user_data_array)

    # Predict
    if st.button("Predict Attrition"):
        prediction = attrition_model.predict(scaled_data)[0]

        if prediction == 1:
            st.error("⚠ The model predicts the employee is likely to ATTRITE.")
        else:
            st.success("✅ The model predicts the employee will STAY.")


# -------------------------------------------------------
# PERFORMANCE PREDICTION PAGE
# -------------------------------------------------------
# -------------------------------------------------------
# PERFORMANCE PREDICTION PAGE
# -------------------------------------------------------
elif page == "📊 Performance Prediction":
    st.title("📊 Employee Performance Prediction")

    st.markdown("Enter employee data below:")

    # SAME 15 FEATURES
    Age = st.number_input("Age", 18, 60, 30)
    Gender_Encoded = st.selectbox("Gender", ["Male", "Female"])
    Gender_Encoded = 1 if Gender_Encoded == "Male" else 0

    MonthlyIncome = st.number_input("Monthly Income", 1000, 500000, 25000)
    JobLevel = st.slider("Job Level", 1, 5, 2)
    JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    YearsAtCompany = st.number_input("Years at Company", 0, 40, 3)

    OverTime_Encoded = st.selectbox("OverTime", ["Yes", "No"])
    OverTime_Encoded = 1 if OverTime_Encoded == "Yes" else 0

    DistanceFromHome = st.number_input("Distance From Home", 1, 50, 10)
    TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 5)
    YearsInCurrentRole = st.number_input("Years in Current Role", 0, 18, 3)
    EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    WorkLifeBalance = st.slider("Work Life Balance", 1, 4, 3)
    RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
    JobInvolvement = st.slider("Job Involvement", 1, 4, 3)
    StockOptionLevel = st.slider("Stock Option Level", 0, 3, 1)

    # Build user input list
    user_data = [
        Age, Gender_Encoded, MonthlyIncome, JobLevel, JobSatisfaction,
        YearsAtCompany, OverTime_Encoded, DistanceFromHome, TotalWorkingYears,
        YearsInCurrentRole, EnvironmentSatisfaction, WorkLifeBalance,
        RelationshipSatisfaction, JobInvolvement, StockOptionLevel
    ]

    user_data_array = np.array(user_data).reshape(1, -1)
    scaled_data = scaler.transform(user_data_array)

    # Predict
    if st.button("Predict Performance Rating"):
        rating = performance_model.predict(scaled_data)[0]

        # ----------------------------------------------------
        # 🔵 FORCE MODEL OUTPUT TO ONLY 3 OR 4
        # ----------------------------------------------------
        if rating <= 2:
            rating = 3
        else:
            rating = 4
        # ----------------------------------------------------

        st.info(f"⭐ The predicted performance rating is: **{rating}**")
