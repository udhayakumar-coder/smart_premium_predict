import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Smart Premium Predictor",
    page_icon="💰",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
    }

    .title-box {
        background: linear-gradient(90deg, #4b6cb7, #182848);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }

    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: white;
        font-size: 18px;
        height: 3em;
        border-radius: 12px;
        border: none;
    }

    .result-box {
        background: linear-gradient(90deg, #ff9966, #ff5e62);
        padding: 15px;
        border-radius: 12px;
        color: white;
        font-size: 22px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Cache Model & Pipeline
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("smart_premium_model.pkl")

@st.cache_resource
def load_pipeline():
    return joblib.load("preprocessing_pipeline.pkl")

model = load_model()
pipeline = load_pipeline()

# -------------------------------
# Header with Image
# -------------------------------
st.image(
    "https://cdn-icons-png.flaticon.com/512/942/942748.png",
    width=100
)

st.markdown(
    """
    <div class="title-box">
        <h1>💰 Smart Insurance Premium Prediction</h1>
        <p>📊 Predict insurance cost using AI</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# -------------------------------
# Input Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Personal Details")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    Age = st.number_input("🎂 Age", 18, 100, 30)
    Gender = st.selectbox("🚻 Gender", ["Male", "Female"])
    Marital_Status = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced"])
    Education_Level = st.selectbox("🎓 Education Level", ["High School", "Graduate", "Post Graduate"])
    Occupation = st.selectbox("💼 Occupation", ["Salaried", "Self-Employed", "Business", "Unemployed"])
    Annual_Income = st.number_input("💵 Annual Income", 100000, 5000000, 500000)
    Credit_Score = st.number_input("📈 Credit Score", 300, 900, 650)
    Health_Score = st.slider("❤️ Health Score", 0, 100, 70)
    Smoking_Status = st.selectbox("🚬 Smoking Status", ["Yes", "No"])
    Exercise_Frequency = st.selectbox("🏃 Exercise Frequency", ["None", "Low", "Medium", "High"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 🛡️ Policy Details")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    Policy_Type = st.selectbox("📄 Policy Type", ["Health", "Life", "Vehicle", "Property"])
    Insurance_Duration = st.number_input("⏳ Insurance Duration (Years)", 1, 30, 5)
    Policy_Start_Date = st.date_input("📅 Policy Start Date")
    Property_Type = st.selectbox("🏠 Property Type", ["Owned", "Rented"])
    Vehicle_Age = st.number_input("🚗 Vehicle Age", 0, 30, 5)
    Location = st.selectbox("📍 Location", ["Urban", "Semi-Urban", "Rural"])
    Number_of_Dependents = st.number_input("👨‍👩‍👧 Dependents", 0, 10, 1)
    Previous_Claims = st.number_input("📑 Previous Claims", 0, 20, 0)
    Customer_Feedback = st.selectbox(
        "⭐ Customer Feedback",
        ["Poor", "Average", "Good", "Excellent"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Prediction
# -------------------------------
center = st.columns([1, 2, 1])[1]

with center:
    if st.button("🔮 Predict Premium", use_container_width=True):
        input_df = pd.DataFrame([{
            "Gender": Gender,
            "Policy Type": Policy_Type,
            "Occupation": Occupation,
            "Insurance Duration": Insurance_Duration,
            "Age": Age,
            "Health Score": Health_Score,
            "Location": Location,
            "Credit Score": Credit_Score,
            "Marital Status": Marital_Status,
            "Number of Dependents": Number_of_Dependents,
            "Education Level": Education_Level,
            "Smoking Status": Smoking_Status,
            "Property Type": Property_Type,
            "Policy Start Date": Policy_Start_Date,
            "Vehicle Age": Vehicle_Age,
            "Customer Feedback": Customer_Feedback,
            "Exercise Frequency": Exercise_Frequency,
            "Previous Claims": Previous_Claims,
            "Annual Income": Annual_Income
        }])

        X = pipeline.transform(input_df)
        prediction = model.predict(X)

        st.markdown(
            f"""
            <div class="result-box">
                💰 Predicted Insurance Premium: <br><b>{prediction[0]:,.2f}</b>
            </div>
            """,
            unsafe_allow_html=True
        )
