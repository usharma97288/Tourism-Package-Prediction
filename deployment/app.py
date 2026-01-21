
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from huggingface_hub import hf_hub_download

HF_MODEL_REPO_ID = "usharma97288/Tourism-Package-Prediction-Model"
HF_MODEL_FILENAME = "tourism_xgb_model_grid_search.joblib"

SCALER_MEANS = [37.231831395348834, 1.6632751937984496, 15.584786821705427, 2.9493701550387597, 3.7415213178294575, 3.578488372093023, 3.295300387596899, 0.29530038759689925, 3.0608042635658914, 0.6121608527131783, 1.22359496124031, 0.9840116279069767, 23178.46414728682, 4.1729651162790695, 1.022044573643411, 0.27349806201550386]
SCALER_STDS = [9.173409306940236, 0.9205282281614273, 8.397124897469508, 0.7187309758862669, 1.0066642336899188, 0.794934570134564, 1.8560756108879786, 0.4561776722747622, 1.3628993733938415, 0.487257573690397, 0.8525812736576291, 1.019070194775755, 4506.068729559826, 1.4054936947261303, 0.6638130442399522, 0.4462973988421829]
SCALED_FEATURE_NAMES = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome', 'TotalVisitors', 'AgeGroup', 'IncomeGroup']
EXPECTED_FEATURE_ORDER = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome', 'TotalVisitors', 'AgeGroup', 'IncomeGroup', 'TypeofContact_Self Enquiry', 'Occupation_Large Business', 'Occupation_Salaried', 'Occupation_Small Business', 'Gender_Male', 'ProductPitched_Deluxe', 'ProductPitched_King', 'ProductPitched_Standard', 'ProductPitched_Super Deluxe', 'MaritalStatus_Married', 'MaritalStatus_Single', 'MaritalStatus_Unmarried']

DESIGNATION_ORDER = ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP']
AGEGROUP_ORDER = ['18-30', '31-45', '46-60', '60+']
INCOMEGROUP_ORDER = ['Low', 'Medium', 'High']

NOMINAL_COLS = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus"]

@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=HF_MODEL_FILENAME)
    return joblib.load(path)

def preprocess_input(df):
    df = df.copy()

    df["TotalVisitors"] = df["NumberOfPersonVisiting"] + df["NumberOfChildrenVisiting"]

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[18, 30, 45, 60, float("inf")],
        labels=["18-30", "31-45", "46-60", "60+"],
        right=False
    )

    df["IncomeGroup"] = pd.cut(
        df["MonthlyIncome"],
        bins=[0, 25000, 50000, float("inf")],
        labels=["Low", "Medium", "High"],
        right=False
    )

    for col in ["Passport", "OwnCar"]:
        df[col] = df[col].astype(int)

    encoder = OrdinalEncoder(
        categories=[DESIGNATION_ORDER, AGEGROUP_ORDER, INCOMEGROUP_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    df[["Designation", "AgeGroup", "IncomeGroup"]] = encoder.fit_transform(
        df[["Designation", "AgeGroup", "IncomeGroup"]]
    )

    df = pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=True)

    for col in EXPECTED_FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0

    scaler = StandardScaler()
    scaler.mean_ = np.array(SCALER_MEANS)
    scaler.scale_ = np.array(SCALER_STDS)

    df[SCALED_FEATURE_NAMES] = scaler.transform(df[SCALED_FEATURE_NAMES])

    return df[EXPECTED_FEATURE_ORDER]

st.set_page_config(layout="wide")
st.title("Wellness Tourism Package Prediction")

model = load_model()

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Large Business", "Small Business"])
    monthly_income = st.number_input("Monthly Income", 0, 200000, 30000)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    designation = st.selectbox("Designation", DESIGNATION_ORDER)

with col2:
    num_adults = st.slider("Adults Visiting", 0, 6, 2)
    num_children = st.slider("Children Visiting", 0, 4, 0)
    preferred_star = st.slider("Preferred Property Star", 3, 5, 4)
    num_trips = st.slider("Number of Trips", 0, 20, 5)
    passport = st.checkbox("Has Passport")
    own_car = st.checkbox("Owns Car")

with col3:
    typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    product = st.selectbox("Product Pitched", ["Standard", "Deluxe", "Super Deluxe", "King"])
    pitch_score = st.slider("Pitch Satisfaction", 1, 5, 3)
    followups = st.slider("Followups", 0, 6, 2)
    duration = st.slider("Pitch Duration", 0, 60, 15)

if st.button("Predict"):
    input_df = pd.DataFrame([{ 
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_adults,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
        "PitchSatisfactionScore": pitch_score,
        "ProductPitched": product,
        "NumberOfFollowups": followups,
        "DurationOfPitch": duration,
    }])

    processed = preprocess_input(input_df)
    prob = model.predict_proba(processed)[0, 1]

    st.success(f"Purchase Probability: {prob:.2%}")
