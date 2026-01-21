
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import numpy as np

# --- Configuration ---
#HF_MODEL_REPO_ID = "usharma97288/Tourism-Package-Prediction-Prj"
HF_MODEL_REPO_ID = "usharma97288/Tourism-Package-Prediction-Model"
HF_MODEL_FILENAME = "models/tourism_xgb_model.joblib"

# --- Hardcoded Scaler Parameters and Feature Order (from training) ---
SCALER_MEANS = [37.231831395348834, 1.6632751937984496, 15.584786821705427, 2.9493701550387597, 3.7415213178294575, 3.578488372093023, 3.295300387596899, 0.29530038759689925, 3.0608042635658914, 0.6121608527131783, 1.22359496124031, 0.9840116279069767, 23178.46414728682, 4.1729651162790695, 1.0133236434108528, 1.0]
SCALER_STDS = [9.173409306940236, 0.9205282281614273, 8.397124897469508, 0.7187309758862669, 1.0066642336899188, 0.794934570134564, 1.8560756108879786, 0.4561776722747622, 1.3628993733938415, 0.487257573690397, 0.8525812736576291, 1.019070194775755, 4506.068729559826, 1.4054936947261303, 0.6640453308747023, 0.816496580927726]
SCALED_FEATURE_NAMES = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome', 'TotalVisitors', 'AgeGroup', 'IncomeGroup']
EXPECTED_FEATURE_ORDER = ['Passport', 'Designation', 'MaritalStatus_Single', 'NumberOfFollowups', 'MaritalStatus_Unmarried', 'NumberOfChildrenVisiting', 'ProductPitched_Standard', 'Occupation_Large Business', 'ProductPitched_King', 'CityTier', 'Occupation_Salaried', 'AgeGroup', 'MaritalStatus_Married', 'PitchSatisfactionScore', 'PreferredPropertyStar', 'NumberOfPersonVisiting', 'IncomeGroup', 'Occupation_Small Business', 'NumberOfTrips', 'DurationOfPitch', 'Age', 'Gender_Male', 'TotalVisitors', 'TypeofContact_Self Inquiry', 'ProductPitched_Deluxe', 'OwnCar', 'MonthlyIncome', 'ProductPitched_Super Deluxe']

# Ordinal Encoder Categories (from prep.py)
DESIGNATION_ORDER = ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP']
AGEGROUP_ORDER = ['18-30', '31-45', '46-60', '60+']
INCOMEGROUP_ORDER = ['Low', 'Medium', 'High']

NOMINAL_COLS_FOR_DUMMIES = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus']
DUMMY_COLUMNS_EXPECTED = [
    'TypeofContact_Self Inquiry', 'Occupation_Large Business', 'Occupation_Salaried', 'Occupation_Small Business',
    'Gender_Male', 'ProductPitched_Deluxe', 'ProductPitched_King', 'ProductPitched_Standard',
    'ProductPitched_Super Deluxe', 'MaritalStatus_Married', 'MaritalStatus_Single', 'MaritalStatus_Unmarried'
]

# --- Helper Functions for Preprocessing ---
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=HF_MODEL_FILENAME)
    model = joblib.load(model_path)
    return model

def preprocess_input(input_df):
    df = input_df.copy()

    # Feature Engineering
    df['TotalVisitors'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, df['Age'].max()], labels=['18-30', '31-45', '46-60', '60+'], right=False)
    df['IncomeGroup'] = pd.qcut(df['MonthlyIncome'], q=3, labels=['Low', 'Medium', 'High'])

    # Convert bool to int (Passport, OwnCar)
    for col in ['Passport', 'OwnCar']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Ordinal Encoding
    # Create a new encoder instance for each run of preprocess_input to ensure state isolation
    ordinal_encoder = OrdinalEncoder(categories=[DESIGNATION_ORDER, AGEGROUP_ORDER, INCOMEGROUP_ORDER], handle_unknown='use_encoded_value', unknown_value=-1)
    df[['Designation', 'AgeGroup', 'IncomeGroup']] = ordinal_encoder.fit_transform(df[['Designation', 'AgeGroup', 'IncomeGroup']])

    # One-Hot Encoding for nominal columns
    # Need to handle potential missing columns after get_dummies for consistent feature set
    df_encoded = pd.get_dummies(df, columns=NOMINAL_COLS_FOR_DUMMIES, drop_first=True)

    # Add missing dummy columns with 0 and ensure order
    for col in DUMMY_COLUMNS_EXPECTED:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Drop original nominal columns if they still exist and are not part of the dummy set
    for col in NOMINAL_COLS_FOR_DUMMIES:
        if col in df_encoded.columns:
            df_encoded = df_encoded.drop(columns=[col])

    # Initialize StandardScaler with pre-calculated mean and std
    scaler_instance = StandardScaler()
    scaler_instance.mean_ = np.array(SCALER_MEANS)
    scaler_instance.scale_ = np.array(SCALER_STDS)

    # Filter features to scale based on what was scaled during training
    df_scaled = df_encoded.copy()
    df_scaled[SCALED_FEATURE_NAMES] = scaler_instance.transform(df_scaled[SCALED_FEATURE_NAMES])

    # Ensure final DataFrame has the expected columns in the correct order
    # Select only the features present in EXPECTED_FEATURE_ORDER
    final_df = df_scaled[EXPECTED_FEATURE_ORDER]
    return final_df


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Wellness Tourism Package Prediction")
st.write("Enter customer details to predict their likelihood of purchasing the Wellness Tourism Package.")

# Load the model
model = load_model()

# Input features (grouped for better UI)
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Customer Demographics")
    age = st.slider("Age", 18, 80, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Large Business", "Small Business", "Other"])
    monthly_income = st.number_input("Monthly Income", 0, 100000, 30000)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    designation = st.selectbox("Designation", DESIGNATION_ORDER)

with col2:
    st.header("Travel Preferences & History")
    number_of_person_visiting = st.slider("Number of Adults Visiting", 0, 6, 2)
    number_of_children_visiting = st.slider("Number of Children Visiting", 0, 4, 0)
    preferred_property_star = st.slider("Preferred Property Star", 3, 5, 4)
    number_of_trips = st.slider("NumberOfTrips", 0, 20, 5)
    passport = st.checkbox("Holds Passport")
    own_car = st.checkbox("Owns Car")

with col3:
    st.header("Interaction Details")
    typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    number_of_followups = st.slider("NumberOfFollowups", 0, 6, 2)
    duration_of_pitch = st.slider("Duration of Pitch (minutes)", 0, 60, 15)

# Prediction button
if st.button("Predict Purchase Likelihood"):
    input_data = pd.DataFrame([{
        'Age': age,
        'TypeofContact': typeof_contact,
        'CityTier': city_tier,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': number_of_person_visiting,
        'PreferredPropertyStar': preferred_property_star,
        'MaritalStatus': marital_status,
        'NumberOfTrips': number_of_trips,
        'Passport': passport,
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': number_of_children_visiting,
        'Designation': designation,
        'MonthlyIncome': monthly_income,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'ProductPitched': product_pitched,
        'NumberOfFollowups': number_of_followups,
        'DurationOfPitch': duration_of_pitch,
    }])

    # Preprocess input data
    processed_input = preprocess_input(input_data)

    # Make prediction
    prediction_proba = model.predict_proba(processed_input)[:, 1][0] # Probability of purchasing (class 1)

    st.subheader("Prediction Result:")
    if prediction_proba >= 0.5: # Using a threshold of 0.5 for demonstration
        st.success(f"Customer is likely to purchase the Wellness Tourism Package! (Probability: {prediction_proba:.2f})")
    else:
        st.info(f"Customer is less likely to purchase the Wellness Tourism Package. (Probability: {prediction_proba:.2f})")
    st.write(f"Raw Prediction Probability: {prediction_proba:.4f}")
