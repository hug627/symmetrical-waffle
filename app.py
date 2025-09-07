import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Load the trained model
# -------------------------------------------------
st.title("🏦 Financial Inclusion Prediction App")

try:
    model = joblib.load("financial_model.pkl")
    st.success("✅ Model loaded successfully!")
except:
    st.error("❌ Could not load model file. Make sure 'financial_model.pkl' is in the same folder.")
    st.stop()

# -------------------------------------------------
# Define mappings (must match your LabelEncoder encodings during training)
# -------------------------------------------------
country_map = {"Kenya": 0, "Rwanda": 1, "Tanzania": 2, "Uganda": 3}
location_map = {"Urban": 0, "Rural": 1}
cellphone_map = {"Yes": 1, "No": 0}
gender_map = {"Male": 0, "Female": 1}
relationship_map = {
    "Head of Household": 0,
    "Spouse": 1,
    "Child": 2,
    "Other relative": 3,
    "Other": 4
}
marital_map = {
    "Married/Living together": 0,
    "Divorced/Separated": 1,
    "Widowed": 2,
    "Single/Never Married": 3
}
education_map = {
    "No formal education": 0,
    "Primary education": 1,
    "Secondary education": 2,
    "Tertiary education": 3,
    "Vocational/Specialised training": 4,
    "Other/Dont know/RTA": 5
}
job_map = {
    "Self employed": 0,
    "Government Dependent": 1,
    "Formally employed Private": 2,
    "Formally employed Government": 3,
    "Informally employed": 4,
    "Farming and Fishing": 5,
    "Remittance Dependent": 6,
    "Other Income": 7,
    "Dont Know/Refuse to answer": 8,
    "No Income": 9
}

# -------------------------------------------------
# Sidebar inputs
# -------------------------------------------------
st.sidebar.header("📋 Enter Individual Information")

country = st.sidebar.selectbox("Country", list(country_map.keys()))
year = st.sidebar.number_input("Year", value=2016, step=1)
location_type = st.sidebar.selectbox("Location Type", list(location_map.keys()))
cellphone_access = st.sidebar.selectbox("Cellphone Access", list(cellphone_map.keys()))
household_size = st.sidebar.number_input("Household Size", min_value=1, value=3, step=1)
age_of_respondent = st.sidebar.number_input("Age of Respondent", min_value=15, value=30, step=1)
gender_of_respondent = st.sidebar.selectbox("Gender", list(gender_map.keys()))
relationship_with_head = st.sidebar.selectbox("Relationship with Head", list(relationship_map.keys()))
marital_status = st.sidebar.selectbox("Marital Status", list(marital_map.keys()))
education_level = st.sidebar.selectbox("Education Level", list(education_map.keys()))
job_type = st.sidebar.selectbox("Job Type", list(job_map.keys()))

# -------------------------------------------------
# Prepare input DataFrame
# -------------------------------------------------
input_df = pd.DataFrame([[
    country, year, location_type, cellphone_access, household_size,
    age_of_respondent, gender_of_respondent, relationship_with_head,
    marital_status, education_level, job_type
]], columns=[
    "country", "year", "location_type", "cellphone_access", "household_size",
    "age_of_respondent", "gender_of_respondent", "relationship_with_head",
    "marital_status", "education_level", "job_type"
])

st.write("🔍 Preview of Input Data (before encoding):")
st.dataframe(input_df)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔮 Predict"):
    try:
        # Apply mappings to convert categorical strings into numbers
        input_df["country"] = input_df["country"].map(country_map)
        input_df["location_type"] = input_df["location_type"].map(location_map)
        input_df["cellphone_access"] = input_df["cellphone_access"].map(cellphone_map)
        input_df["gender_of_respondent"] = input_df["gender_of_respondent"].map(gender_map)
        input_df["relationship_with_head"] = input_df["relationship_with_head"].map(relationship_map)
        input_df["marital_status"] = input_df["marital_status"].map(marital_map)
        input_df["education_level"] = input_df["education_level"].map(education_map)
        input_df["job_type"] = input_df["job_type"].map(job_map)

        # Debug: Show encoded input
        st.write("✅ Encoded Input Data (ready for model):")
        st.dataframe(input_df)

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Show result
        st.subheader("📊 Prediction Result")
        if prediction == 1:
            st.success("✅ This person is likely to HAVE a bank account.")
        else:
            st.warning("⚠️ This person is UNLIKELY to have a bank account.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
