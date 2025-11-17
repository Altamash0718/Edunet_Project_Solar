import streamlit as st  # This line must be first and correct
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date

# Load the full pipeline (preprocessor + model)
@st.cache_resource
def load_model():
    return joblib.load('models/model_pipeline.pkl')

pipeline = load_model()

st.title('Solar Power Generation Predictor')

# Define feature lists for input preparation (must match training)
numerical_features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
                      'hour_sin', 'hour_cos', 'day_sin', 'month_sin']
categorical_features = ['PLANT_ID']
all_features = numerical_features + categorical_features

# User inputs
st.header('Input Features')
col1, col2 = st.columns(2)
with col1:
    ambient_temp = st.slider('Ambient Temperature (°C)', 20.0, 35.0, 25.0)
    module_temp = st.slider('Module Temperature (°C)', 20.0, 50.0, 30.0)
with col2:
    irradiation = st.slider('Irradiation (W/m²)', 0.0, 1000.0, 500.0)
    plant_id = st.selectbox('Plant ID', [4135001, 4136001])

selected_date = st.date_input('Date', value=date(2020, 6, 1))  # Default to a date in your dataset
selected_hour = st.slider('Hour (0-23)', 0, 23, 12)

# Compute cyclical features
dt = datetime(selected_date.year, selected_date.month, selected_date.day, selected_hour)
hour_sin = np.sin(2 * np.pi * dt.hour / 24)
hour_cos = np.cos(2 * np.pi * dt.hour / 24)
day_of_year = dt.timetuple().tm_yday
day_sin = np.sin(2 * np.pi * day_of_year / 365)
month_sin = np.sin(2 * np.pi * dt.month / 12)

# Prepare input DataFrame (must have exact column order as in training)
input_df = pd.DataFrame({
    'AMBIENT_TEMPERATURE': [ambient_temp],
    'MODULE_TEMPERATURE': [module_temp],
    'IRRADIATION': [irradiation],
    'hour_sin': [hour_sin],
    'hour_cos': [hour_cos],
    'day_sin': [day_sin],
    'month_sin': [month_sin],
    'PLANT_ID': [plant_id]
})

# Make prediction
if st.button('Predict AC Power', type='primary'):
    try:
        prediction = pipeline.predict(input_df)[0]
        st.success(f'Predicted AC Power: {prediction:.2f} kW')
        
        # Optional: Add confidence or explanation
        st.info("Note: Higher irradiation typically leads to higher power output.")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}. Ensure the model pipeline is trained and saved correctly.")