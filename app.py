import streamlit as st
import numpy as np
import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
model = pickle.load(open(model_path, "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))

# Apply Custom CSS
st.markdown(
    """
    <style>
        /* Change background color */
        .stApp {
            background-color: black; /* Light grey background */
        }
        
        h5{
            font-size: 18px;
            font-weight: bold;
            color: #FFFFFF !important;
            margin-bottom: 5px;
        }

        /* Style input fields */
        input {
            border: 2px solid black !important; /* Black border */
            border-radius: 8px !important;
            padding: 8px !important;
            font-size: 18px !important;
            background-color: #ffffff !important;
            color: black !important;
            border-color:white !important;
        }

        /* Style the predict button */
        .stButton>button {
            background-color: #28a745 !important; /* Green */
            color: black !important;
            font-weight: 18px !important;
            border-radius: 10px !important;
            padding: 10px !important;
            width: 100% !important;
            
        }

        /* Change button hover effect */
        .stButton>button:hover {
            background-color:#005d00 !important;
            border-color: #005d00;
        }

        /* Style the output message */
        .stSuccess {
            font-size: 20px !important;
            color: #155724 !important;
            font-weight: bold !important;
            background-color: #d4edda !important; /* Light green background */
            padding: 15px !important;
            border-radius: 10px !important;
            border-color:black !important;
        }
        
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App Title with Inline CSS
st.markdown('<h1 style="text-align: center; color: white;">🔍 Diabetes Prediction App</h1>', unsafe_allow_html=True)
st.markdown('<h4 style="color: white;">Enter your health details below to check your diabetes risk.<h4>',unsafe_allow_html=True)

# Collect User Input
st.markdown('<h5>Pregnancies</h5>', unsafe_allow_html=True)
pregnancies = st.number_input("", min_value=0, max_value=20, step=1, key="pregnancies")

st.markdown('<h5>Glucose Level</h5>', unsafe_allow_html=True)
glucose = st.number_input("", min_value=0, max_value=200, key="glucose")

st.markdown('<h5>Blood Pressure</h5>', unsafe_allow_html=True)
bp = st.number_input("", min_value=0, max_value=150, key="bp")

st.markdown('<h5>Skin Thickness</h5>', unsafe_allow_html=True)
skin_thickness = st.number_input("", min_value=0, max_value=100, key="skin_thickness")

st.markdown('<h5>Insulin Level</h5>', unsafe_allow_html=True)
insulin = st.number_input("", min_value=0, max_value=900, key="insulin")

st.markdown('<h5>BMI (Body Mass Index)</h5>', unsafe_allow_html=True)
bmi = st.number_input("", min_value=0.0, max_value=60.0, key="bmi")

st.markdown('<h5>Diabetes Pedigree Function</h5>', unsafe_allow_html=True)
dpf = st.number_input("", min_value=0.0, max_value=2.5, key="dpf")

st.markdown('<h5>Age</h5>', unsafe_allow_html=True)
age = st.number_input("", min_value=1, max_value=120, key="age")

# Predict Button
if st.button("Predict"):
    # Convert input to numpy array
    new_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scale the input
    new_data_scaled = scaler.transform(new_data)
    
    # Predict
    prediction = model.predict(new_data_scaled)
    
    # Display Result
    result = "🩺 You may have **Diabetes**. Please consult a doctor!" if prediction[0] == 1 else "✅ You are **Healthy**. Keep maintaining a good lifestyle!"
    st.success(result)
