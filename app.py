import streamlit as st
import numpy as np
import pickle
import os

# Error handling for loading model and scaler
try:
    model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
    model = pickle.load(open(model_path, "rb"))
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading scaler: {e}")

# Apply Custom CSS
st.markdown(
    """
    <style>
        /* Custom CSS styles here */
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
    if 'model' in locals() and 'scaler' in locals():  # Ensure model and scaler are loaded
        # Convert input to numpy array
        new_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])

        # Debugging: Check the shape of new_data
        st.write(f"Input Data: {new_data}")
        st.write(f"Shape of Input Data: {new_data.shape}")

        try:
            # Scale the input
            new_data_scaled = scaler.transform(new_data)
            st.write(f"Scaled Data: {new_data_scaled}")
        except Exception as e:
            st.error(f"Error scaling data: {e}")
            new_data_scaled = None

        if new_data_scaled is not None:
            # Predict
            prediction = model.predict(new_data_scaled)

            # Display Result
            result = "🩺 You may have **Diabetes**. Please consult a doctor!" if prediction[0] == 1 else "✅ You are **Healthy**. Keep maintaining a good lifestyle!"
            st.success(result)
    else:
        st.error("Model or scaler is not loaded correctly. Please check the logs.")
