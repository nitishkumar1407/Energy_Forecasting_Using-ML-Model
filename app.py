import streamlit as st
import requests
from datetime import date

st.set_page_config(page_title="Energy Forecasting Using ML", page_icon="⚡")
st.title(" Energy Forecasting Using ML")
st.markdown("Enter the details below to get a predicted energy consumption value.")

city = st.selectbox("City", [
    "Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain", "Sagar", "Rewa", "Satna",
    "Ratlam", "Dewas", "Khargone", "Murwara", "Bhind", "Chhindwara", "Shivpuri"
])
temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0)
selected_date = st.date_input("Date", value=date.today())

if st.button(" Predict Energy Consumption"):
    payload = {
        "city": city,
        "temperature": temperature,
        "date": selected_date.strftime('%Y-%m-%d')
    }

    with st.spinner("Sending data to the backend for prediction..."):
        try:
            response = requests.post("http://127.0.0.1:5001/predict", json=payload, timeout=5)

            if response.status_code == 200:
                result = response.json()
                st.success("✅ Prediction Successful!")
                st.markdown(f"""
                    ### Predicted Energy Consumption: ⁠ {result['predicted_consumption_kWh']} kWh
                    - City: {result['city']}
                    - Date: {result['date']}
                    - Temperature: {result['temperature']} °C
                """)
            else:
                st.error("❌ Backend returned an error.")
                st.json(response.json())
        except Exception as e:
            st.error(f"⚠️ Could not connect to backend: {e}")