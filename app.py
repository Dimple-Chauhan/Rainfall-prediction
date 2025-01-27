#streamlit with background image
import pickle
import pandas as pd
import streamlit as st
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

with open("rainfall_prediction_model.pkl","rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
feature_names = model_data["feature_names"]

st.title("Rainfall Prediction System")

background_image_path = 'images.jpg'  
background_image_base64 = image_to_base64(background_image_path)

st.markdown(f"""
    <style>
        .stApp {{
            background-image: url('data:image/jpeg;base64,{background_image_base64}');
            background-size: cover;
            background-position: center;
            height: 100vh;
        }}

          .stSubheader, .stMarkdown, .stText, .stWrite {{
            color: #00FFFF;
            font-weight:bold;  
        }}

         
    </style>
""", unsafe_allow_html=True)

st.write("Welcome to the Rainfall Prediction App!")

st.sidebar.header("Input Weather Parameters")
pressure = st.sidebar.slider("Pressure (in hPa)", min_value=950, max_value=1050, value=950, step=1)
dewpoint = st.sidebar.slider("Dewpoint (in °C)", min_value=-30, max_value=40, value=100, step=1)
humidity = st.sidebar.slider("Humidity (in %)", min_value=0, max_value=100, value=100, step=1)
cloud = st.sidebar.slider("Cloud Cover (in %)", min_value=0, max_value=100, value=100, step=1)
sunshine = st.sidebar.slider("Sunshine (in hours)", min_value=0, max_value=12, value=5, step=1)
winddirection = st.sidebar.slider("Wind Direction (in degrees)", min_value=0, max_value=360, value=180, step=1)
windspeed = st.sidebar.slider("Wind Speed (in km/h)", min_value=0, max_value=200, value=25, step=1)

input_data = [[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]]
input_df = pd.DataFrame(input_data, columns=feature_names)

if st.button("Predict"):
    probabilities = model.predict_proba(input_df)[0]
    rainfall_chance = probabilities[1] * 100  
    no_rainfall_chance = probabilities[0] * 100  

    st.subheader("✨✨ Prediction Results")
    st.write(f"🌧️ ***Chance of Rainfall:*** {rainfall_chance:.2f}%")
    st.write(f"☀️ ***Chance of No Rainfall:*** {no_rainfall_chance:.2f}%")
else:
    st.write("👆 Adjust the parameters and click **Predict** to see the results.")

st.sidebar.write("----")
st.sidebar.write("••••♥•••• Thank you for using the Rainfall Prediction System!")
