#streamlit without background image
import pickle
import pandas as pd

with open("rainfall_prediction_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
feature_names = model_data["feature_names"]

def get_user_input():
    print("Enter the following weather parameters:")
    pressure = float(input("Pressure (in hPa): "))
    dewpoint = float(input("Dewpoint (in °C): "))
    humidity = float(input("Humidity (in %): "))
    cloud = float(input("Cloud Cover (in %): "))
    sunshine = float(input("Sunshine (in hours): "))
    winddirection = float(input("Wind Direction (in degrees): "))
    windspeed = float(input("Wind Speed (in km/h): "))
    
    return [pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]

input_data = get_user_input()

input_df = pd.DataFrame([input_data], columns=feature_names)

probabilities = model.predict_proba(input_df)[0]

rainfall_chance = probabilities[1] * 100 
no_rainfall_chance = probabilities[0] * 100  

print("\nPrediction Result:")
print(f"Chance of Rainfall: {rainfall_chance:.2f}%")
print(f"Chance of No Rainfall: {no_rainfall_chance:.2f}%")
