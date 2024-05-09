import numpy as np
import altair as alt
import pandas as pd
import streamlit as st
from datetime import time, datetime

from model import fetch_real_weather_data, temp, true_temp, log_datapoint
from model_flow import predict
# st.markdown(
#     r"""
#     <style>
#     .stDeployButton {
#             visibility: hidden;
#         }
#     </style>
#     """, unsafe_allow_html=True
# )

def dict_to_list_of_lists(input_data):
    output_data = {}
    for key, value in input_data.items():
        output_data[key] = [value]
    return output_data

def map_keys(input_data):
    key_mapping = {
        'hour': 'hour',
        'precipitation': 'Amount of precipitation in millimeters (last hour)',
        'atmospheric_pressure': 'Atmospheric pressure at station level (mb)',
        'max_air_pressure': 'Maximum air pressure for the last hour in hPa to tenths',
        'min_air_pressure': 'Minimum air pressure for the last hour in hPa to tenths',
        'solar_radiation': 'Solar radiation KJ/m2',
        'dew_point_temperature': 'Dew point temperature (instant) in celsius degrees',
        'max_dew_point': 'Maximum dew point temperature for the last hour in celsius degrees',
        'min_dew_point': 'Minimum dew point temperature for the last hour in celsius degrees',
        'max_relative_humidity': 'Maximum relative humidity temperature for the last hour in %',
        'min_relative_humidity': 'Minimum relative humidity temperature for the last hour in %',
        'relative_humidity': 'Relative humidity in % (instant)',
        'wind_direction': 'Wind direction in radius degrees (0-360)',
        'wind_gust': 'Wind gust in meters per second',
        'wind_speed': 'Wind speed in meters per second',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'height': 'height',
        'year': 'year',
        'month': 'month',
        'day': 'day',
        'region': 'region'
    }

    mapped_data = {}
    for key in input_data:
        mapped_data[key_mapping[key]] = input_data[key]
    return mapped_data

USE_API = False
def main():
    global true_temp
    global USE_API
    st.title("Outdoor Air Temperature Predictor")

    st.text("Predicted Temperature:")
    predtmp_field = st.empty()

    with predtmp_field.container():
        display = f"""
        <h1 style="text-align:center; font-size:5rem;">{str(st.session_state.get('pred', '--'))}°</h1>
        """
        st.markdown(display, unsafe_allow_html=True)
    
    prediction_rating = st.slider("Rate the prediction (0-10)", min_value=0, max_value=10, value=5)

    submit_button = st.button("Submit Rating")

    
        
        
    realtmp_field = st.empty()
    
    # with realtmp_field.container():
    #     display = f"""
    #     <h1 style="text-align:center; font-size:5rem;">--°</h1>
    #     """
    #     st.markdown(display, unsafe_allow_html=True)

    process = st.button("Predict", use_container_width=True)


    # # Hour (Range: 00:00 - 23:00)
    hour = st.slider("Hour", min_value=0, max_value=23, value=12)

    # Date selection
    selected_date = st.date_input("Select a Date", value=datetime.now())

    # Extract hour, day, month, and year from the selected date
    # hour = selected_date
    day = selected_date.day
    month = selected_date.month
    year = selected_date.year
    if  selected_date == datetime.now().date():
        USE_API = True
        with realtmp_field.container():
            st.text("OpenWeatherAPI Real Temperature: (Based on Longitude and Latitude only)")
            display = f"""
            <h1 style="text-align:center; font-size:5rem;">{str(st.session_state.get('truth', '--'))}°</h1>
            """
            st.markdown(display, unsafe_allow_html=True)
    else:
        USE_API = False
        with realtmp_field.container():
            true_temp = st.number_input("Real Temperature")
    ###########################33
    # Precipitation (Range: 0.0 - 97.2 mm)
    # precipitation = st.slider("Precipitation (mm)", min_value=0.0, max_value=150.0, value=0.0)

    # # Atmospheric Pressure (Range: 852.1 - 1050.0 mb)
    # atmospheric_pressure = st.slider("Atmospheric Pressure (mb)", min_value=500.0, max_value=1500.0, value=1012.0)

    # # Maximum Air Pressure (Range: 832.0 - 1049.8 hPa)
    # min_air_pressure, max_air_pressure = st.slider("Air Pressure Range (hPa)", min_value=500.0, max_value=1200.0, value=(1020.0, 1045.0))

    # # Solar Radiation (Range: 0.0 - 45305.0 KJ/m2)
    # solar_radiation = st.slider("Solar Radiation (KJ/m2)", min_value=0.0, max_value=50000.0, value=1200.0)

    # # Maximum Dew Point Temperature (Range: -10.0 - 44.4 °C)
    # min_dew_point, max_dew_point = st.slider("Dew Point Temperature Range (°C)", min_value=-10.0, max_value=50.0, value=(10.0, 20.0))

    # # Dew Point Temperature (Range: -10.0 - 43.5 °C)
    # dew_point_temperature = st.slider("Dew Point Temperature (°C)", min_value=-10.0, max_value=50.0, value=15.0)

    # # Maximum Relative Humidity (Range: 7.0 - 100.0 %)
    # min_relative_humidity, max_relative_humidity = st.slider("Relative Humidity Range (%)", min_value=0.0, max_value=100.0, value=(30.0, 95.0))

    # # Relative Humidity (Range: 7.0 - 100.0 %)
    # relative_humidity = st.slider("Instant Relative Humidity (%)", min_value=0.0, max_value=100.0, value=55.0)

    # # Wind Direction (Range: 0.0 - 360.0 degrees)
    # wind_direction = st.slider("Wind Direction (degrees)", min_value=0.0, max_value=360.0, value=180.0)

    # # Wind Gust (Range: 0.0 - 99.7 m/s)
    # wind_gust = st.slider("Wind Gust (m/s)", min_value=0.0, max_value=120.0, value=5.0)

    # # Wind Speed (Range: 0.0 - 19.9 m/s)
    # wind_speed = st.slider("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=3.0)

    # # Latitude (Range: -12.75055555 - 4.47749999)
    # latitude = st.slider("Latitude", min_value=-90.000, max_value=90.000, value=0.0)

    # # Longitude (Range: -72.78666666 - -45.91999999)
    # longitude = st.slider("Longitude", min_value=-180.0, max_value=180.0, value=-60.0)

    # # Height (Range: 9.92 - 798.0)
    # height = st.slider("Height", min_value=0.0, max_value=1000.0, value=50.0)
    #####################33
    # Precipitation (Range: 0.0 - 97.2 mm)
    precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=150.0, value=0.0)

    # Atmospheric Pressure (Range: 852.1 - 1050.0 mb)
    atmospheric_pressure = st.number_input("Atmospheric Pressure (mb)", min_value=500.0, max_value=1500.0, value=1012.0)

    # Maximum Air Pressure (Range: 832.0 - 1049.8 hPa)
    min_air_pressure = st.number_input("Min Air Pressure (hPa)", min_value=500.0, max_value=1200.0, value=1020.0)
    max_air_pressure = st.number_input("Max Air Pressure (hPa)", min_value=500.0, max_value=1200.0, value=1045.0)

    # Solar Radiation (Range: 0.0 - 45305.0 KJ/m2)
    solar_radiation = st.number_input("Solar Radiation (KJ/m2)", min_value=0.0, max_value=50000.0, value=1200.0)

    # Maximum Dew Point Temperature (Range: -10.0 - 44.4 °C)
    min_dew_point = st.number_input("Min Dew Point Temperature (°C)", min_value=-10.0, max_value=50.0, value=10.0)
    max_dew_point = st.number_input("Max Dew Point Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0)

    # Dew Point Temperature (Range: -10.0 - 43.5 °C)
    dew_point_temperature = st.number_input("Dew Point Temperature (°C)", min_value=-10.0, max_value=50.0, value=15.0)

    # Maximum Relative Humidity (Range: 7.0 - 100.0 %)
    min_relative_humidity = st.number_input("Min Relative Humidity (%)", min_value=0.0, max_value=100.0, value=30.0)
    max_relative_humidity = st.number_input("Max Relative Humidity (%)", min_value=0.0, max_value=100.0, value=95.0)

    # Relative Humidity (Range: 7.0 - 100.0 %)
    relative_humidity = st.number_input("Instant Relative Humidity (%)", min_value=0.0, max_value=100.0, value=55.0)

    # Wind Direction (Range: 0.0 - 360.0 degrees)
    wind_direction = st.number_input("Wind Direction (degrees)", min_value=0.0, max_value=360.0, value=180.0)

    # Wind Gust (Range: 0.0 - 99.7 m/s)
    wind_gust = st.number_input("Wind Gust (m/s)", min_value=0.0, max_value=120.0, value=5.0)

    # Wind Speed (Range: 0.0 - 19.9 m/s)
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=3.0)

    # Latitude (Range: -12.75055555 - 4.47749999)
    latitude = st.number_input("Latitude", min_value=-90.000, max_value=90.000, value=0.0)

    # Longitude (Range: -72.78666666 - -45.91999999)
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-60.0)

    # Height (Range: 9.92 - 798.0)
    height = st.number_input("Height", min_value=0.0, max_value=1000.0, value=50.0)


    #############333
    state = st.selectbox("State", ['state_AC', 'state_AM', 'state_AP', 'state_PA', 'state_RO', 'state_RR', 'state_TO'], index = 1)

    input_data = {
        "hour": hour,
        "precipitation": precipitation,
        "atmospheric_pressure": atmospheric_pressure,
        "max_air_pressure": max_air_pressure,
        "min_air_pressure": min_air_pressure,
        "solar_radiation": solar_radiation,
        "dew_point_temperature": dew_point_temperature,
        "max_dew_point": max_dew_point,
        "min_dew_point": min_dew_point,
        "max_relative_humidity": max_relative_humidity,
        "min_relative_humidity": min_relative_humidity,
        "relative_humidity": relative_humidity,
        "wind_direction": wind_direction,
        "wind_gust": wind_gust,
        "wind_speed": wind_speed,
        "latitude": latitude,
        "longitude": longitude,
        "height": height,
        "year": year,
        "month": month,
        "day": day,
        "region": state
    }

    if process:
        
        pred = predict(dict_to_list_of_lists(map_keys(input_data)))[0]
        
        with predtmp_field.container():
            display = f"""
            <h1 style="text-align:center; font-size:5rem;">{str(pred)}°</h1>
            """
            st.markdown(display, unsafe_allow_html=True)

            st.session_state['pred'] = pred
            
            # if  prediction_rating > 7:
            #     acceptable = True
            # else:
            #     acceptable = False
            if USE_API:
                newtmp = fetch_real_weather_data(input_data, log_data=False)
                st.session_state['truth'] = newtmp
                with realtmp_field.container():
                    st.text("OpenWeatherAPI Real Temperature: (Based on Longitude and Latitude only)")
                    display = f"""
                    <h1 style="text-align:center; font-size:5rem;">{str(newtmp)}°</h1>
                    """
                    st.markdown(display, unsafe_allow_html=True)

            # elif acceptable:
            #     newtmp = log_datapoint(input_data)
            
    # Check if the submit button is clicked
    if submit_button:
        if  prediction_rating >= 7:
            acceptable = True
        else:
            acceptable = False
        if USE_API:
            newtmp = fetch_real_weather_data(input_data, log_data=acceptable)
            with realtmp_field.container():
                st.text("OpenWeatherAPI Real Temperature: (Based on Longitude and Latitude only)")
                display = f"""
                <h1 style="text-align:center; font-size:5rem;">{str(newtmp)}°</h1>
                """
                st.markdown(display, unsafe_allow_html=True)

        elif acceptable:
            newtmp = log_datapoint(input_data)

    # Display the selected values
    st.write("Selected Values:")
    st.write({
        "Hour": hour,
        "Precipitation (mm)": precipitation,
        "Atmospheric Pressure (mb)": atmospheric_pressure,
        "Max Air Pressure (hPa)": max_air_pressure,
        "Min Air Pressure (hPa)": min_air_pressure,
        "Solar Radiation (KJ/m2)": solar_radiation,
        "Dew Point Temperature (°C)": dew_point_temperature,
        "Max Dew Point (°C)": max_dew_point,
        "Min Dew Point (°C)": min_dew_point,
        "Max Relative Humidity (%)": max_relative_humidity,
        "Min Relative Humidity (%)": min_relative_humidity,
        "Relative Humidity (%)": relative_humidity,
        "Wind Direction (degrees)": wind_direction,
        "Wind Gust (m/s)": wind_gust,
        "Wind Speed (m/s)": wind_speed,
        "Latitude": latitude,
        "Longitude": longitude,
        "Height": height,
        "Year": year,
        "Month": month,
        "Day": day,
        "State": state
    })

if __name__ == "__main__":
    main()
