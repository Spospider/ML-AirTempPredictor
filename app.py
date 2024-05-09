import numpy as np
import altair as alt
import pandas as pd
import streamlit as st
from datetime import time, datetime

from model import fetch_real_weather_data, temp, true_temp

st.markdown(
    r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True
)


def main():
    global true_temp
    st.title("Outdoor Air Temperature Predictor")

    st.text("Predicted Temperature:")
    display = f"""
    <h1 style="text-align:center; font-size:5rem;">{str(temp)}°</h1>
    """
    st.markdown(display, unsafe_allow_html=True)

    st.text("Real Temperature:")
    
    realtmp_field = st.empty()
    
    with realtmp_field.container():
        display = f"""
        <h1 style="text-align:center; font-size:5rem;">{str(true_temp)}°</h1>
        """
        st.markdown(display, unsafe_allow_html=True)

    process = st.button("Predict", use_container_width=True)
    
    
    
    

    # # Hour (Range: 00:00 - 23:00)
    hour = st.slider("Hour", min_value=0, max_value=23, value=12)
    # # Year (Range: 2000 - 2021)
    # year = st.slider("Year", min_value=2000, max_value=2030, value=2021)

    # # Month (Range: 1 - 12)
    # month = st.slider("Month", min_value=1, max_value=12, value=6)

    # # Day (Range: 1 - 31)
    # day = st.slider("Day", min_value=1, max_value=31, value=15)

    # Date selection
    selected_date = st.date_input("Select a Date", value=datetime.now())

    # Extract hour, day, month, and year from the selected date
    # hour = selected_date
    day = selected_date.day
    month = selected_date.month
    year = selected_date.year

    # Precipitation (Range: 0.0 - 97.2 mm)
    precipitation = st.slider("Precipitation (mm)", min_value=0.0, max_value=150.0, value=0.0)

    # Atmospheric Pressure (Range: 852.1 - 1050.0 mb)
    atmospheric_pressure = st.slider("Atmospheric Pressure (mb)", min_value=500.0, max_value=1500.0, value=1012.0)

    # Maximum Air Pressure (Range: 832.0 - 1049.8 hPa)
    min_air_pressure, max_air_pressure = st.slider("Air Pressure Range (hPa)", min_value=500.0, max_value=1200.0, value=(1020.0, 1045.0))


    # Solar Radiation (Range: 0.0 - 45305.0 KJ/m2)
    solar_radiation = st.slider("Solar Radiation (KJ/m2)", min_value=0.0, max_value=50000.0, value=1200.0)

    # Maximum Dew Point Temperature (Range: -10.0 - 44.4 °C)
    min_dew_point, max_dew_point = st.slider("Dew Point Temperature Range (°C)", min_value=-10.0, max_value=50.0, value=(10.0, 20.0))

    # Dew Point Temperature (Range: -10.0 - 43.5 °C)
    dew_point_temperature = st.slider("Dew Point Temperature (°C)", min_value=-10.0, max_value=50.0, value=15.0)

    # Minimum Dew Point Temperature (Range: -10.0 - 39.8 °C)
    # min_dew_point = st.slider("Min Dew Point (°C)", min_value=-10.0, max_value=50)

    # Maximum Relative Humidity (Range: 7.0 - 100.0 %)
    min_relative_humidity, max_relative_humidity = st.slider("Relative Humidity Range (%)", min_value=0.0, max_value=100.0, value=(30.0, 95.0))

    # Minimum Relative Humidity (Range: 3.0 - 100.0 %)
    # min_relative_humidity = st.slider("Min Relative Humidity (%)", min_value=3.0, max_value=100.0)

    # Relative Humidity (Range: 7.0 - 100.0 %)
    relative_humidity = st.slider("Instant Relative Humidity (%)", min_value=0.0, max_value=100.0, value=55.0)

    # Wind Direction (Range: 0.0 - 360.0 degrees)
    wind_direction = st.slider("Wind Direction (degrees)", min_value=0.0, max_value=360.0, value=180.0)

    # Wind Gust (Range: 0.0 - 99.7 m/s)
    wind_gust = st.slider("Wind Gust (m/s)", min_value=0.0, max_value=120.0, value=5.0)

    # Wind Speed (Range: 0.0 - 19.9 m/s)
    wind_speed = st.slider("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=3.0)

    # Latitude (Range: -12.75055555 - 4.47749999)
    latitude = st.slider("Latitude", min_value=-90.000, max_value=90.000, value=0.0)

    # Longitude (Range: -72.78666666 - -45.91999999)
    longitude = st.slider("Longitude", min_value=-180.0, max_value=180.0, value=-60.0)

    # Height (Range: 9.92 - 798.0)
    height = st.slider("Height", min_value=0.0, max_value=1000.0, value=50.0)

    state = st.selectbox("State", ['AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO'], index = 1)

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
        newtmp = fetch_real_weather_data(input_data)
        with realtmp_field.container():
            display = f"""
            <h1 style="text-align:center; font-size:5rem;">{str(newtmp)}°</h1>
            """
            st.markdown(display, unsafe_allow_html=True)


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
