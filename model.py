import pandas as pd
import requests
from datetime import datetime, timedelta
import os


Features = ['hour', 
            'Amount of precipitation in millimeters (last hour)', 
            'Atmospheric pressure at station level (mb)', 
            'Maximum air pressure for the last hour in hPa to tenths', 
            'Minimum air pressure for the last hour in hPa to tenths', 
            'Solar radiation KJ/m2', 
            'Dew point temperature (instant) in celsius degrees', 
            'Maximum dew point temperature for the last hour in celsius degrees', 
            'Minimum dew point temperature for the last hour in celsius degrees', 
            'Maximum relative humidity temperature for the last hour in %', 
            'Minimum relative humidity temperature for the last hour in %', 
            'Relative humidity in % (instant)', 
            'Wind direction in radius degrees (0-360)', 
            'Wind gust in meters per second', 
            'Wind speed in meters per second', 
            'latitude', 
            'longitude', 
            'height', 
            'year', 
            'month', 
            'day', 
            'region']


# Value ranges
# hour, Range: [00:00 - 23:00]
# Amount of precipitation in millimeters (last hour), Range: [0.0 - 97.2]
# Atmospheric pressure at station level (mb), Range: [852.1 - 1050.0]
# Maximum air pressure for the last hour in hPa to tenths, Range: [832.0 - 1049.8]
# Minimum air pressure for the last hour in hPa to tenths, Range: [830.1 - 1050.0]
# Solar radiation KJ/m2, Range: [0.0 - 45305.0]
# Air temperature (instant) in celsius degrees, Range: [-9.0 - 42.2]
# Dew point temperature (instant) in celsius degrees, Range: [-10.0 - 43.5]
# Maximum temperature for the last hour in celsius degrees, Range: [0.0 - 45.0]
# Minimum temperature for the last hour in celsius degrees, Range: [-5.6 - 45.0]
# Maximum dew point temperature for the last hour in celsius degrees, Range: [-10.0 - 44.4]
# Minimum dew point temperature for the last hour in celsius degrees, Range: [-10.0 - 39.8]
# Maximum relative humidity temperature for the last hour in %, Range: [7.0 - 100.0]
# Minimum relative humidity temperature for the last hour in %, Range: [3.0 - 100.0]
# Relative humidity in % (instant), Range: [7.0 - 100.0]
# Wind direction in radius degrees (0-360), Range: [0.0 - 360.0]
# Wind gust in meters per second, Range: [0.0 - 99.7]
# Wind speed in meters per second, Range: [0.0 - 19.9]
# latitude, Range: [-12.75055555 - 4.47749999]
# longitude, Range: [-72.78666666 - -45.91999999]
# height, Range: [9.92 - 798.0]
# year, Range: [2000 - 2021]
# month, Range: [1 - 12]
# day, Range: [1 - 31]





# hour: (-1.661324924409236, 1.6613238399425894)
# Amount of precipitation in millimeters (last hour): (-0.1497523580515604, 48.43107512748384)
# Atmospheric pressure at station level (mb): (-8.046514565906167, 3.2726993058130187)
# Maximum air pressure for the last hour in hPa to tenths: (-9.216766746477896, 3.2426374462591205)
# Minimum air pressure for the last hour in hPa to tenths: (-9.287450015541673, 3.2931754409013263)
# Solar radiation KJ/m2: (-0.5095849837254371, 26.17143732617853)
# Air temperature (instant) in celsius degrees: (-9.0, 42.2)
# Dew point temperature (instant) in celsius degrees: (-9.229323533114952, 6.530252062334661)
# Maximum dew point temperature for the last hour in celsius degrees: (-9.110809575651627, 6.466505916704183)
# Minimum dew point temperature for the last hour in celsius degrees: (-8.954975272456439, 5.51723631169902)
# Maximum relative humidity temperature for the last hour in %: (-4.267006112092695, 1.2588688271722472)
# Minimum relative humidity temperature for the last hour in %: (-3.8116722281523785, 1.4609228258893434)
# Relative humidity in % (instant): (-3.932909104824224, 1.3519512367968591)
# Wind direction in radius degrees (0-360): (0.0, 360.0)
# Wind gust in meters per second: (-1.4092286681020314, 37.09473565289964)
# Wind speed in meters per second: (-1.087787535416167, 15.718275589355494)
# latitude: (-1.7582679381323891, 2.56731023561989)
# longitude: (-2.264501365681861, 1.222322082216378)
# height: (-0.9961730651891934, 4.087723245397265)
# year: (-3.6874548707356056, 1.555826442033825)
# month: (-1.5842730207407454, 1.5796054107258986)
# day: (-1.675871711066531, 1.73336312573232)
# state: [AC, AM, AP, PA, RO, RR, TO]


raw_data = {
    'hour': [14, 22],
    'Amount of precipitation in millimeters (last hour)': [0.0, 20.5],
    'Atmospheric pressure at station level (mb)': [1012, 950.4],
    'Maximum air pressure for the last hour in hPa to tenths': [1045.0, 900.3],
    'Minimum air pressure for the last hour in hPa to tenths': [1020.2, 880.0],
    'Solar radiation KJ/m2': [1200.0, 30000.0],
    'Dew point temperature (instant) in celsius degrees': [15.0, -3.0],
    'Maximum dew point temperature for the last hour in celsius degrees': [20.0, 0.0],
    'Minimum dew point temperature for the last hour in celsius degrees': [10.0, -8.0],
    'Maximum relative humidity temperature for the last hour in %': [95.0, 50.0],
    'Minimum relative humidity temperature for the last hour in %': [30.0, 10.0],
    'Relative humidity in % (instant)': [55.0, 20.0],
    'Wind direction in radius degrees (0-360)': [180.0, 350.0],
    'Wind gust in meters per second': [5.0, 50.0],
    'Wind speed in meters per second': [3.0, 12.0],
    'latitude': [0.0, -10.0],
    'longitude': [-60.0, -70.0],
    'height': [50.0, 750.0],
    'year': [2021, 2005],
    'month': [6, 12],
    'day': [15, 31],
    'region': ['state_AM', 'state_TO']
}


openWeatherAPIKey='da6d5a75b39ac3fc56893fc06eacfa83'

temp=32
true_temp=1

def append_training_data(data):
    """
    Append data to a CSV dataset file or create a new file if it doesn't exist.

    Args:
        data (dict or DataFrame): Data to be appended to the dataset. This should be a dictionary
                                  where keys correspond to column names and values are lists of
                                  data points for each column, or a DataFrame.
        csv_path (str): Path to the CSV dataset file.

    Returns:
        None
    """
    csv_path = 'newdataset.csv'
    # Convert data to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Check if the CSV file exists
    file_exists = os.path.exists(csv_path)

    # Append data to existing CSV file or create a new one
    mode = 'a' if file_exists else 'w'
    header = not file_exists  # Write header only if the file is new

    with open(csv_path, mode, newline='') as file:
        data.to_csv(file, header=header, index=False)

    print(f"Data appended to {csv_path} successfully.")

def fetch_real_weather_data(input_data):

    global true_temp
    # https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}id&appid={API key}
    # http://api.openweathermap.org/data/2.5/forecast?id=524901&appid={API key}
    latitude = input_data['latitude']
    longitude = input_data['longitude']

    response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={openWeatherAPIKey}&units=metric")
    print(response.status_code, response.text)

    data = response.json()
    true_temp = data['main']['temp']

    dt_utc = datetime.utcfromtimestamp(data['dt'])
    # Apply timezone offset to get local datetime
    dt_local = dt_utc + timedelta(seconds=data['timezone'])
    hour = dt_local.hour
    day = dt_local.day
    month = dt_local.month
    year = dt_local.year

    new_datapoint = {
        'hour': [hour],
        'Atmospheric pressure at station level (mb)': [data['main']['pressure']],
        'Maximum air pressure for the last hour in hPa to tenths': [input_data['max_air_pressure']],
        'Minimum air pressure for the last hour in hPa to tenths': [input_data['min_air_pressure']],
        'Solar radiation KJ/m2': [input_data['solar_radiation']],
        'Dew point temperature (instant) in celsius degrees': [input_data['dew_point_temperature']],
        'Maximum dew point temperature for the last hour in celsius degrees': [input_data['max_dew_point']],
        'Minimum dew point temperature for the last hour in celsius degrees': [input_data['min_dew_point']],
        'Maximum relative humidity temperature for the last hour in %': [input_data['max_relative_humidity']],
        'Minimum relative humidity temperature for the last hour in %': [input_data['min_relative_humidity']],
        'Relative humidity in % (instant)': [data['main']['humidity']],
        'Wind direction in radius degrees (0-360)': [data['wind']['deg']],
        'Wind gust in meters per second': [data['wind']['gust']],
        'Wind speed in meters per second': [data['wind']['speed']],
        'latitude': [latitude],
        'longitude': [longitude],
        'height': [input_data['height']],
        'year': [year],
        'month': [month],
        'day': [day],
        'region': [input_data['region']],
        'temperature' : true_temp
    }
    if data.get('rain'):
        snow = data.get('snow', {}).get('1h', 0)
        new_datapoint['Amount of precipitation in millimeters (last hour)'] = data['rain']['1h'] + snow
    else:
        new_datapoint['Amount of precipitation in millimeters (last hour)'] = input_data['precipitation']
    append_training_data(new_datapoint)
    return true_temp

