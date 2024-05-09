import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor, DMatrix, cv
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
import os




# Function to load the saved model and scaler
def load_model_and_scaler(model_path, scaler_path):
    with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
    return model, scaler

# Function to apply one-hot encoding to the region
def one_hot_encode_region(region, regions_list):
    # Ensure the prefix 'state_' is added if not present in user input
    if not region.startswith('state_'):
        region = f'state_{region}'
    # else:
    #     x = pd.DataFrame([{f'{r}': 0 for r in regions_list}])
    #     return pd.DataFrame([{f'{r}': 0 for r in regions_list}])


    # Initialize all region columns to zero
    region_encoding = {f'{r}': 0 for r in regions_list}

    # Set the appropriate region to 1
    region_encoding[region] = 1

    # Convert to DataFrame
    return pd.DataFrame([region_encoding])

# Function to preprocess numerical data
def preprocess_numerical_data(raw_data, scaler):
    # Extract numerical data for scaling
    numerical_data = raw_data.select_dtypes(include=[np.number])

    # Apply scaling
    scaled_numerical_data = scaler.transform(numerical_data)

    # Create DataFrame from scaled data
    scaled_df = pd.DataFrame(scaled_numerical_data, columns=numerical_data.columns)
    
    return scaled_df

# Function to preprocess categorical data
def preprocess_categorical_data(raw_data, regions_list):
    # One-hot encode the region
    region_encoded = pd.concat([one_hot_encode_region(region, regions_list) for region in raw_data])

    # Reset index for region_encoded to align with scaled_df
    region_encoded.reset_index(drop=True, inplace=True)
    
    return region_encoded

# Load model and scaler
if os.path.exists("XGB_weighted_retrain.pkl"):
    model_path = 'XGB_weighted_retrain.pkl'
else:
    model_path = 'Final_XGB_weighted_lessdepthnadCrossValidation1.pkl'

# model_path = 'Final_XGB_weighted_lessdepthnadCrossValidation.pkl'
scaler_path = 'standard_scaler.pkl'
model, scaler = load_model_and_scaler(model_path, scaler_path)

# Assume raw_data is a pandas DataFrame with new user data, e.g.:
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
    'region': ['state_AC', 'state_AM'],
}

raw_data = pd.DataFrame(raw_data)
# print (raw_data)
regions_list = ['state_AC', 'state_AM', 'state_AP', 'state_PA', 'state_RO', 'state_RR', 'state_TO']
features_to_standard_scale = [
    'Dew point temperature (instant) in celsius degrees',
    'Relative humidity in % (instant)',
    'latitude',
    'longitude',
    'Amount of precipitation in millimeters (last hour)',
    'Atmospheric pressure at station level (mb)',
    'Maximum air pressure for the last hour in hPa to tenths',
    'Minimum air pressure for the last hour in hPa to tenths',
    'Solar radiation KJ/m2',
    'Maximum dew point temperature for the last hour in celsius degrees',
    'Minimum dew point temperature for the last hour in celsius degrees',
    'Maximum relative humidity temperature for the last hour in %',
    'Minimum relative humidity temperature for the last hour in %',
    'Wind gust in meters per second',
    'Wind speed in meters per second',
    'height','day','hour','month','year',
    'Wind direction in radius degrees (0-360)'
]

with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)


def preprocess_dict(raw_data):
    global feature_names
    if type(raw_data) != pd.DataFrame:
        raw_data = pd.DataFrame(raw_data)
    region_data = raw_data['region']
    # raw_data = raw_data[features_to_standard_scale]

    # Create a separate DataFrame for numerical data
    numerical_data = raw_data[features_to_standard_scale]

    # Now pass the ordered numerical_data to the preprocess_numerical_data function
    scaled_df = preprocess_numerical_data(numerical_data, scaler)
    region_encoded = preprocess_categorical_data(region_data, regions_list)

    # Combine scaled numerical data with one-hot encoded data
    preprocessed_data = pd.concat([scaled_df, region_encoded], axis=1)
    
    # Ensure the preprocessed data has the same columns as the training data
    preprocessed_data = preprocessed_data.reindex(columns=feature_names, fill_value=0)
    # preprocessed_data = pd.get_dummies(preprocessed_data, columns=['region'], prefix='state')

    if 'Air temperature (instant) in celsius degrees' in raw_data:
        preprocessed_data['Air temperature (instant) in celsius degrees'] = raw_data['Air temperature (instant) in celsius degrees']
    return preprocessed_data
region_data = raw_data['region']


raw_data = raw_data[features_to_standard_scale]

# Create a separate DataFrame for numerical data
numerical_data = raw_data[features_to_standard_scale]

# Now pass the ordered numerical_data to the preprocess_numerical_data function
scaled_df = preprocess_numerical_data(numerical_data, scaler)

region_encoded = preprocess_categorical_data(region_data, regions_list)

# Combine scaled numerical data with one-hot encoded data
preprocessed_data = pd.concat([scaled_df, region_encoded], axis=1)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Ensure the preprocessed data has the same columns as the training data
preprocessed_data = preprocessed_data.reindex(columns=feature_names, fill_value=0)


def predict(data):
    preprocessed_data = preprocess_dict(data)
    try:
        predictions = model.predict(preprocessed_data)
    except:
        predictions = model.predict(DMatrix(preprocessed_data))
    return predictions
# # Use the model to predict
# predictions = model.predict(preprocessed_data)

# # Output the prediction
# print("Predicted values:", predictions)32.59°
import xgboost as xgb
from xgboost import DMatrix, train
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score

def retrain_model():
    # return
    # Load and preprocess the new dataset
    existing_data = pd.read_csv('newdataset.csv')

    # Assuming `preprocess_dict` is your custom function for data processing
    combined_data = preprocess_dict(existing_data)

    # Replace missing values with zeros (or another approach suitable for your data)
    combined_data.fillna(0, inplace=True)

    # Split the dataset into features (X) and target variable (y)
    X = combined_data.drop(columns=['Air temperature (instant) in celsius degrees'])  # Features
    y = combined_data['Air temperature (instant) in celsius degrees']  # Target variable

    # Initialize DMatrix for the new data
    dtrain_new = xgb.DMatrix(X, label=y)

    # Load the existing model if available
    try:
        with open('Final_XGB_weighted_lessdepthnadCrossValidation1.pkl', 'rb') as file:
            existing_model = pickle.load(file)
    except FileNotFoundError:
        print("Initial model not found. Train a base model first.")
        return

    # Set XGBoost training parameters
    print(type(existing_model))
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 12,
        'eta': 0.1,
        'random_state': 42
    }

    # Update the model with the new data (retrain without cross-validation)
    num_boost_round = 10  # Number of boosting rounds to update
    updated_model = xgb.train(
        params,
        dtrain_new,
        num_boost_round=num_boost_round,
        xgb_model=existing_model
    )

    # Make predictions with the updated model
    predictions = updated_model.predict(dtrain_new)

    # Calculate and print performance metrics for the new dataset
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    print(f"Mean Squared Error (MSE) for updated model: {mse}")
    print(f"R-squared (R2) for updated model: {r2}")

    # Save the updated model with pickle
    with open('XGB_weighted_retrain.pkl', 'wb') as file:
        pickle.dump(updated_model, file)

    print("Model retraining completed.")
