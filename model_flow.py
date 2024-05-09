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
    model_path = 'Final_XGB_weighted_lessdepthnadCrossValidation.pkl'

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
    print(scaled_df)
    region_encoded = preprocess_categorical_data(region_data, regions_list)

    # Combine scaled numerical data with one-hot encoded data
    preprocessed_data = pd.concat([scaled_df, region_encoded], axis=1)
    print("after")
    print(preprocessed_data)
    
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
    predictions = model.predict(DMatrix(preprocessed_data))
    return predictions
# # Use the model to predict
# predictions = model.predict(preprocessed_data)

# # Output the prediction
# print("Predicted values:", predictions)32.59°

from xgboost import DMatrix, train
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score

def retrain_model():
    # Load existing data from the CSV file
    existing_data = pd.read_csv('newdataset.csv')

    # Combine existing data with new data
    combined_data = preprocess_dict(pd.read_csv('newdataset.csv'))  # Add your logic here to fetch new data
    # combined_data = pd.concat([existing_data, new_data])
    combined_data.fillna(0)
    print(combined_data)
    # Split the dataset into features (X) and target variable (y)
    X = combined_data.drop(columns=['Air temperature (instant) in celsius degrees'])  # Features
    y = combined_data['Air temperature (instant) in celsius degrees']  # Target variable

    # Define temperature periphery ranges
    lower_periphery = -10
    upper_periphery = 10
    lower_extreme = 40
    upper_extreme = 50

    # Initialize weights for all samples
    weights = np.ones(len(y))

    # Assign higher weights to peripheral temperature regions
    weights[(y >= lower_periphery) & (y <= upper_periphery)] = 10.0
    weights[(y >= lower_extreme) & (y <= upper_extreme)] = 10.0

    # Convert the continuous target variable into categorical bins
    bins = [-10, 20, 30, np.inf]
    labels = ['Very Cold to Moderate', 'Moderate to Warm', 'Warm to Hot']
    y_bins = pd.cut(y, bins=bins, labels=labels)

    # Perform stratified cross-validation
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    mse_scores = []
    r2_scores = []

    for train_index, test_index in skf.split(X, y_bins):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        weights_train = weights[train_index]

        dtrain = DMatrix(X_train, label=y_train, weight=weights_train)
        dtest = DMatrix(X_test, label=y_test)

        params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'max_depth': 12,
        }

        num_rounds = 20

        bst = train(params, dtrain, num_rounds, evals=[(dtest, 'eval')], verbose_eval=False)

        y_pred = bst.predict(dtest)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    print("Mean Squared Error (MSE) For XGBoost with Weighted Samples:", np.mean(mse_scores))
    print("R-squared (R2) For XGBoost with Weighted Samples:", np.mean(r2_scores))

    # Save the model with pickle
    with open('XGB_weighted_retrain.pkl', 'wb') as file:
        pickle.dump(bst, file)

    print("Model retraining completed.")
