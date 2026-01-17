import pandas as pd
import numpy as np

def predict_gru(model, data, max_pv_outputs, weather_scaler, power_scaler):
    """Generate 24-hour forecast using GRU model
    
    Args:
        model: Trained Keras GRU model
        data: DataFrame with weather data (needs last 72 hours)
        weather_scaler: StandardScaler for weather features
        power_scaler: MinMaxScaler for power output
    
    Returns:
        dict with forecast and metadata
    """
    print("Preprocessing data for GRU...")

    # Preprocess
    gru_input = preprocess_for_gru(data, weather_scaler)

    print("Generating GRU prediction...")

    # Predict
    prediction_normalized = model.predict(gru_input, verbose=0)
    prediction_normalized = np.clip(prediction_normalized, 0, 1)

    # Denormalize
    prediction = power_scaler.inverse_transform(prediction_normalized.reshape(-1, 1)).flatten()
    prediction = np.clip(prediction, 0, max_pv_outputs * 1.2)  # hard limit at the maximum possible output + 20%

    print("GRU prediction generated.")

    return {
        'forecast': prediction.tolist(),
        'lookback_hours': 72,
        'horizon_hours': 24
    }


def preprocess_for_gru(data, weather_scaler):
    """Convert raw data to format expected by GRU model
    
    Args:
        data: DataFrame with weather data (needs last 72 hours)
        weather_scaler: StandardScaler fitted during training
        max_pv_outputs: Array of maximum possible PV outputs for each hour

    Returns:
        numpy array of shape (1, 72, 6) - batch_size, timesteps, weather_features
    """

    data = add_cyclic_time_features(data)
    
    core_weather = ['temperature', 'humidity', 'solar_radiation', 'pressure']
    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    weather_columns = core_weather + time_features

    # Extract last 72 hours of weather data
    weather_data = data[weather_columns].values[-72:]
    
    # Normalize using the same scaler from training
    weather_normalized = weather_scaler.transform(weather_data)
    
    # Reshape to (batch size = 1, lookback hours = 72, features = number of weather features)
    gru_input = weather_normalized.reshape(1, 72, len(weather_columns))
    
    return gru_input

def add_cyclic_time_features(df):
    """
    Add cyclic time features using sin/cos encoding.

    The sin/cos encoding preserves the cyclic nature:
    - Keeps hour 23 close to hour 0
    - Keeps December close to January
    """
    # Ensure datetime column exists
    if 'date_time' not in df.columns:
        # Try to find or create datetime
        if 'date' in df.columns and 'time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        elif 'timestamp' in df.columns:
            df['date_time'] = pd.to_datetime(df['timestamp'])
        else:
            # Assume index is datetime or create from index
            try:
                df['date_time'] = pd.to_datetime(df.index)
            except:
                raise ValueError("Could not find or create datetime column")

    df['date_time'] = pd.to_datetime(df['date_time'])

    # Extract time components
    hour = df['date_time'].dt.hour
    day_of_year = df['date_time'].dt.dayofyear
    month = df['date_time'].dt.month

    # Cyclic encoding using sin/cos
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    df['day_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    return df
