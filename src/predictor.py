import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_predictions(models, data, scalers):
    """Generate predictions using loaded models and fetched data
    
    Args:
        models: dict with 'gru' and 'autoformer' models
        data: DataFrame with columns ['timestamp', 'temperature', 'humidity', 
              'solar_radiation', 'wind_speed', 'wind_direction']
        scalers: dict with 'weather_scaler' and 'power_scaler'
    
    Returns:
        dict with predictions from both models
    """
    
    # Preprocess for GRU (Keras)
    gru_input = preprocess_for_gru(data, scalers['weather_scaler'])
    gru_prediction_normalized = models['gru'].predict(gru_input, verbose=0)
    
    # Denormalize predictions back to actual power values
    gru_prediction = scalers['power_scaler'].inverse_transform(
        gru_prediction_normalized
    ).squeeze()
    
    # Preprocess for Autoformer (PyTorch)
    autoformer_input = preprocess_for_autoformer(data)
    
    with torch.no_grad():
        autoformer_output = models['autoformer'].generate(
            **autoformer_input,
            num_parallel_samples=1
        )
        autoformer_prediction = autoformer_output.sequences.mean(dim=1).squeeze().numpy()
    
    # Build results
    results = {
        'gru': {
            'forecast': gru_prediction.tolist(),
            'lookback_hours': 72,
            'horizon_hours': 24
        },
        'autoformer': {
            'forecast': autoformer_prediction.tolist(),
            'lookback_hours': 336,
            'horizon_hours': 96
        }
    }
    
    return results

def preprocess_for_gru(data, weather_scaler):
    """Convert raw data to format expected by GRU model
    
    Args:
        data: DataFrame with weather data (needs last 72 hours)
        weather_scaler: StandardScaler fitted during training
    
    Returns:
        numpy array of shape (1, 72, 5) - batch_size, timesteps, weather_features
    """
    # Define weather columns in the same order as training
    weather_columns = ['temperature', 'humidity', 'solar_radiation',
                       'wind_speed', 'wind_direction']
    
    # Extract last 72 hours of weather data
    weather_data = data[weather_columns].values[-72:]
    
    # Normalize using the same scaler from training
    weather_normalized = weather_scaler.transform(weather_data)
    
    # Reshape to (batch_size=1, timesteps=72, features=5)
    gru_input = weather_normalized.reshape(1, 72, 5)
    
    return gru_input


def preprocess_for_autoformer(data):
    """Convert raw data to format expected by Autoformer model
    
    Args:
        data: DataFrame with timestamp and price columns
    
    Returns:
        dict with tensors in Autoformer format
    """
    # Extract last 336 hours
    recent_data = data.iloc[-336:].copy()
    
    # Extract target values (prices)
    past_values = torch.tensor(recent_data['price'].values, dtype=torch.float32).unsqueeze(0)
    
    # Create time features (hour, day_of_year, month)
    timestamps = pd.to_datetime(recent_data['timestamp'])
    
    hour_of_day = timestamps.dt.hour.values / 23.0
    day_of_year = timestamps.dt.dayofyear.values / 365.0
    month = (timestamps.dt.month.values - 1) / 11.0
    
    past_time_features = np.stack([hour_of_day, day_of_year, month], axis=1)
    past_time_features = torch.tensor(past_time_features, dtype=torch.float32).unsqueeze(0)
    
    # Create future time features (for the 96 forecast hours)
    last_timestamp = timestamps.iloc[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=96,
        freq='h'
    )
    
    future_hour = future_timestamps.hour.values / 23.0
    future_day = future_timestamps.dayofyear.values / 365.0
    future_month = (future_timestamps.month.values - 1) / 11.0
    
    future_time_features = np.stack([future_hour, future_day, future_month], axis=1)
    future_time_features = torch.tensor(future_time_features, dtype=torch.float32).unsqueeze(0)
    
    static_categorical_features = torch.tensor([[0]], dtype=torch.long)
    
    autoformer_input = {
        'past_values': past_values,
        'past_time_features': past_time_features,
        'future_time_features': future_time_features,
        'static_categorical_features': static_categorical_features,
        'past_observed_mask': torch.ones_like(past_values, dtype=torch.bool)
    }
    
    return autoformer_input