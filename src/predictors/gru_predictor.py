import numpy as np

def predict_gru(model, data, weather_scaler, power_scaler):
    """Generate 24-hour forecast using GRU model
    
    Args:
        model: Trained Keras GRU model
        data: DataFrame with weather data (needs last 72 hours)
        weather_scaler: StandardScaler for weather features
        power_scaler: MinMaxScaler for power output
    
    Returns:
        dict with forecast and metadata
    """
    # Preprocess
    gru_input = preprocess_for_gru(data, weather_scaler)
    
    # Predict
    prediction_normalized = model.predict(gru_input, verbose=0)
    
    # Denormalize
    prediction = power_scaler.inverse_transform(prediction_normalized).squeeze()
    
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