import torch
import numpy as np
import pandas as pd
from datetime import timedelta

def predict_autoformer(model, data, max_pv_outputs, weather_scaler, power_scaler, climatology):
    """Generate 96-hour forecast using Autoformer model
    
    Args:
        model: Trained PyTorch Autoformer model
        data: DataFrame with weather data (needs last 336 hours)
        weather_scaler: StandardScaler for weather features
        power_scaler: MinMaxScaler for power output
        climatology: dict with 'clim_table', 'clim_valid', 'clim_global_mean'
    
    Returns:
        dict with forecast and metadata
    """
    # Preprocess
    autoformer_input, max_pv_outputs = preprocess_for_autoformer(
        data, 
        max_pv_outputs,
        weather_scaler,
        climatology['clim_table'],
        climatology['clim_valid'],
        climatology['clim_global_mean']
    )
    
    # Predict
    with torch.no_grad():
        output = model.generate(**autoformer_input)
        prediction_normalized = output.sequences.median(dim=1).values.cpu().numpy()
    
    # Denormalize
    prediction = power_scaler.inverse_transform(
        prediction_normalized.reshape(-1, 1)
    ).flatten()
    
    # Clip to valid range
    prediction = np.clip(prediction, 0, max_pv_outputs * 1.2)  # hard limit at the maximum possible output + 20%
    
    return {
        'forecast': prediction.tolist(),
        'lookback_hours': 336,
        'horizon_hours': 96
    }


def preprocess_for_autoformer(data, max_pv_outputs, weather_scaler, clim_table, clim_valid, clim_global_mean):
    """Convert raw data to format expected by Autoformer model
    
    Args:
        data: DataFrame with timestamp and weather columns (needs last 336 hours)
        max_pv_outputs: Array of maximum possible PV outputs for each hour
        weather_scaler: StandardScaler fitted during training
        clim_table: Climatology lookup table (367, 24, 5)
        clim_valid: Climatology validity mask (367, 24)
        clim_global_mean: Global mean weather (5,)
    
    Returns:
        dict with tensors in Autoformer format
    """
    weather_columns = ['temperature', 'humidity', 'solar_radiation', 'pressure']
    
    # Extract last 336 hours
    recent_data = data.iloc[-336:].copy()
    max_pv_outputs = max_pv_outputs[-336:]
    
    # Get timestamps
    timestamps = pd.to_datetime(recent_data['date_time'])
    
    # ---- PAST WEATHER (last 336 hours) ----
    past_weather = recent_data[weather_columns].values
    past_weather_normalized = weather_scaler.transform(past_weather)
    last_weather = past_weather_normalized[-1]  # Last observation for blending
    
    # ---- FUTURE WEATHER PROXY (next 96 hours) ----
    # Generate future timestamps
    last_timestamp = timestamps.iloc[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=96,
        freq='h'
    )
    
    # Extract day of year and hour for climatology lookup
    future_doy = future_timestamps.dayofyear.values
    future_hour = future_timestamps.hour.values
    
    # Create future weather proxy using climatology
    future_weather_proxy = make_future_weather_proxy(
        last_weather=last_weather,
        future_doy=future_doy,
        future_hour=future_hour,
        clim_table=clim_table,
        clim_valid=clim_valid,
        clim_global_mean=clim_global_mean,
        step_minutes=60,
        alpha_end=0.2,
        horizon_hours=96
    )
    
    # ---- TIME FEATURES ----
    # Past time features
    past_hour = timestamps.dt.hour.values / 23.0
    past_doy = timestamps.dt.dayofyear.values / 365.0
    past_month = (timestamps.dt.month.values - 1) / 11.0
    past_time_base = np.stack([past_hour, past_doy, past_month], axis=1)
    
    # Concatenate with weather
    past_time_features = np.concatenate([past_time_base, past_weather_normalized], axis=1)
    past_time_features = torch.tensor(past_time_features, dtype=torch.float32).unsqueeze(0)
    
    # Future time features
    future_hour = future_timestamps.hour.values / 23.0
    future_doy = future_timestamps.dayofyear.values / 365.0
    future_month = (future_timestamps.month.values - 1) / 11.0
    future_time_base = np.stack([future_hour, future_doy, future_month], axis=1)
    
    # Concatenate with weather proxy
    future_time_features = np.concatenate([future_time_base, future_weather_proxy], axis=1)
    future_time_features = torch.tensor(future_time_features, dtype=torch.float32).unsqueeze(0)
    
    # ---- PAST VALUES (all zeros - we don't use past power) ----
    past_values = torch.zeros((1, 336), dtype=torch.float32)
    past_observed_mask = torch.zeros((1, 336), dtype=torch.bool)  # All unobserved
    
    # ---- STATIC FEATURES ----
    static_categorical_features = torch.tensor([[0]], dtype=torch.long)
    
    autoformer_input = {
        'past_values': past_values,
        'past_time_features': past_time_features,
        'future_time_features': future_time_features,
        'static_categorical_features': static_categorical_features,
        'past_observed_mask': past_observed_mask
    }
    
    return autoformer_input, max_pv_outputs


# ============================ #
# CLIMATOLOGY HELPER FUNCTIONS #
# ============================ #

def alpha_exponential(lead_hours, alpha_end=0.2, horizon_hours=96):
    """Calculate exponential decay for weather blending"""
    k = -np.log(alpha_end) / horizon_hours
    return np.exp(-k * lead_hours).astype(np.float32)


def make_future_weather_proxy(
    last_weather, future_doy, future_hour,
    clim_table, clim_valid, clim_global_mean,
    step_minutes=60, alpha_end=0.2, horizon_hours=96
):
    """
    Create future weather proxy by blending last observation with climatology
    
    Args:
        last_weather: (W,) - last observed weather vector
        future_doy, future_hour: (H,) - day of year and hour for forecast
        clim_table: (367, 24, W) - climatology lookup table
        clim_valid: (367, 24) - validity mask
        clim_global_mean: (W,) - global mean weather
    
    Returns:
        (H, W) - proxy weather for forecast horizon
    """
    future_doy = future_doy.astype(np.int16)
    future_hour = future_hour.astype(np.int8)
    
    H = len(future_doy)
    
    # Calculate alpha values for exponential decay
    lead_hours = (np.arange(1, H + 1) * step_minutes) / 60.0
    alphas = alpha_exponential(lead_hours, alpha_end=alpha_end, horizon_hours=horizon_hours).astype(np.float32)
    
    # Get climatology vectors for each future timestep
    clim_vecs = clim_table[future_doy, future_hour, :]  # (H, W)
    valid = clim_valid[future_doy, future_hour]          # (H,)
    
    # Replace missing climatology with global mean
    if not valid.all():
        clim_vecs = clim_vecs.copy()
        clim_vecs[~valid] = clim_global_mean
    
    # Blend: starts close to last_weather, decays toward climatology
    proxy = alphas[:, None] * last_weather[None, :] + (1.0 - alphas)[:, None] * clim_vecs
    return proxy.astype(np.float32)