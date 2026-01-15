import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CLIMATOLOGY = pd.read_csv(os.path.join(MODELS_DIR, 'climatology.csv'))
WEATHER_COLUMNS = ['temperature', 'humidity', 'solar_radiation', 'pressure']

def get_proxy(data):
    last_weather = data[WEATHER_COLUMNS].iloc[-1].values

    last_datetime = pd.to_datetime(data['date_time'].iloc[-1])
    future_dates = pd.date_range(
        start=last_datetime + pd.Timedelta(hours=1),
        periods=96,  # 96 hours = 4 days
        freq='H'
    )
    future_doy = future_dates.dayofyear.values
    future_hour = future_dates.hour.values

    proxy = make_future_weather_proxy(
        last_weather=last_weather,
        future_doy=future_doy,
        future_hour=future_hour,
        step_minutes=60,
        alpha_end=0.2,
        horizon_hours=96
    )

    return pd.DataFrame(
        proxy,
        columns=WEATHER_COLUMNS,
        index=future_dates
    )

def alpha_exponential(lead_hours, alpha_end=0.2, horizon_hours=96):
    """Calculate exponential decay for weather blending"""
    k = -np.log(alpha_end) / horizon_hours
    return np.exp(-k * lead_hours).astype(np.float32)

def make_future_weather_proxy(
    last_weather, future_doy, future_hour,
    step_minutes=60, alpha_end=0.2, horizon_hours=96
):
    """
    last_weather: (W,) array or dict - last observed weather
    future_doy, future_hour: (H,) arrays - day of year and hour for forecast
    climatology_df: pandas DataFrame indexed by (day_of_year, hour)
    returns: (H, W) array or DataFrame
    """

    H = len(future_doy)
    
    # Calculate alpha values for exponential decay
    lead_hours = (np.arange(1, H + 1) * step_minutes) / 60.0
    alphas = alpha_exponential(lead_hours, alpha_end=alpha_end, horizon_hours=horizon_hours)
    
    # Get global mean for fallback
    global_mean = CLIMATOLOGY.mean().values
    
    # Look up climatology for each future timestep
    clim_vecs = []
    for doy, hr in zip(future_doy, future_hour):
        try:
            # Try to get climatology for this (doy, hour)
            clim_vec = CLIMATOLOGY.loc[(doy, hr)].values
        except KeyError:
            # If missing, use global mean
            clim_vec = global_mean
        clim_vecs.append(clim_vec)
    
    clim_vecs = np.array(clim_vecs)  # (H, W)
    
    # Convert last_weather to array if needed
    if isinstance(last_weather, dict):
        last_weather = np.array([last_weather[col] for col in CLIMATOLOGY.columns])
    
    # Blend: starts close to last_weather, decays toward climatology
    proxy = alphas[:, None] * last_weather[None, :] + (1.0 - alphas)[:, None] * clim_vecs
    
    return proxy
