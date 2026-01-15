import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CLIMATOLOGY = pd.read_csv(os.path.join(MODELS_DIR, 'climatology.csv')).set_index(["day_of_year", "hour"])
WEATHER_COLUMNS = ['temperature', 'humidity', 'solar_radiation', 'pressure']

def get_proxy(data):
    last_datetime = pd.to_datetime(data['date_time'].iloc[-1])
    future_dates = pd.date_range(
        start=last_datetime + pd.Timedelta(hours=1),
        periods=96,  # 96 hours = 4 days
        freq='H'
    )
    future_doy = future_dates.dayofyear.values
    future_hour = future_dates.hour.values

    proxy = make_future_weather_proxy(
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

def make_future_weather_proxy(
    future_doy, future_hour,
    step_minutes=60, alpha_end=0.2, horizon_hours=96
):
    """
    last_weather: (W,) array or dict - last observed weather
    future_doy, future_hour: (H,) arrays - day of year and hour for forecast
    climatology_df: pandas DataFrame indexed by (day_of_year, hour)
    returns: (H, W) array or DataFrame
    """
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
    
    proxy = np.array(clim_vecs)  # (H, W)
    
    return proxy
