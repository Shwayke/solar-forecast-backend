import os
import numpy as np
import pandas as pd
from src.model_loader import MODELS_DIR

CLIMATOLOGY = pd.read_csv(os.path.join(MODELS_DIR, 'climatology.csv')).set_index(["day_of_year", "hour"])
WEATHER_COLUMNS = ['temperature', 'humidity', 'solar_radiation', 'pressure']

def get_proxy(data):
    last_datetime = pd.to_datetime(data['date_time'].iloc[-1])
    future_dates = pd.date_range(
        start=last_datetime + pd.Timedelta(hours=1),
        periods=96,  # 4 days
        freq='h'
    )
    future_doy = future_dates.dayofyear.values
    future_hour = future_dates.hour.values

    proxy = make_future_weather_proxy(
        future_doy=future_doy,
        future_hour=future_hour
    )

    return pd.DataFrame(
        proxy,
        columns=WEATHER_COLUMNS,
        index=future_dates
    )

def make_future_weather_proxy(future_doy, future_hour):
    """
    future_doy, future_hour: arrays - day of year and hour for forecast
    returns: array or DataFrame
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
    
    proxy = np.array(clim_vecs)
    
    return proxy
