import pandas as pd
from ast import Dict
from datetime import datetime, timedelta
from typing import List
from src.fetchers.AshalimStation import AshalimStation
from src.fetchers.IMSWeatherAPI import IMSWeatherAPI

def fetch_data():
    """Fetch data from external API"""

    print("Fetching data from IMS API...")

    api_client = IMSWeatherAPI()

    station_ashalim = AshalimStation(api_client)

    from_date, to_date = get_time_range()

    print(f"Fetching data from {from_date} to {to_date}...")

    data = station_ashalim.get_normalized_data(from_date, to_date)

    print(f"Fetched {len(data)} records.")

    return fill_data_gaps(data)


def get_time_range():
    """
    Get current time rounded to last 10 minutes and 14 days prior.
    
    Returns:
        tuple: (end_time, start_time) both as datetime objects
    """
    now = datetime.now()
    
    # Round down to last 10-minute interval
    minutes = (now.minute // 10) * 10
    end_time = now.replace(minute=minutes, second=0, microsecond=0)
    
    # Get time exactly 14 days prior
    start_time = end_time - timedelta(days=14)
    
    return start_time, end_time

def fill_data_gaps(data: List[Dict], max_gap_hours: int = 6) -> pd.DataFrame:
    """
    Fill gaps in weather data using time-based interpolation.
    Small gaps: interpolate
    Large gaps: fill with hourly averages (same hour of day)
    """
    df = pd.DataFrame(data)
    df['date_time'] = pd.to_datetime(df['date_time'], utc=True)
    df.set_index('date_time', inplace=True)
    df.sort_index(inplace=True)
    
    numeric_cols = ['temperature', 'humidity', 'solar_radiation', 'pressure']
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
            
        # Step 1: Interpolate small gaps (up to max_gap_hours)
        df[col] = df[col].interpolate(method='time', limit=max_gap_hours)
        
        # Step 2: Fill remaining with hourly averages
        # (e.g., missing 2pm value gets average of all 2pm readings)
        hourly_avg = df.groupby(df.index.hour)[col].transform('mean')
        df[col] = df[col].fillna(hourly_avg)
        
        # Step 3: Fallback - if still any NaN (shouldn't happen, but just in case)
        df[col] = df[col].fillna(df[col].mean())

    df.reset_index(inplace=True)
    
    return df