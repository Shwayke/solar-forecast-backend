import requests
import json
from datetime import datetime
from typing import Optional, Dict, List
from src.config import API_URL, API_KEY

class IMSWeatherAPI:
    """
    Client for the Israeli Meteorological Service (IMS) Weather Data API
    """
    BASE_URL = API_URL
    def __init__(self):
        """
        Initialize the API client with your API token

        Args:
            api_token: Your API token from IMS
        """
        self.headers = {
            "Authorization": f"ApiToken {API_KEY}"
        }

    def get_hourly_data_by_date_range(self, station_id: int,
                                      from_year: int, from_month: int, from_day: int,
                                      to_year: int, to_month: int, to_day: int,
                                      channel_id: Optional[int] = None) -> Dict:
        """
        Get hourly data for a date range (filters 10-minute data to keep every 6th record)

        Args:
            station_id: Station number
            from_year, from_month, from_day: Start date
            to_year, to_month, to_day: End date
            channel_id: Optional channel number

        Returns:
            Dict with filtered data containing hourly intervals
        """
        # Get the full 10-minute data
        data = self.get_data_by_date_range(
            station_id, from_year, from_month, from_day,
            to_year, to_month, to_day, channel_id
        )

        # Filter to keep every 6th data point (hourly intervals)
        if 'data' in data and isinstance(data['data'], list):
            data['data'] = data['data'][::6]

        return data


    def get_data_by_date_range(self, station_id: int,
                               from_year: int, from_month: int, from_day: int,
                               to_year: int, to_month: int, to_day: int,
                               channel_id: Optional[int] = None) -> Dict:
        """
        Get data for a date range

        Args:
            station_id: Station number
            from_year, from_month, from_day: Start date
            to_year, to_month, to_day: End date
            channel_id: Optional channel number
        """
        from_date = f"{from_year}/{from_month:02d}/{from_day:02d}"
        to_date = f"{to_year}/{to_month:02d}/{to_day:02d}"

        if channel_id:
            url = f"{self.BASE_URL}/stations/{station_id}/data/{channel_id}?from={from_date}&to={to_date}"
        else:
            url = f"{self.BASE_URL}/stations/{station_id}/data?from={from_date}&to={to_date}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()