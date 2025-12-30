from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
from src.fetchers.IMSWeatherAPI import IMSWeatherAPI

class BaseStation(ABC):
    """
    Abstract base class for station-specific wrappers
    """

    def __init__(self, api_client: IMSWeatherAPI, station_id: int, station_name: str):
        """
        Initialize the station wrapper

        Args:
            api_client: Instance of IMSWeatherAPI
            station_id: The station ID for this wrapper
            station_name: The station name for this wrapper
        """
        self.api = api_client
        self.station_id = station_id
        self.station_name = station_name
        self._station_info = None

    def get_station_info(self) -> Dict:
        """Get and cache station information"""
        if self._station_info is None:
            self._station_info = self.api.get_station_info(self.station_id)
        return self._station_info

    def get_raw_data(self, from_date: datetime, to_date: datetime,
                     channel_id: Optional[int] = None) -> Dict:
        """
        Get raw data from the API

        Args:
            from_date: Start date
            to_date: End date
            channel_id: Optional channel ID
        """
        return self.api.get_hourly_data_by_date_range(
            station_id=self.station_id,
            from_year=from_date.year,
            from_month=from_date.month,
            from_day=from_date.day,
            to_year=to_date.year,
            to_month=to_date.month,
            to_day=to_date.day,
            channel_id=channel_id
        )

    @abstractmethod
    def get_normalized_data(self, from_date: datetime, to_date: datetime) -> List[Dict]:
        """
        Get data in a normalized format

        Returns:
            List of dictionaries with standardized keys:
            {
                'timestamp': datetime,
                'temperature': float,
                'humidity': float,
                'solar radiation': float
                'wind_speed': float,
                'wind_direction': float
            }
        """
        pass

    @abstractmethod
    def _parse_station_data(self, raw_data: Dict) -> List[Dict]:
        """
        Parse station-specific raw data into normalized format

        Args:
            raw_data: Raw API response

        Returns:
            List of normalized data dictionaries
        """
        pass
