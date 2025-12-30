from typing import Dict, List
from datetime import datetime
from src.fetchers import IMSWeatherAPI
from src.fetchers.BaseStation import BaseStation

class AshalimStation(BaseStation):

    def __init__(self, api_client: IMSWeatherAPI):
        super().__init__(api_client, station_id=381, station_name="Ashalim")

    def get_normalized_data(self, from_date: datetime, to_date: datetime) -> List[Dict]:
        """Get normalized data for Station 381"""
        raw_data = self.get_raw_data(from_date, to_date)
        return self._parse_station_data(raw_data)

    def _parse_station_data(self, raw_data: Dict) -> List[Dict]:

        normalized = []

        data_records = raw_data.get('data', [])

        for record in data_records:
            # Create a dictionary to easily access channel values by name
            channels_dict = {}
            for channel in record.get('channels', []):
                if channel.get('valid', False):  # Only use valid data
                    channels_dict[channel['name']] = channel['value']

            # Map the channels to standardized fields
            normalized_record = {
                'date_time': datetime.fromisoformat(record.get('datetime')),
                'temperature': channels_dict.get('TD'),  # TD = Temperature (Dew point or Temperature)
                'humidity': channels_dict.get('RH'),  # RH = Relative Humidity
                'solar_radiation': channels_dict.get('Grad'),  # Grad = Solar Radiation
                'wind_speed': channels_dict.get('WS'),  # WS = Wind Speed
                'wind_direction': channels_dict.get('WD')  # WD = Wind Direction
            }

            normalized.append(normalized_record)

        return normalized