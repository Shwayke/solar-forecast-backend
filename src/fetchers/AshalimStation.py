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

        # Calculate how many hours to skip from start and end
        hours_to_skip_start = from_date.hour
        hours_to_skip_end = 23 - to_date.hour
        
        # Filter the data list
        if 'data' in raw_data and isinstance(raw_data['data'], list):
            if hours_to_skip_end == 0:
                raw_data['data'] = raw_data['data'][hours_to_skip_start:]
            else:
                raw_data['data'] = raw_data['data'][hours_to_skip_start:-hours_to_skip_end]
        

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
                'temperature': channels_dict.get('TD'),  # TD = Temperature
                'humidity': channels_dict.get('RH'),  # RH = Relative Humidity
                'solar_radiation': channels_dict.get('Grad'),  # Grad = Solar Radiation
                'pressure': channels_dict.get('BP'),  # BP = Pressure
            }

            normalized.append(normalized_record)

        return normalized