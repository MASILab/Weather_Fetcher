import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError
from http.client import RemoteDisconnected
import json
import os
from pathlib import Path
import time
import logging
from dotenv import load_dotenv
import collections
import datetime 
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def build_filename(lat, lon, record_id=None, start_date=None, end_date=None):
    """Build consistent filename for any data type"""
    filename = f"{record_id}" if record_id and not pd.isna(record_id) else f"lat{lat}_lon{lon}"
    if start_date and end_date:
        filename += f"_{start_date}_to_{end_date}"
    return filename

def file_exists(output_dir, data_type, filename, extension):
    """Check if file exists for given data type, creating directory if needed."""
    if not output_dir:
        return False

    folder_map = {'weather': 'Weather', 'air_quality': 'AQI', 'water_quality': 'WaterQuality'}
    folder = Path(output_dir) / folder_map[data_type]
    folder.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    file_path = folder / f"{filename}.{extension}"
    return file_path.exists()

def filter_existing_rows(df, output_dir, data_type, extension, date_columns):
    """Filter DataFrame to only include rows with missing output files"""
    if not output_dir:
        return df
    
    keep_rows = []
    for i, row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        record_id = row.get('ID', None) if 'ID' in df.columns else None
        start_date, end_date = LocalOpenMeteoAPI._extract_dates(None, row, date_columns)
        
        filename = build_filename(lat, lon, record_id, start_date, end_date)
        
        if not file_exists(output_dir, data_type, filename, extension):
            keep_rows.append(i)
    
    return df.loc[keep_rows]

class LocalOpenMeteoAPI:
    """Client for interacting with a locally hosted Open-Meteo API"""
    
    # Default parameters based on Open-Meteo's documentation
    DEFAULT_HOURLY = ["weather_code,boundary_layer_height,cloud_cover,cloud_cover_high,cloud_cover_low,cloud_cover_mid,dew_point_2m,precipitation,rain,relative_humidity_2m,pressure_msl,sea_surface_temperature,apparent_temperature,sunshine_duration,diffuse_radiation,direct_radiation,direct_normal_irradiance,global_tilted_irradiance,shortwave_radiation,snowfall,snow_depth,snowfall_water_equivalent,soil_moisture_0_to_7cm,soil_moisture_100_to_255cm,soil_moisture_28_to_100cm,soil_moisture_7_to_28cm,soil_temperature_0_to_7cm,soil_temperature_100_to_255cm,soil_temperature_28_to_100cm,soil_temperature_7_to_28cm,temperature_2m,total_column_integrated_water_vapour,vapour_pressure_deficit,surface_pressure,wind_speed_10m,wind_speed_100m,wind_direction_10m,wind_direction_100m,wind_gusts_10m,et0_fao_evapotranspiration"]
    
    DEFAULT_DAILY = ["weather_code,precipitation_sum,precipitation_hours,temperature_2m_max,temperature_2m_min,temperature_2m_mean,apparent_temperature_max,apparent_temperature_min,apparent_temperature_mean,shortwave_radiation_sum,sunshine_duration,daylight_duration,sunset,sunrise,rain_sum,snowfall_sum,wind_direction_10m_dominant,wind_speed_10m_max,wind_gusts_10m_max,et0_fao_evapotranspiration"]
    
    def __init__(self, base_url="http://127.0.0.1:8080/v1", model="copernicus_era5"):
        """
        Initialize the API client
        
        Args:
            base_url (str): Base URL of the local Open-Meteo API
            model (str): Default weather model to use
        """
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
    
    def get_archive(self, latitude, longitude, start_date=None, end_date=None,
                    hourly_params=None, daily_params=None, model=None, output_path=None):
        """
        Get weather archive for a specific location
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            start_date (str): Start date in format YYYY-MM-DD
            end_date (str): End date in format YYYY-MM-DD
            hourly_params (list): List of hourly parameters to request (defaults to all available)
            daily_params (list): List of daily parameters to request (defaults to all available)
            model (str): Weather model to use (defaults to instance default)
            output_path (Path): Path to save JSON response (optional)
            
        Returns:
            dict: JSON response from the API
        """
        model = model or self.model
        
        # Default to all available data if no parameters specified
        if hourly_params is None and daily_params is None:
            hourly_params = self.DEFAULT_HOURLY
            daily_params = self.DEFAULT_DAILY

        url = f"{self.base_url}/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "model": model,
            "hourly": hourly_params,
            "daily": daily_params
        }

        response = self.session.get(url, params=params)
        # print(response.url)
        response.raise_for_status()
        logger.info(f"Requesting archive for lat={latitude}, lon={longitude}")
        data = response.json()
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved response to {output_path}")
            
        return data
    
    def process_csv(self, csv_path, output_dir=None, delay=0.0, 
                    hourly_params=None, daily_params=None, model=None,
                    collect_weather=True, collect_air_quality=False, collect_water_quality=False):
        """Process a CSV file with lat/long coordinates"""
        df = pd.read_csv(csv_path)
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            raise ValueError("CSV must contain 'latitude' and 'longitude' columns")
        
        # Create output directories
        if output_dir:
            output_dir = Path(output_dir)
            for data_type in ['Weather', 'AQI', 'WaterQuality']:
                if (data_type == 'Weather' and collect_weather) or \
                   (data_type == 'AQI' and collect_air_quality) or \
                   (data_type == 'WaterQuality' and collect_water_quality):
                    (output_dir / data_type).mkdir(exist_ok=True, parents=True)
        
        # Find date columns
        date_columns = self._find_date_columns(df)
        
        # Filter for each data type separately
        weather_df = filter_existing_rows(df, output_dir, 'weather', 'json', date_columns) if collect_weather else pd.DataFrame()
        air_df = filter_existing_rows(df, output_dir, 'air_quality', 'csv', date_columns) if collect_air_quality else pd.DataFrame()
        water_df = filter_existing_rows(df, output_dir, 'water_quality', 'csv', date_columns) if collect_water_quality else pd.DataFrame()
        
        logger.info(f"Weather: {len(weather_df)}/{len(df)} files to download")
        logger.info(f"Air Quality: {len(air_df)}/{len(df)} files to download")
        logger.info(f"Water Quality: {len(water_df)}/{len(df)} files to download")
        
        # Initialize collectors
        collectors = {}
        if collect_air_quality:
            collectors['air_quality'] = AirQualityCollector_AirNow()
        if collect_water_quality:
            collectors['water_quality'] = WaterQualityCollector()
        
        results = {'weather': [], 'air_quality': [], 'water_quality': []}
        
        # Process each data type with its filtered DataFrame
        if collect_weather:
            for i, row in weather_df.iterrows():
                lat = row['latitude']
                lon = row['longitude']
                record_id = row.get('ID', None) if 'ID' in df.columns else None
                start_date, end_date = self._extract_dates(row, date_columns)
                
                weather_result = self._collect_weather_data(
                    lat, lon, record_id, start_date, end_date, 
                    hourly_params, daily_params, model, output_dir
                )
                results['weather'].append(weather_result)
                
                if i < len(weather_df) - 1 and delay > 0:
                    time.sleep(delay)
        
        # Process other data types
        for data_type, collector in collectors.items():
            target_df = air_df if data_type == 'air_quality' else water_df
            
            for i, row in target_df.iterrows():
                lat = row['latitude']
                lon = row['longitude']
                record_id = row.get('ID', None) if 'ID' in df.columns else None
                start_date, end_date = self._extract_dates(row, date_columns)
                
                if start_date and end_date:
                    data_result = collector.collect_data(lat, lon, start_date, end_date)
                    self._save_data_result(data_result, data_type, lat, lon, record_id, 
                                         start_date, end_date, output_dir)
                    results[data_type].append(data_result)
                
                if i < len(target_df) - 1 and delay > 0:
                    time.sleep(delay)
        
        return results

    def _find_date_columns(self, df):
        """Find date columns with flexible naming"""
        date_columns = {}
        for col in df.columns:
            if col.lower() in ['start_date', 'start', 'begin_date', 'begin']:
                date_columns['start_date'] = col
            elif col.lower() in ['end_date', 'end', 'finish_date', 'finish']:
                date_columns['end_date'] = col
        return date_columns

    def _extract_dates(self, row, date_columns):
        """Extract start and end dates from row"""
        start_date = None
        end_date = None
        if 'start_date' in date_columns:
            start_date = row[date_columns['start_date']]
            if pd.isna(start_date):
                start_date = None
        if 'end_date' in date_columns:
            end_date = row[date_columns['end_date']]
            if pd.isna(end_date):
                end_date = None
        return start_date, end_date

    def _collect_weather_data(self, lat, lon, record_id, start_date, end_date, 
                            hourly_params, daily_params, model, output_dir):
        """Collect weather data for a location"""
        out_path = None
        if output_dir:
            filename = f"{record_id}" if record_id and not pd.isna(record_id) else f"lat{lat}_lon{lon}"
            if start_date and end_date:
                filename += f"_{start_date}_to_{end_date}"
            filename += ".json"
            out_path = output_dir / "Weather" / filename

        return self.get_archive(
            latitude=lat,
            longitude=lon,
            start_date=start_date,
            end_date=end_date,
            hourly_params=hourly_params,
            daily_params=daily_params,
            model=model,
            output_path=out_path
        )

    def _save_data_result(self, data_result, data_type, lat, lon, record_id, 
                         start_date, end_date, output_dir):
        """Save data result to file"""
        if not data_result.empty and output_dir:
            folder_map = {'air_quality': 'AQI', 'water_quality': 'WaterQuality'}
            folder = folder_map.get(data_type, data_type)
            
            filename = f"{record_id}" if record_id and not pd.isna(record_id) else f"lat{lat}_lon{lon}"
            filename += f"_{start_date}_to_{end_date}.csv"
            
            file_path = output_dir / folder / filename
            data_result.to_csv(file_path, index=False)
            logger.info(f"Saved {data_type} data to {file_path}")

class AirQualityCollector_EPA_AQS:
    """Collects air quality data from EPA AQS API"""

    def __init__(self, pause=30.0):
        self.eqsa_api_key = os.getenv('EPA_AQS_KEY')
        if not self.eqsa_api_key:
            logger.warning("EPA_AQS_KEY not found in environment variables")
        self.email = os.getenv('EMAIL')
        if not self.email:
            logger.warning("EMAIL not found in environment variables")
        self.aqi_params_path = os.getenv('AQI_PARAMS_PATH')
        if not self.aqi_params_path:
            logger.warning("AQI_PARAMS_PATH not found in environment variables, using default parameters (PM2.5 Raw data, Ozone, Carbon Monoxide, PM10 Total 0-10um STP)")
            self.params = ["88501", "44201", "42101", '81102']

        self.pause = pause
        self.request_times = collections.deque(maxlen=10)

    def collect_data(self, latitude, longitude, start_date, end_date):
        """Collect air quality data for given coordinates and date range"""
        if not self.eqsa_api_key:
            logger.warning("No EPA AQS API key available, skipping air quality data")
            return pd.DataFrame()

        if self.aqi_params_path and os.path.exists(self.aqi_params_path):
            param_df = pd.read_csv(self.aqi_params_path)
            params = param_df['param'].astype(str).tolist()
        else:
            params = self.params

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        date_ranges = self._split_date_range_by_quarter(start_date, end_date)
        
        all_records = []
        for date_start, date_end in date_ranges:
            records = self._collect_air_quality_data(
                params, latitude, longitude, date_start.strftime('%Y%m%d'), 
                date_end.strftime('%Y%m%d')
            )
            all_records.extend(records)

        if not all_records:
            logger.info(f"No air quality data for lat={latitude}, lon={longitude}")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df['request_latitude'] = latitude
        df['request_longitude'] = longitude
        logger.info(f"Collected {len(df)} air quality records for lat={latitude}, lon={longitude}")

        return df

    def _split_date_range_by_quarter(self, start_date, end_date):
        """Split date range into 3-month (quarterly) chunks"""
        ranges = []
        current_start = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        while current_start <= end_date:
            # Calculate the end of the current quarter
            next_month = ((current_start.month - 1) // 3 + 1) * 3 + 1
            if next_month > 12:
                quarter_end = pd.Timestamp(year=current_start.year, month=12, day=31)
            else:
                quarter_end = pd.Timestamp(year=current_start.year, month=next_month, day=1) - pd.Timedelta(days=1)
            current_end = min(quarter_end, end_date)
            ranges.append((current_start, current_end))
            current_start = current_end + pd.Timedelta(days=1)

        return ranges

    def _collect_air_quality_data(self, params, latitude, longitude, start_str, end_str):
        """Collect air quality data for a specific date range"""
        all_records = []
        margin = 1.0

        for i in range(0, len(params), 5):
            now = time.time()
            if len(self.request_times) == 2:
                elapsed = now - self.request_times[0]
                if elapsed < 60:
                    time.sleep(60 - elapsed)
            self.request_times.append(time.time())

            params_chunk = ','.join(params[i:i+5])
            url = (
                f"https://aqs.epa.gov/data/api/sampleData/byBox?"
                f"&email={self.email}&key={self.eqsa_api_key}"
                f"&param={params_chunk}"
                f"&bdate={start_str}&edate={end_str}"
                f"&minlat={latitude-margin}&maxlat={latitude+margin}&minlon={longitude-margin}&maxlon={longitude+margin}"
            )

            logger.info(f"Requesting air quality data for lat={latitude}, lon={longitude}, params={params_chunk}, start={start_str}, end={end_str}")
            while True:
                try:
                    response = requests.get(url)
                    data = response.json()
                    records = data.get("Data", [])
                    if records:
                        all_records.extend(records)
                    break
                except (RemoteDisconnected, requests.exceptions.RequestException, ValueError) as e:
                    logger.warning(f"Error occurred: {e}, retrying...")
                    time.sleep(60)

            time.sleep(self.pause)

        return all_records
    
class AirQualityCollector_AirNow:

    """Collects air quality data from AirNow API"""

    def __init__(self, pause=7.2):
        self.airnow_api_key = os.getenv('AIRNOW_API_KEY')
        if not self.airnow_api_key:
            logger.warning("AIRNOW_API_KEY not found in environment variables")
        self.pause = max(pause, 7.2)  # Enforce minimum pause for rate limit
        self.last_request_time = None

    def collect_data(self, latitude, longitude, start_date, end_date):
        """Collect air quality data for given coordinates and date range"""
        if not self.airnow_api_key:
            logger.warning("No AirNow API key available, skipping air quality data")
            return pd.DataFrame()
        
        bbox_margin = 0.5
        west = longitude - bbox_margin
        south = latitude - bbox_margin
        east = longitude + bbox_margin
        north = latitude + bbox_margin
        
        all_data = []
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        num_days = (end_date - start_date).days + 1

        for n in range(num_days):
            # Enforce rate limit
            if self.last_request_time:
                elapsed = time.time() - self.last_request_time
                if elapsed < self.pause:
                    time.sleep(self.pause - elapsed)
            self.last_request_time = time.time()

            day = start_date + datetime.timedelta(days=n)
            day_str = day.strftime("%Y-%m-%d")
            url = (
                f"https://www.airnowapi.org/aq/data/?startDate={day_str}T0&endDate={day_str}T23"
                f"&parameters=OZONE,PM25,PM10,CO,NO2,SO2"
                f"&BBOX={west},{south},{east},{north}"
                f"&dataType=B&format=application/json&verbose=1&monitorType=2"
                f"&includerawconcentrations=1&API_KEY={self.airnow_api_key}"
            )

            response = requests.get(url)
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df['request_latitude'] = latitude
                df['request_longitude'] = longitude
                all_data.append(df)
                logger.info(f"Collected {len(df)} air quality records for lat={latitude}, lon={longitude} on {day_str}")
            else:
                logger.info(f"No air quality data for lat={latitude}, lon={longitude} on {day_str}")

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()

class WaterQualityCollector:
    """Collects water quality data from waterqualitydata.us API"""
    
    def __init__(self, pause=5.0):
        self.base_url = "https://www.waterqualitydata.us/data/Result/search"
        self.pause = pause

    def collect_data(self, latitude, longitude, start_date, end_date):
        """Collect water quality data for given coordinates and date range"""
        params = {
            'countrycode': 'US',
            'within': '100',
            'lat': latitude,
            'long': longitude,
            'startDateLo': pd.to_datetime(start_date).strftime('%m-%d-%Y'),
            'startDateHi': pd.to_datetime(end_date).strftime('%m-%d-%Y'),
            'mimeType': 'csv',
            'zip': 'no',
            'dataProfile': 'resultPhysChem',
            'providers': 'NWIS',
            'providers': 'STORET'
        }

        while True:
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                break  # Success, exit loop
            except (ChunkedEncodingError, requests.exceptions.RequestException) as e:
                logger.warning(f"Request failed: {e}, retrying...")
                time.sleep(5)  # Wait before retrying

        # Parse CSV directly from response text
        df = pd.read_csv(pd.io.common.StringIO(response.text))
        
        if df.empty:
            logger.info(f"No water quality data for lat={latitude}, lon={longitude}")
            return pd.DataFrame()

        # Add request metadata
        df['request_latitude'] = latitude
        df['request_longitude'] = longitude
        logger.info(f"Collected {len(df)} water quality records for lat={latitude}, lon={longitude}")
        
        # Sleep to avoid overwhelming the API
        time.sleep(self.pause)
        
        return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Request data from local Open-Meteo API and AirNow API')
    parser.add_argument('--csv-file', required=True, help='CSV file with latitude and longitude columns (optional ID column for unique naming)')
    parser.add_argument('--output', '-o', help='Directory to save JSON responses')
    parser.add_argument('--model', default='copernicus_era5', help='Weather model to use')
    parser.add_argument('--hourly', help='Comma-separated hourly parameters (default: all available)')
    parser.add_argument('--daily', help='Comma-separated daily parameters (default: all available)')
    parser.add_argument('--hourly-only', action='store_true', help='Request only hourly data')
    parser.add_argument('--daily-only', action='store_true', help='Request only daily data')
    parser.add_argument('--delay', type=float, default=0.0, help='Delay between requests in seconds')
    parser.add_argument('--weather-only', action='store_true', help='Collect only weather data')
    parser.add_argument('--air-quality-only', action='store_true', help='Collect only air quality data')
    parser.add_argument('--water-quality-only', action='store_true', help='Collect only water quality data')
    parser.add_argument('--all', action='store_true', help='Collect weather, air quality, and water quality data')
    
    args = parser.parse_args()
    
    api = LocalOpenMeteoAPI(model=args.model)
    
    # Parse parameters
    hourly_params = args.hourly.split(',') if args.hourly else None
    daily_params = args.daily.split(',') if args.daily else None
    
    # Handle exclusive flags
    if args.hourly_only:
        daily_params = []
        if not hourly_params:
            hourly_params = api.DEFAULT_HOURLY
    elif args.daily_only:
        hourly_params = []
        if not daily_params:
            daily_params = api.DEFAULT_DAILY
    
    # Determine what data to collect
    collect_weather = not (args.air_quality_only or args.water_quality_only)
    collect_air_quality = args.air_quality_only or args.all
    collect_water_quality = args.water_quality_only or args.all
    
    results = api.process_csv(
        csv_path=args.csv_file,
        output_dir=args.output,
        delay=args.delay,
        hourly_params=hourly_params,
        daily_params=daily_params,
        model=args.model,
        collect_weather=collect_weather,
        collect_air_quality=collect_air_quality,
        collect_water_quality=collect_water_quality
    )
    
    weather_count = len(results['weather']) if results['weather'] else 0
    air_quality_count = len([df for df in results['air_quality'] if not df.empty]) if results['air_quality'] else 0
    water_quality_count = len([df for df in results['water_quality'] if not df.empty]) if results['water_quality'] else 0
    
    print(f"Processed {weather_count} weather locations, {air_quality_count} air quality locations, and {water_quality_count} water quality locations")