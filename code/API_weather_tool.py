"""
Here I did my own (Andre's version) version of how to collect and cache the weather data. I've maintined the csv fetching and add a way of downloading only the air quality or the weather data or both.
This functionality--selecting only air quality to be downloaded--would pair well with the Open-Meteo local host, with the 80 years worth of weather data (Corpernicus).
The AirNow API has a lot of restrictions for big data collection (Don't know how big we can go yet).
"""

# TODO: - I think my data colleciton is not being done on the specified date range, but I could not test it since the NOAA website is down for maintenance (Which could be why the fetching was wonky)

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import argparse
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class LocationInfo:
    latitude: float
    longitude: float
    name: str
    bounding_box: List[float]


class WeatherCollector:
    """Weather and air quality data collector"""
    
    def __init__(self):
        load_dotenv()
        self.airnow_api_key = os.getenv('AIRNOW_API_KEY')
        self.nominatim_user_agent = os.getenv('NOMINATIM_USER_AGENT')
        
        if not self.nominatim_user_agent:
            raise ValueError("NOMINATIM_USER_AGENT not found in .env file")
        
        self.geolocator = Nominatim(user_agent=self.nominatim_user_agent)
    
    def resolve_location(self, location_input: str) -> LocationInfo:
        """Convert location string to LocationInfo"""
        # Handle coordinate format: "lat40.7128_lon-74.0060"
        if location_input.startswith("lat") and "_lon" in location_input:
            parts = location_input.replace("lat", "").split("_lon")
            if len(parts) == 2:
                lat, lon = float(parts[0]), float(parts[1])
                # bbox format: [south, north, west, east]
                bbox = [lat-0.5, lat+0.5, lon-0.5, lon+0.5]
                return LocationInfo(lat, lon, location_input, bbox)
        
        # Handle city name
        location = self.geolocator.geocode(location_input, exactly_one=True)
        if location:
            # Nominatim returns boundingbox as [south, north, west, east]
            bbox = [float(x) for x in location.raw['boundingbox']]
            return LocationInfo(location.latitude, location.longitude, location_input, bbox)
        
        raise ValueError(f"Could not resolve location: {location_input}")
    
    def get_weather_search_urls(self, location: LocationInfo, start_date: str, end_date: str) -> List[str]:
        """Get NOAA search URLs for all pages"""
        bbox = location.bounding_box
        base_url = (
            f"https://www.ncei.noaa.gov/access/search/data-search/local-climatological-data"
            f"?bbox={bbox[1]},{bbox[2]},{bbox[0]},{bbox[3]}"
            f"&startDate={start_date}T00:00:00&endDate={end_date}T23:59:59"
        )
        
        # Get first page to determine total pages
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        try:
            driver.get(base_url + "&pageNum=1")
            time.sleep(3)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            page_info = soup.find(string=re.compile(r"Page\s+\d+\s+of\s+\d+"))
            
            if not page_info:
                return [base_url + "&pageNum=1"]
            
            match = re.search(r"Page\s+\d+\s+of\s+(\d+)", page_info)
            total_pages = int(match.group(1)) if match else 1
            
            return [base_url + f"&pageNum={i}" for i in range(1, total_pages + 1)]
        finally:
            driver.quit()
    
    def get_csv_download_links(self, search_urls: List[str]) -> List[str]:
        """Extract CSV download links from search pages"""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        csv_links = []
        try:
            for url in search_urls:
                print(f"Processing search page: {url}")
                driver.get(url)
                
                # Wait for page content to load
                time.sleep(3)
                
                # Look for CSV download links
                links = driver.find_elements(By.CSS_SELECTOR, ".card-header a")
                for link in links:
                    href = link.get_attribute("href")
                    if href and href not in csv_links:
                        csv_links.append(href)
                        
                # If no links found with card-header, try broader search
                if not links:
                    links = driver.find_elements(By.TAG_NAME, "a")
                    for link in links:
                        href = link.get_attribute("href")
                        if href and (".csv" in href or "download" in href.lower()) and href not in csv_links:
                            csv_links.append(href)
                            
        except Exception as e:
            print(f"Error extracting CSV links: {e}")
        finally:
            driver.quit()
        
        return csv_links
    
    def fetch_csv_data(self, url: str) -> Optional[pd.DataFrame]:
        """Fetch individual CSV file"""
        response = requests.get(url)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text), low_memory=False)
        return None
    
    def collect_weather_data(self, location: LocationInfo, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect weather data for location and date range"""
        print(f"Collecting weather data for {location.name} from {start_date} to {end_date}")
        
        # Get search URLs
        search_urls = self.get_weather_search_urls(location, start_date, end_date)
        print(f"Found {len(search_urls)} search pages")
        
        # Get CSV download links
        csv_links = self.get_csv_download_links(search_urls)
        print(f"Found {len(csv_links)} CSV files")
        
        if not csv_links:
            return pd.DataFrame()
        
        # Fetch all CSV files concurrently
        dataframes = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self.fetch_csv_data, url): url for url in csv_links}
            
            for future in as_completed(future_to_url):
                df = future.result()
                if df is not None and not df.empty:
                    dataframes.append(df)
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            print(f"Collected {len(combined_df)} weather records")
            return combined_df
        
        return pd.DataFrame()
    
    def collect_air_quality_data(self, location: LocationInfo, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect air quality data"""
        if not self.airnow_api_key:
            print("No AirNow API key found, skipping air quality data")
            return pd.DataFrame()
        
        print(f"Collecting air quality data for {location.name}")
        
        bbox = location.bounding_box
        # AirNow API expects: west,south,east,north
        west, south, east, north = bbox[2], bbox[0], bbox[3], bbox[1]
        
        url = (
            f"https://www.airnowapi.org/aq/data/?startDate={start_date}T0&endDate={end_date}T23"
            f"&parameters=OZONE,PM25,PM10,CO,NO2,SO2"
            f"&BBOX={west},{south},{east},{north}"
            f"&dataType=B&format=application/json&verbose=1&monitorType=2"
            f"&includerawconcentrations=1&API_KEY={self.airnow_api_key}"
        )
        
        print(f"AirNow API URL: {url}")
        response = requests.get(url)
        print(f"AirNow API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                print(f"Collected {len(df)} air quality records")
                return df
            else:
                print("AirNow API returned empty data")
        else:
            print(f"AirNow API error: {response.status_code} - {response.text}")
        
        return pd.DataFrame()
    
    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: str, location_name: str):
        """Save data to CSV files using incremental approach"""
        os.makedirs(output_dir, exist_ok=True)
        
        for data_type, df in data.items():
            if not df.empty:
                filename = f"{location_name}_{data_type}.csv"
                filepath = os.path.join(output_dir, filename)
                self.append_data_to_file(df, filepath)
            else:
                print(f"No new {data_type} data to save")
    
    def check_existing_data(self, filepath: str, start_date: str, end_date: str) -> List[str]:
        """Check existing file and return list of missing dates"""
        if not os.path.exists(filepath):
            return self.generate_date_range(start_date, end_date)
        
        try:
            existing_df = pd.read_csv(filepath)
            if existing_df.empty:
                return self.generate_date_range(start_date, end_date)
            
            # Find date column (different sources use different column names)
            date_column = None
            possible_date_columns = ['DATE', 'Date', 'date', 'DateLocal', 'ValidDate', 'datetime', 'UTC', 'UTC_datetime']
            for col in possible_date_columns:
                if col in existing_df.columns:
                    date_column = col
                    break
            
            if not date_column:
                print(f"No date column found in {filepath}, fetching all data")
                return self.generate_date_range(start_date, end_date)
            
            # Extract existing dates
            existing_dates = pd.to_datetime(existing_df[date_column]).dt.strftime('%Y-%m-%d').unique()
            existing_dates_set = set(existing_dates)
            
            # Generate requested date range
            requested_dates = self.generate_date_range(start_date, end_date)
            
            # Find missing dates
            missing_dates = [date for date in requested_dates if date not in existing_dates_set]
            
            if missing_dates:
                print(f"Found {len(existing_dates)} existing dates, need {len(missing_dates)} more")
            else:
                print(f"All requested dates already exist in {filepath}")
            
            return missing_dates
            
        except Exception as e:
            print(f"Error reading existing file {filepath}: {e}")
            return self.generate_date_range(start_date, end_date)
    
    def generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """Generate list of dates between start_date and end_date"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        return [date.strftime('%Y-%m-%d') for date in date_range]
    
    def append_data_to_file(self, new_data: pd.DataFrame, filepath: str):
        """Append new data to existing file, or create new file"""
        if os.path.exists(filepath) and not new_data.empty:
            try:
                existing_data = pd.read_csv(filepath)
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                
                # Remove duplicates if they exist (based on all columns)
                combined_data = combined_data.drop_duplicates()
                
                combined_data.to_csv(filepath, index=False)
                print(f"Appended {len(new_data)} new rows to {filepath} (total: {len(combined_data)} rows)")
            except Exception as e:
                print(f"Error appending to {filepath}: {e}, saving as new file")
                new_data.to_csv(filepath, index=False)
        elif not new_data.empty:
            new_data.to_csv(filepath, index=False)
            print(f"Created new file {filepath} with {len(new_data)} rows")
    
    def collect_weather_data_incremental(self, location: LocationInfo, start_date: str, end_date: str, filepath: str) -> pd.DataFrame:
        """Collect weather data incrementally, only fetching missing dates"""
        missing_dates = self.check_existing_data(filepath, start_date, end_date)
        
        if not missing_dates:
            return pd.DataFrame()  # No new data needed
        
        # Group consecutive dates into ranges to minimize API calls
        date_ranges = self.group_consecutive_dates(missing_dates)
        
        all_new_data = []
        for date_start, date_end in date_ranges:
            print(f"Fetching weather data for date range: {date_start} to {date_end}")
            data = self.collect_weather_data(location, date_start, date_end)
            if not data.empty:
                all_new_data.append(data)
        
        if all_new_data:
            combined_new_data = pd.concat(all_new_data, ignore_index=True)
            return combined_new_data
        
        return pd.DataFrame()
    
    def collect_air_quality_data_incremental(self, location: LocationInfo, start_date: str, end_date: str, filepath: str) -> pd.DataFrame:
        """Collect air quality data incrementally, only fetching missing dates"""
        missing_dates = self.check_existing_data(filepath, start_date, end_date)
        
        if not missing_dates:
            return pd.DataFrame()  # No new data needed
        
        # Group consecutive dates into ranges to minimize API calls
        date_ranges = self.group_consecutive_dates(missing_dates)
        
        all_new_data = []
        for date_start, date_end in date_ranges:
            print(f"Fetching air quality data for date range: {date_start} to {date_end}")
            data = self.collect_air_quality_data(location, date_start, date_end)
            if not data.empty:
                all_new_data.append(data)
        
        if all_new_data:
            combined_new_data = pd.concat(all_new_data, ignore_index=True)
            return combined_new_data
        
        return pd.DataFrame()
    
    def group_consecutive_dates(self, dates: List[str]) -> List[tuple]:
        """Group consecutive dates into ranges to minimize API calls"""
        if not dates:
            return []
        
        dates.sort()
        date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        
        ranges = []
        start_date = date_objects[0]
        end_date = date_objects[0]
        
        for i in range(1, len(date_objects)):
            if (date_objects[i] - end_date).days == 1:
                end_date = date_objects[i]
            else:
                ranges.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
                start_date = date_objects[i]
                end_date = date_objects[i]
        
        ranges.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        return ranges


def main():
    parser = argparse.ArgumentParser(description="Weather and Air Quality Data Collection")
    
    # Location input
    location_group = parser.add_mutually_exclusive_group()
    location_group.add_argument("--city", type=str, help="City name or coordinates (lat40.7128_lon-74.0060)")
    location_group.add_argument("--csv-file", type=str, help="CSV file with locations")
    
    # Date parameters
    parser.add_argument("--date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    
    # Data selection
    parser.add_argument("--weather-only", action="store_true", help="Collect only weather data")
    parser.add_argument("--air-quality-only", action="store_true", help="Collect only air quality data")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="weather_data", help="Output directory")
    parser.add_argument("--init-env", action="store_true", help="Create .env template")
    
    args = parser.parse_args()
    
    if args.init_env:
        if not os.path.exists('.env'):
            with open('.env', 'w') as f:
                f.write("# Nominatim User Agent (Required)\n")
                f.write("NOMINATIM_USER_AGENT=your_unique_user_agent_here\n\n")
                f.write("# AirNow API Key (Optional)\n")
                f.write("AIRNOW_API_KEY=your_airnow_api_key_here\n")
            print("Created .env template file")
        return
    
    # Validate arguments for data collection
    if not args.city and not args.csv_file:
        parser.error("Either --city or --csv-file is required for data collection")
    
    # Validate arguments
    if args.city and not args.date:
        parser.error("--date is required when using --city")
    
    # Determine what data to collect
    collect_weather = not args.air_quality_only
    collect_air_quality = not args.weather_only
    
    # Initialize collector
    collector = WeatherCollector()
    
    if args.city:
        # Single location mode
        location = collector.resolve_location(args.city)
        end_date = args.end_date or args.date
        location_name = args.city.replace("lat", "").replace("_lon", "_")
        
        data = {}
        
        if collect_weather:
            weather_filepath = os.path.join(args.output_dir, f"{location_name}_weather.csv")
            weather_data = collector.collect_weather_data_incremental(location, args.date, end_date, weather_filepath)
            if not weather_data.empty:
                data['weather'] = weather_data
        
        if collect_air_quality:
            air_quality_filepath = os.path.join(args.output_dir, f"{location_name}_air_quality.csv")
            air_quality_data = collector.collect_air_quality_data_incremental(location, args.date, end_date, air_quality_filepath)
            if not air_quality_data.empty:
                data['air_quality'] = air_quality_data
        
        if data:
            collector.save_data(data, args.output_dir, location_name)
        else:
            print("No new data to collect")
    
    elif args.csv_file:
        # Batch processing mode
        df = pd.read_csv(args.csv_file)
        
        # Check required columns
        required_cols = ['latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {', '.join(missing_cols)}")
        
        # Check for date columns
        if 'start_date' not in df.columns and not args.date:
            raise ValueError("CSV must contain 'start_date' column or --date must be provided")
        
        for idx, row in df.iterrows():
            lat, lon = float(row['latitude']), float(row['longitude'])
            location_id = f"lat{lat:.4f}_lon{lon:.4f}"
            
            start_date = row.get('start_date', args.date)
            end_date = row.get('end_date', args.end_date or start_date)
            
            if not start_date:
                print(f"Skipping row {idx+1}: No start_date")
                continue
            
            print(f"\nProcessing location {idx+1}/{len(df)}: {location_id}")
            
            location = collector.resolve_location(location_id)
            location_dir = os.path.join(args.output_dir, location_id)
            
            data = {}
            
            if collect_weather:
                weather_filepath = os.path.join(location_dir, f"{location_id}_weather.csv")
                weather_data = collector.collect_weather_data_incremental(location, start_date, end_date, weather_filepath)
                if not weather_data.empty:
                    data['weather'] = weather_data
            
            if collect_air_quality:
                air_quality_filepath = os.path.join(location_dir, f"{location_id}_air_quality.csv")
                air_quality_data = collector.collect_air_quality_data_incremental(location, start_date, end_date, air_quality_filepath)
                if not air_quality_data.empty:
                    data['air_quality'] = air_quality_data
            
            if data:
                collector.save_data(data, location_dir, location_id)
            else:
                print(f"No new data to collect for {location_id}")
            
            # Rate limiting
            if idx < len(df) - 1:
                time.sleep(2)


if __name__ == '__main__':
    main()