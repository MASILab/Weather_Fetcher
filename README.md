# WeatherAPI

Fetch weather, air quality, and water quality data using Open-Meteo.

## Usage

### Option 1: API Only

Use `API_weather_tool.py` to fetch all data directly from APIs.

### Option 2: Local Docker + APIs

Use a local Docker container for weather data and fetch air/water quality from APIs.

**Setup:**

1. Create a Docker volume:
   ```bash
    # Create a Docker volume to store weather data
    docker volume create --name open-meteo-data # this is in /var/lib/docker/volumes

    # Create a Docker volume on a specific path
    docker volume create \
      --driver local \
      --opt type=none \
      --opt device=<path_to_folder>/open-meteo-historical \
      --opt o=bind \
        open-meteo-historical

   ```

2. Run `OpenMeteoDataFetcher.sh` to download weather data
   - Set `DAYS` to specify how many days of historical data to fetch

3. Use `LocalOpenMeteoAPI.py` to fetch data from the local Docker container and APIs

**Note:** The fetcher uses Copernicus satellite data by default. Edit the script to select other satellites.

## Resources

- [Open-Meteo Docker Tutorial](https://github.com/open-meteo/open-data/tree/main/tutorial_weather_api)
