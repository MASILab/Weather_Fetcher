WeatherAPI

There are two ways of fetching the data:

1- Using only APIs.

2- Using a local docker for the weather data and calling the APIs for air quality and water quality data.

For 1:

Use API_weather_tool.py

For 2:

Create a local docker for OpenMeteo data:

    # Create a Docker volume to store weather data
    docker volume create --name open-meteo-data # this is in /var/lib/docker/volumes

    # Create a Docker volume on a specific path
    docker volume create \
      --driver local \
      --opt type=none \
      --opt device=<path_to_folder>/open-meteo-historical \
      --opt o=bind \
        open-meteo-historical

Use OpenMeteoDataFetcher.sh to get all the data from OpenMeteo 

Uncomment to select other satelites. I use corpernicus.

    DAYS represent how many days backwards the fetcher should look into: today - DAYS

Use LocalOpenMeteoAPI.py to fetch data from inside the OpenMeteo docker and air + water APIs

More info on building the docker: https://github.com/open-meteo/open-data/tree/main/tutorial_weather_api
