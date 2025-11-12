#!/bin/bash

# Set the path to your data volume
DATA_PATH="open-meteo-historical"
DAYS="40000"

# Define models and their groups of variables
declare -A models

# Since 2017
# models["ecmwf_ifs"]="boundary_layer_height,cloud_cover,cloud_cover_high,cloud_cover_low,cloud_cover_mid,dew_point_2m,direct_radiation,precipitation,pressure_msl,sea_surface_temperature,shortwave_radiation,snowfall_water_equivalent,soil_moisture_0_to_7cm,soil_moisture_100_to_255cm,soil_moisture_28_to_100cm,soil_moisture_7_to_28cm,soil_temperature_0_to_7cm,soil_temperature_100_to_255cm,soil_temperature_28_to_100cm,soil_temperature_7_to_28cm,static,temperature_2m,total_column_integrated_water_vapour,wind_gusts_10m,wind_u_component_100m,wind_u_component_10m,wind_v_component_100m,wind_v_component_10m"

# Since 1940
models["copernicus_era5"]="boundary_layer_height,cloud_cover,cloud_cover_high,cloud_cover_low,cloud_cover_mid,dew_point_2m,direct_radiation,precipitation,pressure_msl,sea_surface_temperature,shortwave_radiation,snowfall_water_equivalent,soil_moisture_0_to_7cm,soil_moisture_100_to_255cm,soil_moisture_28_to_100cm,soil_moisture_7_to_28cm,soil_temperature_0_to_7cm,soil_temperature_100_to_255cm,soil_temperature_28_to_100cm,soil_temperature_7_to_28cm,static,temperature_2m,total_column_integrated_water_vapour,wind_gusts_10m,wind_u_component_100m,wind_u_component_10m,wind_v_component_100m,wind_v_component_10m"

# 2-5 years fine resolution models
# # GFS model
# models["ncep_gfs013"]="diffuse_radiation,latent_heat_flux,precipitation,relative_humidity_2m,sensible_heat_flux,shortwave_radiation,showers,surface_temperature,temperature_2m,total_column_integrated_water_vapour,uv_index,uv_index_clear_sky,wind_u_component_10m,wind_v_component_10m"

# # HRRR model
# models["ncep_hrrr_conus"]="diffuse_radiation,latent_heat_flux,precipitation,relative_humidity_2m,sensible_heat_flux,shortwave_radiation,surface_temperature,temperature_2m,total_column_integrated_water_vapour,wind_u_component_10m,wind_v_component_10m"

# # NBM model
# models["ncep_nbm_conus"]="cape,precipitation,relative_humidity_2m,shortwave_radiation,surface_temperature,temperature_2m,wind_direction_10m,wind_gusts_10m,wind_speed_10m"

# Process each model and its variable groups
for model_key in "${!models[@]}"; do
    # Extract base model name (remove any suffix like _2, _3)
    base_model=$(echo "$model_key" | sed 's/_[0-9]*$//')
    
    # Get variables for this model group
    variables="${models[$model_key]}"
    
    echo "Syncing model: $base_model with variables: $variables"
    
    # Run the sync command for this model and its variables
    docker run -it --rm -v $DATA_PATH:/app/data ghcr.io/open-meteo/open-meteo sync \
        "$base_model" \
        "$variables" \
        --past-days $DAYS
    
    # Add a small delay between requests to prevent potential rate limiting
    sleep 1
done

echo "Weather data synced successfully"
