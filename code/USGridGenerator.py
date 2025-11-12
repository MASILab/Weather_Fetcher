import geopandas as gpd
from shapely.geometry import Point

import numpy as np
import pandas as pd

def generate_us_grid(grid_size_degrees=1.0):
    """
    Generate uniform grid coordinates covering the continental United States
    
    Args:
        grid_size_degrees (float): Grid cell size in degrees
        
    Returns:
        pandas.DataFrame: DataFrame with 'latitude' and 'longitude' columns
    """
    # Continental US approximate bounds
    lat_min, lat_max = 24.0, 50.0  # Florida to Canadian border
    lon_min, lon_max = -125.0, -66.0  # West coast to East coast
    
    # Generate coordinate arrays
    latitudes = np.arange(lat_min, lat_max + grid_size_degrees, grid_size_degrees)
    longitudes = np.arange(lon_min, lon_max + grid_size_degrees, grid_size_degrees)
    
    # Create meshgrid and flatten to coordinate pairs
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    
    return pd.DataFrame({
        'latitude': lat_grid.flatten(),
        'longitude': lon_grid.flatten() 
    })

def generate_us_grid_filtered(grid_size_degrees=1.0, boundary_shapefile=None):
    """Generate grid coordinates filtered to actual US boundaries"""
    if boundary_shapefile is None:
        boundary_shapefile = '/home/teixeia/HyperSpatialProject/data/LAYERS/cb_2018_us_nation_5m.shp'
    us_boundary = gpd.read_file(boundary_shapefile)
    # If needed, filter for the US polygon (depends on shapefile)
    # us_boundary = us_boundary[us_boundary['NAME'] == 'United States']
    grid_df = generate_us_grid(grid_size_degrees)
    geometry = [Point(xy) for xy in zip(grid_df.longitude, grid_df.latitude)]
    geo_df = gpd.GeoDataFrame(grid_df, geometry=geometry, crs='EPSG:4326')
    return geo_df[geo_df.within(us_boundary.union_all())].drop('geometry', axis=1)

def save_lat_long_to_csv(filename, grid_size_degrees=1.0):
    """
    Save latitude and longitude grid to CSV file
    
    Args:
        filename (str): Output filename
        grid_size_degrees (float): Grid cell size in degrees
    """
    grid_df = generate_us_grid_filtered(grid_size_degrees)
    # Add start date and end date - 20 years ago from June to one year after
    grid_df['start_date'] = pd.Timestamp.now().normalize() - pd.DateOffset(years=1, month=6, day=1)
    grid_df['end_date'] = pd.Timestamp.now().normalize() - pd.DateOffset(years=0, month=6, day=1)
    grid_df.to_csv(filename, index=False)

    print(f"Grid saved to {filename}")

if __name__ == "__main__":
    # Example usage
    save_lat_long_to_csv('/home/teixeia/HyperSpatialProject/data/LAYERS/us_grid_coordinates_now.csv', grid_size_degrees=0.25)
    print("US grid coordinates generated and saved.")