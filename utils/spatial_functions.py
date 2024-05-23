import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial.distance import cdist
from shapely.geometry import Point
from shapely.geometry import Polygon


def spatial_features(den, gdf, poly_gdf, poly,range=5):
    """
    Calculate spatial features based on density, GeoDataFrame, and polygon.

    Args:
        den (int): Density of evenly spaced points within the polygon.
        gdf (gpd.GeoDataFrame): GeoDataFrame containing point data.
        poly_gdf (gpd.GeoDataFrame): GeoDataFrame containing polygon data.
        poly (shapely.geometry.Polygon): Polygon geometry.

    Returns:
        gdf (pd.DataFrame): Updated DataFrame with spatial features.
    """
    # Create evenly spaced points within the polygon
    x = np.linspace(poly.bounds[0], poly.bounds[2], den)
    y = np.linspace(poly.bounds[1], poly.bounds[3], den)
    xx, yy = np.meshgrid(x, y)
    ggdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.flatten(), yy.flatten()))
    ggdf.crs = poly_gdf.crs
    ggdf = ggdf[ggdf.within(poly)]

    # Add the x and y coordinates as columns
    ggdf['x'] = ggdf.geometry.x
    ggdf['y'] = ggdf.geometry.y

    # Calculate the distance from each point to each point in the gdf
    dist = cdist(gdf[['x', 'y']], ggdf[['x', 'y']], metric='euclidean')

    # Get the range percentile of the distances
    pct_ra = np.percentile(dist, range)
    dist[dist > pct_ra] = np.max(dist.flatten())

    # Convert gdf to DataFrame and join distance DataFrame
    gdf = pd.DataFrame(gdf)
    dist_df = pd.DataFrame(dist, index=gdf.index)
    gdf = gdf.join(dist_df, on=dist_df.index)

    return gdf

def grid(raster_size, poly, gdf):
    """
    Create a grid of points with the desired density.

    Args:
        raster_size (int): Desired density of grid points.
        poly (Polygon): field boundary. 

    Returns:
        grid_df (gpd.GeoDataFrame): GeoDataFrame containing grid points.
    """
    # Define your grid spacing
    grid_spacing = raster_size

    # Get the bounds of the polygon
    minx, miny, maxx, maxy = poly.bounds

    # Create the grid
    x_coords = np.arange(minx, maxx, grid_spacing)
    y_coords = np.arange(miny, maxy, grid_spacing)
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)
            if point.within(poly):
                grid_points.append(point)

    # Convert to a GeoDataFrame
    grid_df = gpd.GeoDataFrame(geometry=grid_points)
    #initialize a crs for the grid
    grid_df.crs = gdf.crs
    # print the number of points in the grid
    print('Number of points in grid: ', len(grid_df))
    return grid_df
