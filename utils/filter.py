import pandas as pd 
import geopandas as gpd
import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from shapely.geometry import LineString



# Parameter definition
def load_data(y_path = "data/raw_Data/Field_Clemenskop.shp"):
    # write docstring
    """
    Load the data from a shape file and convert it to a GeoDataFrame.
    
    """


    # Read in the shape file
    gdf = gpd.read_file(y_path)
    gdf = gdf.to_crs("EPSG:32632")

    # drop x and y columns 
    gdf = gdf.drop(columns=["X", "Y"])
    # create x and y columns from geometry
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y

    return gdf

def calculate_zero_flags(gdf):
    # write docstring
    """
    Calculate zero flags for the AreaYield, TimeYield, Speed and Moisture columns.
    """

    # AreaYield Zero flag
    gdf['Zero_AreaYield'] = [1 if gdf.loc[i, 'AreaYield'] == 0 else 0 for i in range(len(gdf))]

    try:
        # Moisture Zero flag
        gdf['Zero_Moisture'] = [1 if gdf.loc[i, 'Moisture'] == 0 else 0 for i in range(len(gdf))]

        # TimeYield Zero flag
        gdf['Zero_TimeYield'] = [1 if gdf.loc[i, 'TimeYield'] == 0 else 0 for i in range(len(gdf))]

        # Speed Zero flag
        gdf['Zero_Speed'] = [1 if gdf.loc[i, 'Speed'] == 0 else 0 for i in range(len(gdf))]

    except KeyError:
        pass

    return gdf



def calculate_path_numbers(gdf):
    # Calculate path numbers based on duration
    gdf['Duration'] = 2

    for u in range(1, len(gdf)):
        gdf.loc[u, 'Duration'] = gdf.loc[u, 'Time'] - gdf.loc[u - 1, 'Time']

    path_number = 1
    for v in range(len(gdf)):
        if gdf.loc[v, 'Duration'] <= 3:
            gdf.loc[v, 'Path'] = path_number
        else:
            path_number += 1
            gdf.loc[v, 'Path'] = path_number

    # Get all the unique path numbers
    unique_paths = gdf.Path.unique()

    # Column to identify path numbers to ignore because they consist of only one point
    gdf['Path_ignore'] = 0

    # Identify corresponding path numbers
    for i in range(len(unique_paths)):
        if (sum(gdf.Path == unique_paths[i]) < 2):
            gdf.loc[gdf['Path'] == unique_paths[i], ['Path_ignore']] = 1

    # Create a DataFrame only containing path numbers with more than one point
    paths_with_multiple_points = gdf.drop(gdf[gdf['Path_ignore'] == 1].index)
    lines = paths_with_multiple_points.groupby('Path')['geometry'].apply(lambda x: LineString(x.tolist()))
    lines = gpd.GeoDataFrame(lines, geometry='geometry')

    # Add start and stop time to the lines
    lines['StartTime'] = paths_with_multiple_points.groupby(['Path'])['Time'].min()
    lines['StopTime'] = paths_with_multiple_points.groupby(['Path'])['Time'].max()

    # Define coordinate system
    lines = lines.set_crs("EPSG:32632")

    single_point_paths = gdf[gdf['Path_ignore'] == 1]  # Subset for single point paths
    single_point_paths_grouped = single_point_paths.groupby('Path')['geometry'].apply(lambda x: MultiPoint(x.tolist()))
    single_point_paths_grouped = gpd.GeoDataFrame(single_point_paths_grouped, geometry='geometry')

    single_point_paths_grouped['StartTime'] = single_point_paths.groupby(['Path'])['Time'].min()  # Add timestamps
    single_point_paths_grouped['StopTime'] = single_point_paths.groupby(['Path'])['Time'].max()  # Add timestamps

    single_point_paths_grouped = single_point_paths_grouped.set_crs("EPSG:32632")  # Set reference system

    lines = pd.concat([lines, single_point_paths_grouped])  # Merge lines and single point paths

    return gdf, lines

#  TODO: write a test for the line df(?)

def calculate_min_distance(gdf, lines):
    # Calculate minimum distances between points and lines
    for i in range(len(gdf)):
        # Filter for lines that are before in time of the current point
        filter_lines = lines[(lines['StartTime'] < gdf.loc[i]['Time']) & (lines['StopTime'] < gdf.loc[i]['Time'])]
        # Calculate minimum distance to the remaining lines
        gdf.loc[i, ['min_distance']] = filter_lines.distance(gdf.loc[i].geometry).min()
        # caclculate the median distance
    ww = gdf['min_distance'].median()
    return gdf , ww


def calculate_overlap_flag(gdf, ww):
    # Create an overlap flag
    gdf['overlap'] = 0
    tolerance = 0.12 * ww # TODO: make tolerance a parameter
    for i in range(len(gdf)):
        if (gdf.min_distance[i] < ww - tolerance):  # Distance < 9m is considered overlap
            gdf.loc[i, ['overlap']] = 1

    return gdf




def calculate_SDF_flag(gdf, m : int = 3): # m is the number of standard deviations 
    i = 0
    gdf['SDF'] = 0

    # Filter the data
    Filter = gdf[(gdf['Zero_AreaYield'] == 0) & 
                (gdf['Zero_Moisture'] == 0) & 
                (gdf['Zero_TimeYield'] == 0) &
                (gdf['Zero_Speed'] == 0) &
                (gdf['overlap'] == 0)]

    # Calculate Max and Min
    Max = Filter['AreaYield'].mean() + m * Filter['AreaYield'].std()
    Min = Filter['AreaYield'].mean() - m * Filter['AreaYield'].std()

    # Loop through the data
    for l in range(len(gdf)):
        if gdf.loc[i, 'AreaYield'] > Max or gdf.loc[i, 'AreaYield'] < Min:
            gdf.loc[i, 'SDF'] = 1
        i += 1

    return gdf




def save_gdf_as_shp(gdf, path):
    
    # base_path_i = os.path.join("C://", "Users", "Legion Pro",
    #                           "OneDrive - Forschungs- u. Entwicklungszentrum FH-Kiel GmbH",
    #                           "GitLab", "ki_anbauplanung", "data", "Filter_ev", "data")
    # Save gdf as shapefile
    gdf.to_file(path)
    print("Done")
# write a main function to call all the functions and test them
def main():
    gdf = load_data()
    gdf = calculate_zero_flags(gdf)
    gdf, lines = calculate_path_numbers(gdf)
    gdf, ww = calculate_min_distance(gdf, lines)
    gdf = calculate_overlap_flag(gdf, ww)
    gdf = calculate_SDF_flag(gdf)
    save_gdf_as_shp(gdf, r"data\Filter_ev")

main()

