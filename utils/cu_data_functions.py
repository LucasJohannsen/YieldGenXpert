import cudf
import cuspatial
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def geo_vista_data(path, path_b):
    """
    Import data from the GeoVista file and filter the points within the field boundary polygon.

    Args:
        path (str): Path to the GeoVista file.
        path_b (str): Path to the field boundary polygon file.

    Returns:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing the filtered points within the polygon.
        poly_gdf (gpd.GeoDataFrame): GeoDataFrame representing the field boundary polygon.
    """
    # Create the dataframe from the GeoVista file
    with open(path) as f:
        data = f.readlines()
        data = [line.split() for line in data]
        df = pd.DataFrame(data)
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        df = df.apply(pd.to_numeric)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.RW, df.HW))
        gdf = gdf.rename(columns={"RW": "x", "HW": "y", "WERT": "z"})
        gdf.crs = 'epsg:32632'

    # Create the polygon for the field boundary
    poly_gdf = gpd.read_file(path_b)
    poly_gdf = poly_gdf.to_crs('epsg:32632')
    poly = poly_gdf.geometry.iloc[0]

    # Filter the points that are within the polygon from gdf
    gdf = gdf[gdf.within(poly)]

    return gdf, poly_gdf, poly

def harvester_data(path, path_b):
    """
    Read in a shapefile and perform filtering based on a polygon shapefile.

    Args:
        path (str): Path to the shapefile.
        path_b (str): Path to the polygon shapefile.

    Returns:
        gdf (gpd.GeoDataFrame): Filtered GeoDataFrame.
        poly_gdf (gpd.GeoDataFrame): Polygon GeoDataFrame.
        poly (shapely.geometry.Polygon): Polygon geometry.
    """
    # Import the shapefile as a GeoDataFrame
    df = gpd.read_file(path)

    # Read in the polygon shapefile as a GeoDataFrame
    poly_gdf = gpd.read_file(path_b)

    # Change the CRS to EPSG:32632
    df = df.to_crs('EPSG:32632')
    poly_gdf = poly_gdf.to_crs('EPSG:32632')

    # Extract the polygon geometry
    poly = poly_gdf.geometry.iloc[0]

    # Filter the points that are within the polygon from the GeoDataFrame
    df = df[df.within(poly)]
    # clean data
    df = df[df.Zero_AreaY != 1]
    df = df[df.Zero_TimeY != 1]
    df = df[df.Zero_Speed != 1]
    df = df[df.Zero_Moist != 1]
    df = df[df.overlap != 1]
    df = df[df.SDF != 1]
    df.rename(columns={'AreaYield': 'z'}, inplace=True)
    df = df.drop(columns=['X', 'Y'])
    df['x'] = df.geometry.x
    df['y'] = df.geometry.y
    gdf = df[['x', 'y', 'z', 'geometry']]
    return gdf, poly_gdf, poly


def cu_dgm_data(dgm_files, path_b, slope_files=None, aspect_files=None):
    """
    Import and preprocess data from multiple DGM files.

    Args:
        dgm_files (list): List of file paths for DGM files.
        slope_files (list, optional): List of file paths for slope files.
        aspect_files (list, optional): List of file paths for aspect files.
        path_b (str): Path to the polygon shapefile.

    Returns:
        dgm (gpd.GeoDataFrame): GeoDataFrame containing the DGM data and slope.
    """
    # Read and concatenate DGM files
    dgm = cudf.concat([cudf.read_csv(file, header=None, delimiter='\t') for file in dgm_files], ignore_index=True)
    # Split the columns and rename them
    dgm[['x', 'y', 'h']] = dgm[0].str.split(" ", expand=True)

    # Print the column names
    print(dgm.columns)
    dgm = dgm.drop(columns=[0])
    
    # Convert to float32
    dgm = dgm.astype({'x': 'float32', 'y': 'float32', 'h': 'float32'})
    
    # Make geometry using cuspatial
    dgm['geometry'] = cuspatial.points(dgm['x'], dgm['y'])

    # ... (same for slope and aspect files if provided)
    if slope_files is not None:
        # Read and concatenate DGM slope files
        dgm_slope = cudf.concat([cudf.read_csv(file, header=None, delimiter='\t') for file in slope_files], ignore_index=True)
        dgm_slope[['x', 'y', 's']] = dgm_slope[0].str.split(" ", expand=True)
        dgm_slope = dgm_slope.drop(columns=[0])
        # Convert to float32
        dgm_slope = dgm_slope.astype({'x': 'float32', 'y': 'float32', 's': 'float32'})
        # Make geometry using cuspatial
        dgm_slope['geometry'] = cuspatial.points(dgm_slope['x'], dgm_slope['y'])
        # merge the dgm and dgm_slope 
        dgm = cudf.merge(dgm, dgm_slope)
        dgm.rename(columns={'x_left': 'x', 'y_left': 'y'}, inplace=True)
        dgm = dgm[['x', 'y', 'h', 's', 'geometry']]
    
    if aspect_files is not None and slope_files is not None:
        # Read and concatenate DGM aspect files
        dgm_aspect = cudf.concat([cudf.read_csv(file, header=None, delimiter='\t') for file in aspect_files], ignore_index=True)
        dgm_aspect[['x', 'y', 'a']] = dgm_aspect[0].str.split(" ", expand=True)
        dgm_aspect = dgm_aspect.drop(columns=[0])
        # Convert to float32
        dgm_aspect = dgm_aspect.astype({'x': 'float32', 'y': 'float32', 'a': 'float32'})
        # Make geometry using cuspatial
        dgm_aspect['geometry'] = cuspatial.points(dgm_aspect['x'], dgm_aspect['y'])
        # merge the dgm and dgm_aspect
        dgm = cudf.merge(dgm, dgm_aspect)
        dgm.rename(columns={'x_left': 'x', 'y_left': 'y'}, inplace=True)
        dgm = dgm[['x', 'y', 'h', 's', 'a', 'geometry']]
    
    if aspect_files is not None and slope_files is None:
        # Read and concatenate DGM aspect files
        dgm_aspect = cudf.concat([cudf.read_csv(file, header=None, delimiter='\t') for file in aspect_files], ignore_index=True)
        dgm_aspect[['x', 'y', 'a']] = dgm_aspect[0].str.split(" ", expand=True)
        dgm_aspect = dgm_aspect.drop(columns=[0])
        # Convert to float32
        dgm_aspect = dgm_aspect.astype({'x': 'float32', 'y': 'float32', 'a': 'float32'})
        # Make geometry using cuspatial
        dgm_aspect['geometry'] = cuspatial.points(dgm_aspect['x'], dgm_aspect['y'])
        # merge the dgm and dgm_aspect
        dgm = cudf.merge(dgm, dgm_aspect)
        dgm.rename(columns={'x_left': 'x', 'y_left': 'y'}, inplace=True)
        dgm = dgm[['x', 'y', 'h', 'a', 'geometry']]

    poly_gdf = gpd.read_file(path_b)
    # Change the CRS to EPSG:32632
    poly_gdf = poly_gdf.to_crs('EPSG:32632')
    dgm = dgm.to_pandas()  # Convert cudf DataFrame back to pandas DataFrame
    dgm['geometry'] = dgm.apply(lambda row: Point(row['x'], row['y']), axis=1)
    dgm = gpd.GeoDataFrame(dgm, geometry='geometry')
    dgm.crs =poly_gdf.crs
    # Extract the polygon geometry
    poly = poly_gdf.geometry.iloc[0]
    # Filter the points that are within the polygon from the GeoDataFrame
    dgm = dgm[dgm.geometry.within(poly)]

    return dgm