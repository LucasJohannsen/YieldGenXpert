import time
start_time = time.time()
######### Import packages #########
import pandas as pd 
import geopandas as gpd
import datetime
import numpy as np
import itertools
import optuna
from sklearn.model_selection import KFold
from shapely.geometry import Point


import cudf
from cuml.ensemble import RandomForestRegressor
from cuml.model_selection import train_test_split
from cuml.metrics.regression import mean_squared_error, r2_score , mean_absolute_error


from utils.data_functions import harvester_data, dgm_data, geo_vista_data
from utils.spatial_functions import spatial_features
from utils.optuna_hyper import spatial_hyperparameter

path = '/mnt/c/Users/Legion Pro/OneDrive - Forschungs- u. Entwicklungszentrum FH-Kiel GmbH/GitLab/ki_anbauplanung/data/Interpolationsfeld/Field.shp'
path_b = '/mnt/c/Users/Legion Pro/OneDrive - Forschungs- u. Entwicklungszentrum FH-Kiel GmbH/GitLab/ki_anbauplanung/data/Interpolationsfeld/Feldgrenze/Parzellen_cut.shp'

dgm_files = [
    '/mnt/c/Users/Legion Pro/OneDrive - Forschungs- u. Entwicklungszentrum FH-Kiel GmbH/GitLab/ki_anbauplanung/data/Cleaned_Data/dgm1_32_567_6021_1_sh.xyz',
    '/mnt/c/Users/Legion Pro/OneDrive - Forschungs- u. Entwicklungszentrum FH-Kiel GmbH/GitLab/ki_anbauplanung/data/Cleaned_Data/dgm1_32_566_6021_1_sh.xyz',
    '/mnt/c/Users/Legion Pro/OneDrive - Forschungs- u. Entwicklungszentrum FH-Kiel GmbH/GitLab/ki_anbauplanung/data/Cleaned_Data/dgm1_32_566_6020_1_sh.xyz'
]

folder = '/mnt/c/Users/Legion Pro/OneDrive - Forschungs- u. Entwicklungszentrum FH-Kiel GmbH/GitLab/ki_anbauplanung/data/Interpolationsfeld/Ergebnisse'



def rf_pipeline(co, den, sp, best_param):
    # Print the parameters
    print('Running experiment with co = {}, den = {}, sp = {}'.format(co, den, sp))
    
    # Load the data
    gdf, poly_gdf, poly = harvester_data(path, path_b)
    
    # If co is true, load the DGM data and merge with gdf
    if co == True:
        dgm = dgm_data(dgm_files=dgm_files, path_b=path_b)
        gdf['x'] = gdf['x'].astype(float)
        gdf['y'] = gdf['y'].astype(float)
        dgm['x'] = dgm['x'].astype(float)
        dgm['y'] = dgm['y'].astype(float)
        merged_df = gpd.sjoin_nearest(gdf, dgm, how='left', distance_col= True)
        merged_df.rename(columns={'x_left': 'x', 'y_left': 'y'}, inplace=True)
        merged_df = merged_df[['x', 'y', 'z', 'h', 'geometry']]
        gdf = merged_df.copy()
        gdf['h'] = gdf['h'].astype(float)
        print('Data imported')

    #create a grid of points with the desired density
    # Define your grid spacing
    grid_spacing = 10

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
    grid_df.to_csv
    #### append the h values from the dgm to the grid
    grid_df['x'] = grid_df.geometry.x
    grid_df['y'] = grid_df.geometry.y
    # grid_df = gpd.sjoin_nearest(grid_df, dgm, how='left', distance_col= True)
    # grid_df.rename(columns={'x_left': 'x', 'y_left': 'y'}, inplace=True)
    # grid_df = grid_df[['x', 'y', 'h', 'geometry']]

    # Create spatial features
    if sp == True:    
        gdf = spatial_features(den, gdf, poly_gdf, poly, range=best_param['ra'])
        grid_df = spatial_features(den, grid_df, poly_gdf, poly, range=best_param['ra'])
        print('Number of columns in gdf: ', len(gdf.columns), ' and Number of columns in grid_df: ', len(grid_df.columns))
        print('Spatial features created')
    
    # Create X and y dataframes
    X = gdf.drop(columns=['z', 'geometry'])
    y = gdf['z']
    X = X.astype('float32')
    y = y.astype('float32')
    grid_df = grid_df.drop(columns=['geometry'])
    grid_df = grid_df.astype('float32')
    X = cudf.from_pandas(X)
    y = cudf.from_pandas(y)
    print('Number of rows in X: ', len(X), ' and Number of rows in grid_df: ', len(grid_df))
    # Train a RandomForestRegressor model on all the gdf data
    tree_param = {'max_depth': 85, 'n_estimators': 508, 'min_samples_split': 2 ,'min_samples_leaf': 1, 'max_features': 'auto', 'bootstrap': True} 
    begin = time.time()
        # Create the model
    rf = RandomForestRegressor(
            n_estimators=tree_param['n_estimators'],
            max_depth=tree_param['max_depth'],
            min_samples_split=tree_param['min_samples_split'],
            min_samples_leaf=tree_param['min_samples_leaf'],
            max_features=tree_param['max_features'],
            bootstrap=tree_param['bootstrap'],
            random_state=0,
            n_streams=1,
            n_bins = 234,# macht erstmal keinen unterschied
            split_criterion='mse', # 2 or 'mse' for mean squared error
        )

    rf.fit(X, y)

    # Use the trained model to predict the z values into the grid 
    X_grid = grid_df.copy()
    if 'geometry' in X_grid.columns:
        X_grid = X_grid.drop(columns=['geometry'])
    # Check that X and X_grid have the same columns, in the same order
    assert X.columns.equals(X_grid.columns), "X and X_grid must have the same columns in the same order"

    X_grid = cudf.from_pandas(X_grid)
    grid_df['z'] = rf.predict(X_grid).to_pandas()
    print('Random Forest finished')
    end = time.time()
    runtime = end - begin
    print('Runtime: ', runtime)
    # Save the dgm DataFrame with the predicted z values to a CSV file
    #just keep x, y, z
    grid_df = grid_df[['x', 'y', 'z']]
    # convert to geo df
    grid_df = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.x, grid_df.y))
    now = datetime.datetime.now()
    grid_df.to_csv(f"{folder}/Grid{grid_spacing}_Predicted_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")

field = 'Field'
splits = 0.3
vista = False

all_study_summaries = optuna.study.get_all_study_summaries(storage="sqlite:///db.sqlite3")

# Check if the study exists
study_exists = any(study.study_name == f"study_{field}" for study in all_study_summaries)

# If the study does not exist, run it
if not study_exists:
    spatial_hyperparameter(path=path, path_b=path_b, dgm_files=dgm_files, aspect_files=None, field=field, co=False, sp=True, split=splits, vista=vista)

# Erstellen Sie ein neues Optuna-Studienobjekt mit der Verbindung zur SQLite-Datenbank
study = optuna.load_study(study_name=f"study_"+field, storage="sqlite:///db.sqlite3")

# Holen Sie sich den besten Versuch
best_trial = study.best_trial

# Sie können dann auf die Eigenschaften des besten Versuchs zugreifen, z.B.
best_trial_value = best_trial.value
best_params = best_trial.params
print(f"Bestes Ergebnis: {best_trial_value}")
print(f"Parameter für bestes Ergebnis: {best_params}")


co = False
den = best_params['den']
sp = True
aspect_files = None
slope_files = None

# Call the function with the desired parameters
rf_pipeline(co, den, sp, best_params)

print("--- %s seconds ---" % (time.time() - start_time))
