import time
import optuna
from optuna_dashboard import run_server
import pandas as pd 
import geopandas as gpd
import numpy as np
from sklearn.metrics import mean_squared_error
import numpy as np
import os

import cudf
from cuml.ensemble import RandomForestRegressor
from cuml.model_selection import train_test_split
from cuml.metrics import mean_squared_error, r2_score



from YieldGenXpert.utils.data_functions import harvester_data, dgm_data, geo_vista_data
from YieldGenXpert.utils.spatial_functions import spatial_features


def spatial_hyperparameter(path, path_b, dgm_files, aspect_files, field, co=True, sp=True, split=0.3, slope_files=None, vista=False):
    # existing code, with the paths replaced by function arguments
    start_time = time.time() 
    if vista: 
        gdf, poly_gdf, poly = geo_vista_data(path, path_b)
    else:
        gdf, poly_gdf, poly = harvester_data(path, path_b)
    if co == True:
        dgm = dgm_data(dgm_files=dgm_files, path_b=path_b, aspect_files=aspect_files, slope_files=slope_files)
        gdf['x'] = gdf['x'].astype(float)
        gdf['y'] = gdf['y'].astype(float)
        dgm['x'] = dgm['x'].astype(float)
        dgm['y'] = dgm['y'].astype(float)
        #merged_df = gdf.merge(dgm[['x', 'y', 'h']], on=['x', 'y'], how='left')
        merged_df = gpd.sjoin_nearest(gdf, dgm, how='left', distance_col= True)
        # rename x_left and y_left to x and y
        merged_df.rename(columns={'x_left': 'x', 'y_left': 'y'}, inplace=True)
        #just keep x, y, z, h and geometry
        if slope_files is not None and len(slope_files) > 0 and aspect_files is not None and len(aspect_files) > 0:
            merged_df = merged_df[['x', 'y', 'z', 'h', 's','a', 'geometry']]
        elif slope_files is not None and len(slope_files) > 0 and (aspect_files is None or len(aspect_files) == 0):
            merged_df = merged_df[['x', 'y', 'z', 'h','s', 'geometry']]
        elif aspect_files is not None and len(aspect_files) > 0 and (slope_files is None or len(slope_files) == 0):
            merged_df = merged_df[['x', 'y', 'z', 'h','a', 'geometry']]
        else:
            merged_df = merged_df[['x', 'y', 'z','h', 'geometry']]
        gdf = merged_df.copy()

        if slope_files is not None:
            gdf['s'] = gdf['s'].astype(float)
        if aspect_files is not None:
            gdf['a'] = gdf['a'].astype(float)
        print('Data imported')


    def create_objective(gdf, poly_gdf, poly):
        if gdf.empty or poly_gdf.empty or not poly:
            raise ValueError("gdf, poly_gdf, and poly must be provided")

        def objective(trial):
            # Hyperparameter, die wir optimieren möchten
            den = trial.suggest_int("den", 10, 50)
            ra = trial.suggest_int("ra", 5, 15)

            # Hyperparameter, die wir festsetzen
            n_estimators = 508
            max_depth = 85
            min_samples_split = 2
            min_samples_leaf = 1
            max_features = 'auto'
            bootstrap = True

            # Erstellen der spatial features
            gdf_new = spatial_features(den, gdf, poly_gdf, poly, range=ra)
            # create a X and y dataframe from gdf_new
            X = gdf_new.drop(columns=['z', 'geometry'])
            y = gdf_new['z']
            # change the data type of the columns to float32
            X = X.astype('float32')
            y = y.astype('float32')
            X = cudf.from_pandas(X)
            y = cudf.from_pandas(y)
            ######### Split the data into training and testing sets #########
            X_train_crs, X_test_crs, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42, shuffle=True)
            # Definieren und trainieren des Modells
            model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                min_samples_split=min_samples_split, 
                min_samples_leaf=min_samples_leaf, 
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=0,
                n_streams=1
            )
            model.fit(X_train_crs, y_train)
            
            # Vorhersagen auf dem Testset und Berechnung des Fehlers
            Y_pred = model.predict(X_test_crs)
            error = mean_squared_error(y_test, Y_pred)
            
            return  error
        return objective

        
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="study_"+field,
        direction="minimize",
    )
    try:
        objective = create_objective(gdf, poly_gdf, poly)
        study.optimize(objective, n_trials=20)
    except ValueError as e:
        print(f"Failed to create objective function: {e}")
    end_time = time.time()  # End the timer
    print(f'The study took {end_time - start_time} seconds to run')
    return study.best_params


def hyperparameters(gdf,poly_gdf, poly ):
    if gdf.empty or poly_gdf.empty or not poly:
        raise ValueError("Yield Data and Field Boundary must be provided")
    def objective(trial):
        # Hyperparameter, die wir optimieren möchten
        den = trial.suggest_int("den", 10, 50)
        ra = trial.suggest_int("ra", 5, 15)
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 10, 100)
        
        # Hyperparameter, die wir festsetzen
        min_samples_split = 2
        min_samples_leaf = 1
        max_features = 'auto'
        bootstrap = True

        # Erstellen der spatial features
        gdf_new = spatial_features(den, gdf, poly_gdf, poly, range=ra)
        # create a X and y dataframe from gdf_new
        X = gdf_new.drop(columns=['z', 'geometry'])
        y = gdf_new['z']
        # change the data type of the columns to float32
        X = X.astype('float32')
        y = y.astype('float32')
        X = cudf.from_pandas(X)
        y = cudf.from_pandas(y)
        ######### Split the data into training and testing sets #########
        X_train_crs, X_test_crs, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
        # Definieren und trainieren des Modells
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=0,
            n_streams=1
        )
        model.fit(X_train_crs, y_train)
        
        # Vorhersagen auf dem Testset und Berechnung des Fehlers
        Y_pred = model.predict(X_test_crs)
        error = mean_squared_error(y_test, Y_pred)
        
        return objective
    study = optuna.create_study()
    try:
        objective = hyperparameters(gdf, poly_gdf, poly)
        study.optimize(objective, n_trials=20)
    except ValueError as e:
        print(f"Failed to create objective function: {e}")

    study.best_params
