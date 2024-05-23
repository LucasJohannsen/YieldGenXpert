import time
import optuna
from optuna_dashboard import run_server
import pandas as pd 
import geopandas as gpd
import numpy as np
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from pykrige.ok import OrdinaryKriging

from utils.data_functions import harvester_data, geo_vista_data

from utils.data_functions import harvester_data, dgm_data, geo_vista_data
from utils.spatial_functions import spatial_features


def OK_hyperparameter(path, path_b, field,  split=0.3, slope_files=None, vista=False):
    start_time = time.time() 
    if vista:
        gdf, poly_gdf, poly = geo_vista_data(path, path_b)
    else:
        gdf, poly_gdf, poly = harvester_data(path, path_b)
    print('Data imported')
    # create a X and y dataframe from gdf
    X = gdf.drop(columns=['z', 'geometry'])
    y = gdf['z']

    def create_objective(X, y):
        def objective(trial):
            # Hyperparameter beim OK, die wir optimieren möchten
            variogram_model = trial.suggest_categorical("variogram_model", ["linear", "power", "gaussian", "spherical", "exponential"])
            weight = trial.suggest_categorical("weight", [True, False])
                
            # Definieren und trainieren des Modells
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42, shuffle=True) 
            model = OrdinaryKriging(X_train.x, X_train.y, y_train.values
                                    , variogram_model=variogram_model
                                    , weight=weight
            )
            Y_pred, ss = model.execute('points', X_test.x, X_test.y)
            
            error = mean_squared_error(y_test, Y_pred)
            
            return error
        
        return objective

        
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="study_OK_"+field,
        direction="minimize",
    )
    try:
        objective = create_objective(X,y)
        study.optimize(objective, n_trials=10)
    except ValueError as e:
        print(f"Failed to create objective function: {e}")
    end_time = time.time()  # End the timer
    print(f'The study took {end_time - start_time} seconds to run')
    return study.best_params
