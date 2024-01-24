import time
start_time = time.time()
######### Import packages #########
import pandas as pd 
import geopandas as gpd
import datetime
import numpy as np
import itertools
import optuna

import cudf
from cuml.ensemble import RandomForestRegressor
from cuml.model_selection import train_test_split
from cuml.metrics.regression import mean_squared_error, r2_score , mean_absolute_error


from utils.data_functions import harvester_data, dgm_data, geo_vista_data
from utils.spatial_functions import spatial_features
from utils.optuna_hyper import spatial_hyperparameter

vista = False

# wenn geovista daten dann:
if vista:
    fields = [ 'data/Datensammlung Küchenkoppel',
                'data/Datensammlung Hasenberg',
                'data/Datensammlung Achterkoppel',
            ]
else:
    fields = [ 'data/Cleaned_Data/1. Clemenskopel',
            #    'data/Cleaned_Data/2. Hafbrede',
            #    'data/Cleaned_Data/3. Halbschlag',
            #    'data/Cleaned_Data/4. Hinterer_B',
            #    'data/Cleaned_Data/5. Rothenhof',
            #    'data/Cleaned_Data/6. Wildkoppel',
            ]


splits = np.arange(0.1, 0.95, 0.05)
splits = np.round(splits, 2)  # round to 2 decimal places
splits = splits[::-1]

def rf_pipeline(co, den, sp, splits, best_param):
    result = []
    for i in range(len(splits)):
        print('Split: ', splits[i])
        ######### Import data #########
        if vista:
            gdf, poly_gdf, poly = geo_vista_data(path, path_b)
        else:
            gdf, poly_gdf, poly = harvester_data(path, path_b)
        ######### Create Co_Variables #########
        if co == True:
            dgm = dgm_data(dgm_files=dgm_files, path_b=path_b, aspect_files=None, slope_files=None)
            gdf['x'] = gdf['x'].astype(float)
            gdf['y'] = gdf['y'].astype(float)
            dgm['x'] = dgm['x'].astype(float)
            dgm['y'] = dgm['y'].astype(float)
            merged_df = gpd.sjoin_nearest(gdf, dgm, how='left', distance_col= True)
            merged_df.rename(columns={'x_left': 'x', 'y_left': 'y'}, inplace=True)
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
        ######### Create spatial feautures  #########
        if sp == True:    
            gdf = spatial_features(den, gdf, poly_gdf, poly, range=best_param['ra'])
            print('Spatial features created')
        ######### Prepare the data #########
        X = gdf.drop(columns=['z', 'geometry'])
        y = gdf['z']
        X = X.astype('float32')
        y = y.astype('float32')
        X = cudf.from_pandas(X)
        y = cudf.from_pandas(y)
        ######### Split the data into training and testing sets #########
        X_train_crs, X_test_crs, y_train, y_test = train_test_split(X, y, test_size=splits[i], random_state=42, shuffle=True) 
        print('Data prepared')
        ######### Random Forest #########
        tree_param = {'max_depth': 85, 'n_estimators': 508, 'min_samples_split': 2 ,'min_samples_leaf': 1, 'max_features': 'auto', 'bootstrap': True} # co1
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
            criterion='mse',
        )
        # Fit the model
        rf.fit(X_train_crs, y_train)
        # Predict the test data
        predictions = rf.predict(X_test_crs)
        # Calculate Metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        print('Random Forest finished')
        # save the curent split value with the mse and r2 values
        results = pd.DataFrame({'mse': mse,'rmse': rmse ,'r2': r2,'mae': mae}, index=[splits[i]])
        result.append(results)
    # save the results as a csv file
    den = best_param['den']
    result = pd.concat(result)
    if co and sp:
        result.to_csv(folder2 + '/RF_spco/'+ field + '.csv')
    elif not co and sp:
        result.to_csv(folder2 + '/RF_sp/'+ field + '.csv')
    # elif co and not sp:
    #     result.to_csv(folder + '/RF_co'+str(count)+'_Results.csv')
    else:
        result.to_csv(folder2 + '/RF/'+ field + '.csv')
    print('Results saved')
def parameter_tuning():
    # Define possible values for parameters
    co_values = [True, False]  
    den_value = best_params['den']  
    sp_values = [True, False]  

    # Iterate over all combinations of co_values and sp_values
    for co, sp in itertools.product(co_values, sp_values):
        start_time = time.time()  
        # When sp is True, use the den_value
        if not (sp is False and co is True):
            den = den_value
            print(f'Running experiment with co = {co}, sp = {sp}')
            rf_pipeline(co, den, sp, splits, best_params)
        else:
            # When sp is False, use a fixed den value
            den = 60
            print(f'Running experiment with co = {co},  sp = {sp}')
            rf_pipeline(co, den, sp, splits, best_params)
        end_time = time.time()  # End the timer
        print(f'The experiment took {end_time - start_time} seconds to run')
    # Call the parameter tuning function


for i in range(len(fields)):
    field = fields[i].split(' ')[1]
    print(field)
    if vista:
        path = fields[i] + '/Ertrag_Vista/Field.DAT'
    else:
        path = fields[i] + '/Field.shp'
    path_b = fields[i] + '/Feldgrenze/Parzellen_cut.shp'
    dgm_files = fields[i] + '/DGM/merged.xyz'
    #aspect_files = fields[i] + '/DGM/merged_aspect.xyz'
    aspect_files = None
    folder = fields[i] + '/Ergebnisse'
    folder2 = 'data/Models2/'
    slope_files = None
    count = 0
    if dgm_files is not None:
        count += 1
    if aspect_files is not None:
        count += 1
    # check the hyperparameter
    # Get all study summaries
    all_study_summaries = optuna.study.get_all_study_summaries(storage="sqlite:///db.sqlite3")

    # Check if the study exists
    study_exists = any(study.study_name == f"study_{field}" for study in all_study_summaries)

    # If the study does not exist, run it
    if not study_exists:
        spatial_hyperparameter(path=path, path_b=path_b, dgm_files=dgm_files, aspect_files=None, field=field, co=True, sp=True, split=0.5, vista=vista)

    # Erstellen Sie ein neues Optuna-Studienobjekt mit der Verbindung zur SQLite-Datenbank
    study = optuna.load_study(study_name=f"study_"+field, storage="sqlite:///db.sqlite3")

    # Holen Sie sich den besten Versuch
    best_trial = study.best_trial

    # Sie können dann auf die Eigenschaften des besten Versuchs zugreifen, z.B.
    best_trial_value = best_trial.value
    best_params = best_trial.params
    print(f"Bestes Ergebnis: {best_trial_value}")
    print(f"Parameter für bestes Ergebnis: {best_params}")

    parameter_tuning()

print("--- %s seconds ---" % (time.time() - start_time))

