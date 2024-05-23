import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold, GridSearchCV
import itertools
import datetime
import matplotlib.pyplot as plt

def best_params(param_grid, n_splits, X_train_crs, y_train):
    """
    Perform grid search with cross-validation to find the best parameters for RandomForestRegressor.

    Args:
        param_grid (dict): Dictionary of parameter grid.
        n_splits (int): Number of cross-validation splits.
        X_train_crs (csr_matrix): GeoDataFrame without target and geometry columns for training.
        y_train (array): Array of target values for training.

    Returns:
        best_params (dict): Dictionary containing the best parameters for each data split.
    """
    best_params = {}

    rf = RandomForestRegressor(random_state=0)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
   
    grid_search.fit(X_train_crs, y_train)
    best_params = grid_search.best_params_

    return best_params


# def best_params(param_grid, n_splits, X_train_crs, y_train):
#     """
#     Perform grid search with cross-validation to find the best parameters for RandomForestRegressor.

#     Args:
#         param_grid (dict): Dictionary of parameter grid.
#         n_splits (int): Number of cross-validation splits.
#         X_test_crs (csr_matrix): GeoDataFrame without target and geometry columns for training.
#         target_train_sp (array): Array target values for training.

#     Returns:
#         best_params (dict): dictionary containing the best parameters for each data split.
#     """
#     best_params = []

#     rf = RandomForestRegressor(random_state=0)
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
#     grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

   
#     grid_search.fit(X_train_crs, y_train.values.ravel())
#     best_params.append(grid_search.best_params_)

#     return best_params


def rf_sp(X_test_crs, target_train_sp, X_train_crs, target_test_sp, best_params):
    """
    Train and evaluate a RandomForestRegressor model using the specified parameters.

    Args:
        X_test_crs (csr_matrix): GeoDataFrame without target and geometry columns for training.
        target_train_sp (array): Array of target values for training.
        X_train_crs (csr_matrix): GeoDataFrame without target and geometry columns for testing.
        target_test_sp (array): Array of target values for testing.
        best_params (Dictionary): Dictionary containing the best parameters.

    Returns:
        results (list): List of lists containing the results for each data split.
    """

    rf = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        random_state=0
    )
    rf.fit(X_test_crs, target_test_sp.ravel())

    # Predict the test data
    predictions = rf.predict(X_train_crs)

    # Calculate MSE and R2
    mse = mean_squared_error(target_train_sp, predictions)
    r2 = r2_score(target_train_sp, predictions)

    # Save the results in a list
    result = [
        best_params['n_estimators'],
        best_params['max_depth'],
        mse,
        r2
    ]


    return result


# def rf_sp(X_test_crs, target_train_sp, X_train_crs, target_test_sp, best_params):
#     """
#     Train and evaluate a RandomForestRegressor model using the specified parameters.

#     Args:
#         X_test_crs (csr_matrix): GeoDataFrame without target and geometry columns for training.
#         target_train_sp (array): Array of target values for training.
#         X_train_crs (csr_matrix): GeoDataFrame without target and geometry columns for testing.
#         target_test_sp (array): Array of target values for testing.
#         best_params (Dictionary): Dictionary containing the best parameters.

#     Returns:
        
#     """
#     results = []

#     for i in range(len(X_train_crs)):
#         rf = RandomForestRegressor(
#             n_estimators=best_params[i]['n_estimators'],
#             max_depth=best_params[i]['max_depth'],
#             random_state=0
#         )
#         rf.fit(X_test_crs[i], target_train_sp[i].values.ravel())

#         # Predict the test data
#         predictions = rf.predict(X_train_crs[i])

#         # Calculate MSE and R2
#         mse = mean_squared_error(target_test_sp[i], predictions)
#         r2 = r2_score(target_test_sp[i], predictions)

#         # Save the results in a list
#         result = [
#             best_params[i]['n_estimators'],
#             best_params[i]['max_depth'],
#             mse,
#             r2
#         ]
#         results.append(result)
        

#         print('Progress: ' + str(i+1) + ' of ' + str(len(X_train_crs)) + ' completed')

#     return results

def pred_results(predictions, target_test_sp,results, best_params,i):
        # Calculate MSE and R2
        mse = mean_squared_error(target_test_sp[i], predictions)
        r2 = r2_score(target_test_sp[i], predictions)

        # Save the results in a list
        result = [
            best_params[i]['n_estimators'],
            best_params[i]['max_depth'],
            mse,
            r2
        ]
        results.append(result)

def viz_results(predictions, target_test_sp, i, test):
    # visualize the results
    plt.scatter(test[i].x,test[i].y, c=(target_test_sp[i] - predictions), cmap='bwr')
    plt.colorbar()
    plt.title('Residuals')
    plt.show()
    plt.scatter(target_test_sp[i], predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()
    plt.scatter(target_test_sp[i], target_test_sp[i] - predictions)
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [0, 0])
    plt.show()
    plt.hist(target_test_sp[i] - predictions, bins = 25)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()

def writing_results(results, co, sp, den,folder):
    """
    Write the results to a CSV file.

    Args:
        results (list): List of lists containing the results.
        co (bool): Flag indicating whether it is co_data or not.
        sp (bool): Flag indicating whether it is sp_data or not.


    Returns:
        None
    """
    now = datetime.datetime.now()

    results_df = pd.DataFrame(results, columns=['split', 'n_estimators', 'max_depth', 'MSE', 'R2'])

    if co and sp:
        results_df.to_csv(folder + '\\RF_sp_co_Results_'+ now.strftime("%Y-%m-%d_%H-%M-%S_") + str(den) +'.csv')
    elif not co and sp:
        results_df.to_csv(folder + '\\RF_sp_Results_'+ now.strftime("%Y-%m-%d_%H-%M-%S_") + str(den) + '.csv')
    elif co and not sp:
        results_df.to_csv(folder + '\\RF_co_Results_'+ now.strftime("%Y-%m-%d_%H-%M-%S_") +  '.csv')
    else:
        results_df.to_csv(folder + '\\RF_Results_'+ now.strftime("%Y-%m-%d_%H-%M-%S_")  + '.csv')



def rf_sp_v(X_test_crs, target_train_sp, X_train_crs, target_test_sp, best_params,test):
    """
    Train and evaluate a RandomForestRegressor model using the specified parameters.

    Args:
        X_test_crs (list): List of GeoDataFrames without target and geometry columns for training.
        target_train_sp (list): List of target values for training.
        X_train_crs (list): List of GeoDataFrames without target and geometry columns for testing.
        target_test_sp (list): List of target values for testing.
        best_params (list): List of dictionaries containing the best parameters for each data split.

    Returns:
        results (list): List of lists containing the results for each data split.
        visualizations (list): List of visualizations for each data split.
    """
    results = []
    visualizations = []

    for i in range(len(X_train_crs)):
        rf = RandomForestRegressor(
            n_estimators=best_params[i]['n_estimators'],
            max_depth=best_params[i]['max_depth'],
            random_state=0
        )
        rf.fit(X_test_crs[i], target_train_sp[i].values.ravel())

        # Predict the test data
        predictions = rf.predict(X_train_crs[i])

        # Calculate MSE and R2
        mse = mean_squared_error(target_test_sp[i], predictions)
        r2 = r2_score(target_test_sp[i], predictions)

        # Save the results in a list
        result = [
            best_params[i]['n_estimators'],
            best_params[i]['max_depth'],
            mse,
            r2
        ]
        results.append(result)

        # Generate visualizations
        visualization = viz_results(predictions, target_test_sp[i], i, test[i])
        visualizations.append(visualization)

        print('Progress: ' + str(i+1) + ' of ' + str(len(X_train_crs)) + ' completed')

    return results, visualizations

def writing_viz(visualizations, co, sp, den, folder):
    """
    Write the visualizations to png files.

    Args:
        visualizations (list): List of visualizations.
        co (bool): Flag indicating whether it is co_data or not.
        sp (bool): Flag indicating whether it is sp_data or not.
        den (int): Density value.
        folder (str): Path to the folder to save the visualizations.

    Returns:
        None
    """
    now = datetime.datetime.now()

    for i, viz in enumerate(visualizations):
        if co and sp:
            viz.savefig(folder + f'\\RF_sp_co_Results_{now.strftime("%Y-%m-%d_%H-%M-%S_")}{den}_{i}.png')
        elif not co and sp:
            viz.savefig(folder + f'\\RF_sp_Results_{now.strftime("%Y-%m-%d_%H-%M-%S_")}{den}_{i}.png')
        elif co and not sp:
            viz.savefig(folder + f'\\RF_co_Results_{now.strftime("%Y-%m-%d_%H-%M-%S_")}{i}.png')
        else:
            viz.savefig(folder + f'\\RF_Results_{now.strftime("%Y-%m-%d_%H-%M-%S_")}{i}.png')
