import time
import pandas as pd 
import os
import geopandas as gpd
import datetime
import numpy as np
# check the environment and set an variable to use the right mode
from YieldGenXpert.utils.styles import get_runtime_mode

mode = get_runtime_mode()

if mode == 'gpu':
    import cudf
    from cuml.ensemble import RandomForestRegressor 
    from cuml.metrics.regression import mean_squared_error, r2_score , mean_absolute_error
else:
    from scipy.sparse import csr_matrix
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor



def rf(mode, best_param, X_train, X_test, y_train, y_test):
    """
    Perform a Random Forest regression on the data.

    Args:
        mode (str): Mode to run the Random Forest in ('cpu' or 'gpu').
        best_param (dict): Dictionary containing the best hyperparameters.
        X_train (pd.DataFrame): Training data.
        X_test (pd.DataFrame): Testing data.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.

    Returns:
        mse (float): Mean squared error of the model.
        r2 (float): R-squared value of the model.
    """
    # Check if the mode is 'gpu'
    if mode == 'gpu':
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        X_train_csr, X_test_csr, y_train, y_test = cudf.from_pandas(X_train), cudf.from_pandas(X_test),cudf.from_pandas(y_train), cudf.from_pandas(y_test)
        # Create the Random Forest model
        model = RandomForestRegressor(**best_param)
        # Fit the model
        model.fit(X_train_csr, y_train)
        # Predict the values
        y_pred = model.predict(X_test_csr).to_pandas()
        # Calculate the mean squared error
        mse = mean_squared_error(y_test.to_pandas(), y_pred)
        # Calculate the R-squared value
        r2 = r2_score(y_test, y_pred)
    else:
        # Create crs matrix from X_test and X_train
        X_train_csr = csr_matrix(X_train)
        X_test_csr = csr_matrix(X_test)
        # Create the Random Forest model
        model = RandomForestRegressor(**best_param)
        # Fit the model
        model.fit(X_train_csr, y_train)
        # Predict the values
        y_pred = model.predict(X_test_csr)
        # Calculate the mean squared error
        mse = mean_squared_error(y_test, y_pred)
        # Calculate the R-squared value
        r2 = r2_score(y_test, y_pred)

    return mse, r2

