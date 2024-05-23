import pandas as pd
import geopandas as gpd
import numpy as np



def train_test_split(gdf, splits):
    """
    Split the GeoDataFrame into training and testing sets based on the specified splits.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame.
        splits (list): List of data splits in percentage (e.g., [10, 20, 30]).

    Returns:
        train (list): List of training GeoDataFrames.
        test (list): List of testing GeoDataFrames.
        target_train_sp (list): List of target values for training.
        target_test_sp (list): List of target values for testing.
        train_sp (list): List of training GeoDataFrames without target and geometry columns.
        test_sp (list): List of testing GeoDataFrames without target and geometry columns.
    """
    train = []
    test = []
    target_train_sp = []
    target_test_sp = []
    train_sp = []
    test_sp = []
    drop = ['z', 'geometry']  # Columns to drop (target and geometry)
   
    for factor in splits:
        # Perform the train-test split
        train_data = gdf.sample(frac=factor, random_state=1)
        test_data = gdf[~gdf.index.isin(train_data.index)]

        # Append to the lists
        train.append(train_data)
        test.append(test_data)

        # Save the target columns in a one-dimensional array
        target_train_sp.append(pd.DataFrame(train_data['z']))
        target_test_sp.append(pd.DataFrame(test_data['z']))

        # Remove the target and geometry columns from the dataframes
        train_sp.append(train_data.drop(columns=drop))
        test_sp.append(test_data.drop(columns=drop))

    return train, test, target_train_sp, target_test_sp, train_sp, test_sp