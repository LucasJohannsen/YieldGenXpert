# rf_interpolation.py
from scipy.sparse import csr_matrix
import subprocess
from YieldGenXpert.utils.styles import get_runtime_mode
mode = get_runtime_mode()

if mode == 'gpu':
    import cudf
    from cuml.ensemble import RandomForestRegressor 
    from cuml.metrics.regression import mean_squared_error, r2_score , mean_absolute_error
else:
    from sklearn.ensemble import RandomForestRegressor


class RFInterpolation:
    def __init__(self, mode, tree_param):
        self.mode = mode
        self.tree_param = tree_param
        self.model = RandomForestRegressor(**tree_param)

    def fit(self, gdf):
        X = gdf.drop(columns=['z', 'geometry'])
        y = gdf['z']

        if self.mode == 'cpu':
            X = csr_matrix(X)
            self.model.fit(X, y)
        elif self.mode == 'gpu':
            X = cudf.from_pandas(X)
            y = cudf.from_pandas(y)
            self.model.fit(X, y)

    def predict(self, grid_df):
        X_grid = grid_df.copy()
        if 'geometry' in X_grid.columns:
            X_grid = X_grid.drop(columns=['geometry'])

        if self.mode == 'cpu':
            X_grid = csr_matrix(X_grid)
            grid_df['z'] = self.model.predict(X_grid)
        elif self.mode == 'gpu':
            X_grid = cudf.from_pandas(X_grid)
            grid_df['z'] = self.model.predict(X_grid)

        return grid_df[['x', 'y', 'z']]
