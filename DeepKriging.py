#imports 
import time
start_time = time.time()
import pandas as pd 
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.spatial.distance import cdist
import random
random.seed(1)
import torch
torch.manual_seed(1)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math
import torch.nn as nn
import datetime
import os
from utils.data_functions import harvester_data, dgm_data, geo_vista_data
from utils.spatial_functions import spatial_features
from sklearn.model_selection import KFold

##################################### Create the DNN ###############################################################

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size ,hidden_dim=100):
        super(MultiLayerPerceptron, self).__init__()
        
        self.input_size = input_size
        
        self.lin1 = nn.Linear(input_size, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)
        self.lin4 = nn.Linear(hidden_dim,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)

 
    def forward(self, x):
        out1 = nn.ReLU()(self.lin1(x))
        out1 = self.dropout(out1)
        out1 = self.batchnorm1(out1)
        out2 = nn.ReLU()(self.lin2(out1))
        out2 = self.dropout(out2)
        out3 = nn.ReLU()(self.lin3(out2))
        out3 = self.batchnorm2(out3)
        out4 = self.lin4(out3)
        
        return out4
    
def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X)
    
criterion = torch.nn.MSELoss()
learning_rate=0.0008985921108824352
epochs=348
torch.manual_seed(1)

######################################################################################################################
vista = True

if vista:
    fields = [ 'data/Datensammlung Küchenkoppel',
                'data/Datensammlung Hasenberg',
            #    'data/Datensammlung Achterkoppel',
            ]
else:
    fields = [ #'data/Cleaned_Data/1. Clemenskopel',
               'data/Cleaned_Data/2. Hafbrede',
               'data/Cleaned_Data/3. Halbschlag',
               'data/Cleaned_Data/4. Hinterer_B',
               'data/Cleaned_Data/5. Rothenhof',
               'data/Cleaned_Data/6. Wildkoppel',
            ]




splits = 0.3
kf = KFold(n_splits=5, shuffle=True, random_state=42)
co = True

def dk_pipeline(co, splits):
    result = []
    if vista:
        gdf, poly_gdf, poly = geo_vista_data(path, path_b)
    else:
        gdf, poly_gdf, poly = harvester_data(path, path_b)
    if co == True:
        dgm = dgm_data(dgm_files=dgm_files, path_b=path_b, aspect_files=aspect_files, slope_files=None)
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

    # Create separate scaler objects for each column
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    z_scaler = MinMaxScaler(feature_range=(0, 1))
    h_scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit and transform the 'x' and 'y' columns
    gdf[['x', 'y']] = x_scaler.fit_transform(gdf[['x', 'y']])
    gdf['z'] = z_scaler.fit_transform(gdf[['z']])
    gdf['h'] = h_scaler.fit_transform(gdf[['h']])

    s = gdf[['x','y']].values
    N = s.shape[0]

    num_basis = [10**2,19**2,37**2,73**2]
    knots_1dx = [np.linspace(0,1,math.floor(np.sqrt(i))) for i in num_basis]  
    knots_1dy = [np.linspace(0,1,math.floor(np.sqrt(i))) for i in num_basis] 

    # ##Wendland kernel
    basis_size = 0
    phi = np.zeros((N, sum(num_basis)))
    for res in range(len(num_basis)):
        theta = 1/np.sqrt(num_basis[res])*2.5
        knots_x, knots_y = np.meshgrid(knots_1dx[res],knots_1dy[res])
        knots = np.column_stack((knots_x.flatten(),knots_y.flatten()))
        for i in range(num_basis[res]):
            d = np.linalg.norm(s-knots[i,:],axis=1)/theta
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    phi[j,i + basis_size] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
                else:
                    phi[j,i + basis_size] = 0
        basis_size = basis_size + num_basis[res]

    # merge gdf and phi
    phi = pd.DataFrame(phi)
    # join gdf and phi on the index
    gdf.index = pd.RangeIndex(start=0, stop=len(gdf))
    gdf = gdf.join(phi)
    gdf.columns = gdf.columns.astype(str)
    print('DeepKriging feautures created')
    ######### Create the train test splits to compare Methods regarding data density  ########
    X = gdf.drop(columns=['z', 'geometry'])
    y = gdf['z']
    # change the data type of the columns to float32
    X = X.astype('float32')
    y = y.astype('float32')

        ######### 5 KFold CV #########
    results_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # create tensors from the dataframes
        X_train_tensor = torch.tensor(X_train.values)
        y_train_tensor = torch.tensor(y_train.values)
        X_test_tensor = torch.tensor(X_test.values)
        y_test_tensor = torch.tensor(y_test.values)
        y_train_tensor = y_train_tensor.unsqueeze(1)
        y_test_tensor = y_test_tensor.unsqueeze(1)
        print('Data structure created')

        model=MultiLayerPerceptron(X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        model.train()
        
        for epoch in range(epochs):
            y_train_pred = model(X_train_tensor)
        
            optimizer.zero_grad()
            
            train_loss = criterion(y_train_pred, y_train_tensor)
            
            train_loss.backward()
            
            optimizer.step()
            
        #   print("Epoch: %d, loss: %1.5f" % (epoch, train_loss.item()))

        model.eval()
        y_train_pred=predict(model, X_train_tensor)
        print(y_train_pred.shape)
        y_test_pred=predict(model, X_test_tensor)

        # calculate the metrics unscaled 
        # calculate the metrics unscaled 
        y_train_pred = y_train_pred.detach().numpy().reshape(-1, 1)
        y_test_pred = y_test_pred.detach().numpy().reshape(-1, 1)
        y_train_tensor = y_train_tensor.detach().numpy().reshape(-1, 1)
        y_test_tensor = y_test_tensor.detach().numpy().reshape(-1, 1)

        # transform the tensors back to the original scale
        y_train_orig = z_scaler.inverse_transform(y_train_tensor)
        y_test_orig = z_scaler.inverse_transform(y_test_tensor)
        y_train_pred_orig = z_scaler.inverse_transform(y_train_pred)
        y_test_pred_orig = z_scaler.inverse_transform(y_test_pred)
        # calculate the metrics
        mse_train = mean_squared_error(y_train_orig, y_train_pred_orig)
        mse = mean_squared_error(y_test_orig, y_test_pred_orig)
        r2_train = r2_score(y_train_orig, y_train_pred_orig)
        r2 = r2_score(y_test_orig, y_test_pred_orig)
        mae_train = mean_absolute_error(y_train_orig, y_train_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_test_pred_orig)
        rmse_train = np.sqrt(mse_train)
        rmse = np.sqrt(mse)
        results = pd.DataFrame({'field': field ,'mse': mse,'rmse': rmse ,'r2': r2,'mae': mae}, index=[splits])
        results_list.append(results)
        print('DeepKriging finished')

    final_results = pd.concat(results_list) 
    result = final_results.mean()
    #result.to_csv(folder + '/DK_Results.csv')
    result.to_csv(folder2) 

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
    folder2 = 'data/Models2/DeepKriging/' + field + '.csv' 
    slope_files = None
    count = 0
    if dgm_files is not None:
        count += 1
    if aspect_files is not None:
        count += 1
    
    dk_pipeline(co, splits)
    print('Field finished')

# save the execution time
print("--- %s seconds ---" % (time.time() - start_time))


