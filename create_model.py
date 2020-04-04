import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from utils import *

feature_data = pd.read_csv("saved_data/features.csv")
feature_data.columns = [i for i in range(len(feature_data.columns))]

p_column= len(feature_data.columns)-1
spo2_column = p_column - 1
r_column = spo2_column -1
ref_r_column = r_column - 1


 
def get_line(p, f_D, show = True):
    
    from copy import copy
    f_d = copy(f_D) 
    X,y  =  np.asarray(f_d[(f_d[p_column] == p ) & (f_d[spo2_column] > 60) & (f_d[spo2_column] <95)][ref_r_column]).reshape((-1,1)),np.asarray(f_d[(f_d[p_column] == p ) & (f_d[spo2_column] > 60) & (f_d[spo2_column] <95)][spo2_column]).reshape((-1,1))
    model = LinearRegression()
    
    model.fit(np.concatenate([X**(i+1) for i in range(1)], axis = 1),y )
    spo2_pred = model.predict(np.concatenate([X**(i+1) for i in range(1)], axis = 1))
    
      
    args = np.argsort(X.ravel())
    
    if show:
        fig = plt.figure(figsize=[10,4])
    
        fig.suptitle("Patient {}".format(p))
    
        ax = plt.subplot("121")
        ax.set_title("Before Removing Outliers")
        ax.plot(X.ravel()[args],spo2_pred[args], c= "b")
        ax.scatter(X.ravel(),y, c= "b" )
    
    
    m = float(model.coef_[0])
    c = float(model.intercept_)
    
    d = np.abs(m*X - y +c )/np.sqrt(1+m**2)
    mean = np.mean(d)
    std = np.std(d)

    

    X_ = X[np.where(np.logical_and(d < mean+std, d>mean-std))].reshape((-1,1))
    y_ = y[np.where(np.logical_and(d < mean+std, d>mean-std))]

    model.fit(np.concatenate([X_**(i+1) for i in range(1)], axis = 1),y_)
    spo2_pred = model.predict(np.concatenate([X_**(i+1) for i in range(1)], axis = 1))
    
    args = np.argsort(X_.ravel())
    if show:
        ax = plt.subplot("122")
        ax.set_title("After Removing Outliers")
    
        ax.plot(X_.ravel()[args],spo2_pred[args], c= "b")
        ax.scatter(X_.ravel(),y_, c= "b" )
        
        plt.show()
            
    m = float(model.coef_[0])
    c = float(model.intercept_)
    
    return np.asarray([p,m,c])

def create():
    model_df = pd.DataFrame()
    for p in feature_data[p_column].unique():
        row = pd.DataFrame(get_line(p, feature_data, False))
        model_df = pd.concat([model_df,row.T])
    model_df = model_df.reset_index(drop = True)
    model_df.to_csv("saved_data/model.csv", index = False)