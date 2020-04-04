import os 
import pandas as pd 
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_absolute_error

from datetime import datetime

from utils import calibrate

np.random.seed(42)

def make_report(truth, predictions, patient):
    diff = np.abs(truth - predictions)
    report = { "ID": [patient],
              "MAE":  [mean_absolute_error(truth, predictions)],
              "No. of Obs " : [len(diff)],
              "Accuracy":   [(len(diff) - len(diff[np.where(diff > 4 )]) )/ len(diff) * 100]
             }
    return pd.DataFrame(report)  
    
def predict(args, show, save, bland_altman):
    
    report = pd.DataFrame()
    now = datetime.now()      
    test = []
    for arg in args:
        test.append(arg)
    
    calibrated,model,(X,Y) = calibrate(test)

    feature_data = pd.read_csv("saved_data/features.csv")
    feature_data.columns = [i for i in range(len(feature_data.columns))]
    p_column= len(feature_data.columns)-1
    spo2_column = p_column - 1
    r_column = spo2_column -1
    ref_r_column = r_column - 1
    
    test_df= feature_data[feature_data[p_column].isin(test)].iloc[:,[ref_r_column,spo2_column,p_column]]
    mae = []
    for i in range(len(test)):
        p = test[i]
        c = int(calibrated[i])
        predictions = []
        
        
        m,c = model[model[0] == c][1].values,model[model[0]==c][2].values
        
        
        max_x = 0 
        for x in test_df[test_df[p_column] == p][ref_r_column]:
            if x > max_x:
                max_x = x
            predictions.append(m*x + c)

        predictions = np.asarray(predictions)
        predictions[np.where(predictions > 100 )] = 100
        predictions = predictions.reshape((-1,))
        truth = np.asarray(test_df[test_df[p_column] == p][spo2_column]).reshape((-1,))
        running_mae = mean_absolute_error(truth, predictions)
        mae.append(running_mae)
        
        report = pd.concat([report,make_report(truth, predictions, p)], axis = 0)
               
        
        if not(bland_altman):
            fig,ax = plt.subplots(1,1, figsize = [8,8])
            plt.title("Patient {}".format(p))
            ax.plot(X,Y)
            ax.scatter(test_df[test_df[p_column] == p][ref_r_column],test_df[test_df[p_column] == p][spo2_column])
            plt.text(max_x, np.max(predictions)+10, "mae = {:.2f}".format(running_mae), size=12,bbox=dict(boxstyle="round",ec=(0, 0, 0),fc=(1., 0.8, 0.8)))
            
        elif bland_altman:
            fig,ax = plt.subplots(1,2, figsize = [16,8])
            fig.suptitle("Patient {}".format(p))
            ax[0].plot(X,Y)
            ax[0].scatter(test_df[test_df[p_column] == p][ref_r_column],test_df[test_df[p_column] == p][spo2_column])
            ax[0].text(max_x, np.max(predictions)+10, "mae = {:.2f}".format(running_mae), size=12,bbox=dict(boxstyle="round",ec=(0, 0, 0),fc=(1., 0.8, 0.8)))

            diff_bland = truth - np.asarray(predictions)
            mean = (np.asarray(test_df[test_df[p_column] == p][spo2_column]) + predictions ) / 2
            md = np.mean(diff_bland)
            std = np.std(diff_bland)
            plt.scatter(mean,diff_bland)
            ax[1].axhline(md,color='black', linestyle='--',linewidth = 4)
            ax[1].axhline(md + 1.96*std, color='gray', linestyle='--',linewidth = 4)
            ax[1].axhline(md - 1.96*std, color='gray', linestyle='--',linewidth = 4)
            ax[1].axhline(md + 2, color='red', linestyle='--',linewidth = 4)
            ax[1].axhline(md - 2, color='red', linestyle='--',linewidth = 4)
            ax[1].set_xlabel("(y + y')/2", fontsize = 12)
            ax[1].set_ylabel("y - y'" ,fontsize = 12)
            
                               


        if save:
            if not(os.path.exists("vizualize_results")):
                   os.mkdir("vizualize_results")
            date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
            path_now = "vizualize_results/"+date_time
            if not(os.path.exists(path_now)):
                os.mkdir(path_now)
            plt.savefig(path_now+"/Patient {}.png".format(p))
            
            
        if show:
            plt.show()
    if save:
        report.to_csv(path_now+"/report.csv".format(p), index = False)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("test", type = int, nargs = "+", help = "Enter testing patient id")
    parser.add_argument("--show",  dest = "show", action = "store_true",)
    parser.add_argument("--save",dest = "save", action = "store_true")
    parser.add_argument("--bland-altman",dest = "bland_altman", action = "store_true")
    args = parser.parse_args()
    predict(args.test, args.show, args.save, args.bland_altman)
