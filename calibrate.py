import os 
import pandas as pd 
import numpy as np
import scipy.stats


def calibrate(args):
    
    
    train_df = pd.read_csv("saved_data/model.csv")
    train_df.columns = [i for i in range(len(train_df.columns))]

    feature_data = pd.read_csv("saved_data/features.csv")
    feature_data.columns = [i for i in range(len(feature_data.columns))]

    p_column= len(feature_data.columns)-1
    spo2_column = p_column - 1
    r_column = spo2_column -1
    ref_r_column = r_column - 1
    calibrated= []
    
    default_test = args

#     train_df = train_df[~(train_df[0].isin(default_test))].reset_index(drop = True)

    test_df= feature_data[feature_data[p_column].isin(default_test)].iloc[:,[ref_r_column,spo2_column,p_column]]




    for p in test_df[p_column].unique():
        if p in default_test:       

            for run in range(4,0,-1):
                lower = run*5 + 70
                upper = lower + 5
                try:
                    calibration_df = test_df[(test_df[p_column] == p) & (test_df[spo2_column] > lower) & (test_df[spo2_column] < upper)].sample(5)
                    break
                except ValueError:
                    continue

            line = []    
            predictions = []


            for i in calibration_df.index:
                spo2 = calibration_df.loc[i,spo2_column]
                R = calibration_df.loc[i,ref_r_column]
                m = train_df[1]
                c = train_df[2]
                x =np.asarray((spo2 - c)/m)
                minimum = np.argmin(np.abs(x-R))
                line.append(minimum)
                predictions.append(train_df.iloc[minimum,1]*R + train_df.iloc[minimum,2])

            x = np.linspace(0.2,1.5)
            mode = scipy.stats.mode(line)[0][0]

            y = train_df.iloc[mode,1]*x + train_df.iloc[mode,2]
            calibrated.append(train_df.iloc[mode,0])
            
            
    return calibrated, train_df, (x,y)
