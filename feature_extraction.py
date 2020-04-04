import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle as pkl
from sklearn.linear_model import LinearRegression
import scipy.signal
from dtaidistance import dtw
import warnings 
warnings.filterwarnings("ignore")
import argparse
from utils import *




def feature_extractor(args):
    patientwise_data = pkl.load(open("saved_data/patientwise_data.pkl","rb"))
    fs = args.fs
    wlen = args.wlen*fs
    overlap = args.overlap

    probe_dict = {"TX": ['TX_R','TX_IR'],
                  "wrist": ["Probe1_R","Probe1_IR"],
                  "finger": ["Probe2_R","Probe2_IR"],
                  "forehead": ["Probe3_R","Probe3_IR"],
                  "chest" : ["Probe4_R","Probe4_IR"]}

    Ref_probe = probe_dict["finger"]
    TX_probe= probe_dict["TX"]

    SPO2_model = calibrate_and_get_model(False)

    SPO2_trends = {}




    feature_data = pd.DataFrame()
    tx_feature_data = pd.DataFrame()

    tx_components_data = pd.DataFrame()
    components_data = pd.DataFrame()


    for patient in tqdm(patientwise_data.keys(), total = len(patientwise_data.keys())):

    #     spo2_runs = []
    #     print ("PATIENT: ",patient, end= " ")
        with tqdm(total = (((len(patientwise_data[patient]) - wlen)//int((1-overlap)*wlen))) + 1) as pbar:

            for i in range(0,len(patientwise_data[patient]),int((1-overlap)*wlen)):



                ref_raw_signal= patientwise_data[patient].loc[i:i+wlen , Ref_probe].values.reshape((-1,2))
                TX_raw_signal = patientwise_data[patient].loc[i:i+wlen, TX_probe].values.reshape((-1,2))

                ref_R, ref_IR = return_info(ref_raw_signal, wlen, True)
                TX_R,TX_IR = return_info(TX_raw_signal, wlen, True)



                if ref_R[0] is None or ref_IR[1] is None:

                    continue 



                R_val = calculate_R_from_cycle(TX_raw_signal,wlen, False, tx_mode=True)
                ref_R_val = calculate_R_from_cycle(ref_raw_signal, wlen, False, tx_mode = False)


                if ref_R_val is None or R_val is None:
                    pbar.update()

                    continue

                else:


                    spo2 = get_spo2(R_val, SPO2_model)

                    feature_array = extract_features(ref_raw_signal)
                    row = np.append(feature_array,[np.asarray(ref_R_val),np.mean(R_val), np.mean(spo2), patient])

                    temp= pd.DataFrame(row).T

                    feature_data = pd.concat([feature_data, temp]).reset_index(drop = True)
                pbar.update()

    feature_data.to_csv("saved_data/features.csv",index = False)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--fs',default = 600, type = int, help = 'Sampling Frequency')
    parser.add_argument('--wlen',default =4, type = int, help = 'Window Length in seconds')
    parser.add_argument('--overlap',default = 0.25, type = int, help = 'Fraction of overlap (default = 0.25)')
    
    args = parser.parse_args()
    
    feature_extractor(args)