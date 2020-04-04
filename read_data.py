import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle as pkl

patient_data = []
patient_meta = []

def read_data():
    with tqdm(total = len(np.sort(os.listdir(os.getcwd()+"/Dataset"))), desc = "Patient") as pbar:
        for i in range(len(np.sort(os.listdir(os.getcwd()+"/Dataset")))):

            path = os.getcwd()+"/Dataset"+'/'+np.sort(os.listdir(os.getcwd()+"/Dataset"))[i]

            for j in range(len(os.listdir(path))):

                if ".txt" in os.listdir(path)[j]:


                    with open(path +"/" + os.listdir(path)[j],'r') as f:
                        as_string = f.read()
                    d = as_string.split("\n")[:-1]


                    number_of_trials = as_string.count("Patient")

                    ind = []
                    if number_of_trials !=1:
                        for x in range(len(d)):
                            if d[x][0] == "P":
                                ind.append(x)
                            if len(ind) == number_of_trials:
                                break
                    else:
                        ind = [0]

                    for trial in range(len(ind)):

                        if len(ind)==1:
                            meta, data = d[ind[trial]:ind[trial]+5],d[ind[trial]+5:]
                            columns = meta[-1].split("\t")
                            columns[-1] = columns[-1].rstrip("\n")
                            values= np.array([float(r) for x in data for r in x.split("\t") ]).reshape((len(data),-1))
                            temp = pd.DataFrame(values, columns = columns).drop(columns = ["Battery_level","packet loss","packet number"])
                            patient_data.append(temp)
                            patient_meta.append(meta)

                        else:
                            try:
                                meta, data = d[ind[trial]:ind[trial]+5],d[ind[trial]+5:ind[trial+1]]
                            except IndexError:
                                meta, data = d[ind[trial]:ind[trial]+5],d[ind[trial]+5:]

                                pass
                            columns = meta[-1].split("\t")
                            columns[-1] = columns[-1].rstrip("\n")
                            values= np.array([float(r) for x in data for r in x.split("\t") ]).reshape((len(data),-1))
                            temp = pd.DataFrame(values, columns = columns).drop(columns = ["Battery_level","packet loss","packet number"])
                            patient_data.append(temp)
                            patient_meta.append(meta)
            pbar.update()  


    print ("Creating Data to dump") 
    df = pd.DataFrame()
    i=0
    
    with tqdm(total = len(patient_data)) as pbar:
        for meta,data in zip(patient_meta,patient_data):
            name = int(meta[0].split(":")[-1])
            if i!=0:
                if name in df.patient.unique():
                    name= name + 0.1

            temp = pd.DataFrame(data)
            temp["patient"] = name
            df = pd.concat([df,temp], axis = 0)
            i+=1
            pbar.update()

    print ("Dumping data") 
    data_dict = {}
    for k in np.sort(df.patient.unique()):
          data_dict[k] = df[df.patient == k].drop(columns = ['patient',  "Spo2", "Time", "BPM"]).reset_index(drop = True)
            
    if not(os.path.exists("saved_data")):
           os.mkdir("saved_data")
            
          
    with open("saved_data/patientwise_data.pkl", "wb") as f:
        pkl.dump(data_dict,f)
        
        
if __name__ =="__main__":
    read_data()