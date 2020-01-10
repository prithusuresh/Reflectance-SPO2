from utils import return_info,calculate_R_from_cycle,get_spo2, calibrate_and_get_model
import pickle as pkl

detrended_patientwise = pkl.load(open("detrended.pkl",'rb'))
patientwise_data = pkl.load(open("patientwise_data.pkl","rb"))

TX = ['TX_R',	'TX_IR']
wrist = ["Probe1_R",	"Probe1_IR"]
finger = ["Probe2_R",	"Probe2_IR"]
forehead = ["Probe3_R",	"Probe3_IR"]
Probe4 = ["Probe4_R",	"Probe4_IR"]


SPO2_trends = {}
erroneous_data = []
SPO2_model = calibrate_and_get_model(False)
print ("Beginning Run")
for patient in detrended_patientwise.keys():
    print 
    fs = 600
    wlen = 4*fs

    overlap = 0

    spo2_runs = []

    print ("PATIENT: ",patient, end= " ")

    for i in range(0,len(detrended_patientwise[patient]),int((1-overlap)*wlen) ):
        print ("start: ",i, "   end: ",i+wlen)
        
    raw_signal = patientwise_data[patient].loc[i:i+wlen , TX].values.reshape((-1,2))

    R, IR = return_info(raw_signal, wlen, False)
    if R[0] is None or IR[1] is None:
        print("Discarding Signal")
        continue 
  # plot_signal(raw_signal,wlen,)
    R_val = calculate_R_from_cycle(raw_signal,wlen)
    if R_val is None:
        print ("SKIPPING")
        continue

    else:
        print (R_val)
        spo2_runs.append(get_spo2(R_val, SPO2_model))
    

        if get_spo2(R_val, SPO2_model) < 80 or get_spo2(R_val, SPO2_model) > 100:
            meta_dict = {}
            meta_dict["patient"] = patient
            meta_dict["SPO2"] = get_spo2(R_val, SPO2_model)
            meta_dict["Window Index"] = [i,i+wlen]
            meta_dict["R_values"] = R_val
            meta_dict["Signal_information"] = return_info(raw_signal, wlen, True)
            meta_dict["Components"] = [calculate_R_from_cycle.R_components,calculate_R_from_cycle.IR_components]
            erroneous_data.append(meta_dict)

        print ("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        
    
    SPO2_trends[patient] = spo2_runs
    with open("../Patientwise_trend.pkl","wb") as f:
        pkl.dump(SPO2_trends,f)
with open("../erroneous_meta.pkl", "wb") as f:
    pkl.dump(erroneous_data, f)
                
    
