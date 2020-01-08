from utils import *
import pickle as pkl
detrended_patientwise = pkl.load(open("detrended.pkl",'rb'))
SPO2_trends = {}
SPO2_model = calibrate_and_get_model(False)
for patient in detrended_patientwise.keys():
    fs = 600
    wlen = 4*fs

    overlap = 0

    spo2_runs = []

    print ("PATIENT: ",patient, end= " ")

    for i in range(0,len(detrended_patientwise[patient]),int((1-overlap)*wlen) ):
        print ("start: ",i, "   end: ",i+wlen)

        signal = detrended_patientwise[patient].iloc[i:i+wlen , -2:].values.reshape((-1,2))
        # plt.subplot(121)
        # plt.plot(signal[:,0])
        # plt.subplot(122)
        # plt.plot(signal[:,1])
        # plt.show()
        R, IR = return_info(signal, wlen,)
        if R[0] is None or IR[1] is None:
            print("Discarding Signal")
            continue 
    # plot_signal(signal,wlen,)

        R_val = calculate_R_from_cycle(signal,wlen)
        if R_val is None:
            print ("SKIPPING")
            continue

        else:
            print (R_val)
            spo2_runs.append(get_spo2(R_val, SPO2_model))
    

            print ("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    SPO2_trends[patient] = spo2_runs
    with open("Patientwise_trend.pkl","wb") as f:
        pkl.dump(SPO2_trends,f)
