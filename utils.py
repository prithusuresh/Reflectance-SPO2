import os
import numpy as np 
import pandas as pd
from pprint import pprint 
import pickle as pkl
import scipy
import scipy.signal

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams["figure.figsize"] = [30,3]

def fit_trendline(y, fs = 600):

    '''Fit Trendline for detrending '''

    model = LinearRegression()
    X = np.array([j for j in range(len(y))])

    X= X.reshape((-1,1))
    x = np.concatenate([X,X**2], axis = 1)
    model.fit(x,y)
    pred = model.predict(x)

    t = X.reshape(len(X))/fs
    return np.array([y[j] - pred[j] + np.mean(y) for j in range(X.shape[0])]), pred, t

def smooth(signal,window_len=50):
    ''' Smoothen and detrend signal by removing 50 Hz using notch and Savitzky-Golay Filter for smoothening''' 
  # num,denom = scipy.signal.iirnotch(50,1,600) ### How is the quality factor choosen?
  # notched = scipy.signal.lfilter(num,denom,signal)
  # y = savgol_filter(notched, window_len, 1) ### How is the window length and the polynomial required choosen?
  # detrend, pred, t = fit_trendline(y)
    y = pd.DataFrame(signal).rolling(window_len,center = True, min_periods = 1).mean().values.reshape((-1,))
    return y

def peaks_and_valleys(signal, prominence = 300, is_smooth = True , distance = 250):

    """ Return prominent peaks and valleys based on scipy's find_peaks function """

    if is_smooth:

        smoothened = smooth(signal)
        peak_loc = scipy.signal.find_peaks(smoothened, prominence = prominence, distance = distance)[0] #,scipy.signal.find_peaks(smoothened, prominence = prominence, distance = distance )[1]["prominences"]
    
        signal = signal*(-1)
        smoothened = smooth(signal)
        valley_loc = scipy.signal.find_peaks(smoothened, prominence = prominence,distance = distance)[0]
    
        final_peaks_loc, final_valley_loc = discard_outliers(smooth(signal),peak_loc,valley_loc)
    else:
        peak_loc = scipy.signal.find_peaks(signal, prominence = prominence,distance = distance)[0]
        signal = signal*(-1)
        valley_loc = scipy.signal.find_peaks(signal, prominence = prominence,distance = distance)[0]
    
        final_peaks_loc, final_valley_loc = discard_outliers(signal,peak_loc,valley_loc)
 
    return final_peaks_loc, final_valley_loc

def discard_outliers(signal,peaks,valleys):
    """Find peaks or valleys in a signal. 
    Returns peak, and groups of valleys"""

  
  
    val = [[valleys[x-1],valleys[x]] for x in range(1,len(valleys))]
  
    peak = {}
    for i in range(len(val)) :
        x= val[i]
        try:     
            peak[i] = int(peaks[np.where(np.logical_and(peaks> x[0],peaks < x[1]))[0][0]])
        except:
            i+=1
            pass
  
    i=0
    while i<len(val):
        if i not in peak.keys():
            val.pop(i)
        else:
            i+=1
    try:
        assert len(peak) == len(val)
    except AssertionError:
        print ("Unequal peak,valleys, skipping signal. ")
        return None,None
    
    final_val =list(set([x for j in val for x in j]))
    final_val.sort()
    final_peak = [peak[i] for i in peak.keys()]
    final_peak.sort()
 
    return final_peak,final_val

def return_info(signal,wlen, is_smooth = True):
    """ Get smoothened signal, peak location and valley location """
    R_signal = signal[:,0].reshape((-1,))
    IR_signal = signal[:,1].reshape((-1,))
    peaks_R, valleys_R = peaks_and_valleys(R_signal, is_smooth)
    peaks_IR, valleys_IR = peaks_and_valleys(IR_signal, is_smooth)
  
    if peaks_R is None or valleys_R is None or peaks_IR is None or valleys_IR is None:
        return [None, None, None], [None, None,None]
    if is_smooth:
        return [smooth(R_signal),peaks_R,valleys_R],[smooth(IR_signal),peaks_IR,valleys_IR]
    else:
        return [R_signal,peaks_R,valleys_R],[IR_signal,peaks_IR,valleys_IR]

def plot_signal(signal,wlen, is_smooth = True):

    """ plot Red and IR signals along with peaks and valleys"""

    [R_signal,peaks_R, valleys_R],[IR_signal,peaks_IR, valleys_IR] = return_info(signal,wlen, is_smooth)

    if R_signal is not None and IR_signal is not None:
        plt.subplot(121)
        plt.title("SMOOTHENED")
        plt.scatter(valleys_R[0],R_signal[valleys_R[0]],c = "b")
        plt.plot(R_signal)
        plt.subplot(122)
        plt.plot(IR_signal)
        plt.show()

        plt.subplot(121)
        plt.title("LOCATIONS")
        plt.plot(R_signal)
        plt.scatter(peaks_R ,R_signal[peaks_R], c= "r")
        plt.scatter(valleys_R,R_signal[valleys_R], c= "g")
        plt.subplot(122)
        plt.plot(IR_signal)
        plt.scatter(peaks_IR, IR_signal[peaks_IR], c= "r")
        plt.scatter(valleys_IR , IR_signal[valleys_IR], c= "g")
        plt.show()

def assess_quality(R,IR):
    """ get cross-correlation coefficient"""
    R_signal= R[0]
    IR_signal = IR[0]
    return np.corrcoef(R_signal,IR_signal)[0][1]

def extract_cycles(data, show = True,harshness = 0.5):

    """extract location of only good cycles """

    signal,peaks,valleys = data[0],data[1],data[2]
  
    val = [[valleys[x-1],valleys[x]] for x in range(1,len(valleys))]
    peak = []
    i=0
    while i<len(val):
        try: 
            peak.append(peaks[int(np.where(np.logical_and(peaks > val[i][0],peaks < val[i][1]))[0][0])])
            i+=1
        except:
            val.pop(i)

    val = np.array(val,ndmin = 2)
    mean = np.mean(np.diff(val,axis=1))
    std = np.std(np.diff(val,axis=1))

    while harshness <=1:
        good_valleys = [x for x in val if np.diff(x) > mean-harshness*std and np.diff(x) < mean+harshness*std]
        if good_valleys == []:
            harshness +=0.1
        else:
            break    
    try: 
        assert good_valleys!=[]
    except AssertionError:
        print ("Bad Signal")
        return None,None
    
    good_peaks = []
    i=0
    for i in good_valleys: 
       good_peaks.append(peak[int(np.where(np.logical_and(peak > i[0], peak < i[1]))[0][0])])

    good_valleys = np.asarray(good_valleys)
    return good_peaks, good_valleys


def calculate_R_from_cycle(signal, wlen, show = False):

    """ Calculate Final R value """
    R,IR = return_info(signal,wlen)
    corr = assess_quality(R,IR)
    peaks_R,valley_groups_R = extract_cycles(R)
    peaks_IR,valley_groups_IR = extract_cycles(IR)

    if peaks_R is None or valley_groups_R is None or peaks_IR is None or valley_groups_IR is None:
    
        print ("Not Returning R_value. Poor Signal")
        return None
    else:
        print (valley_groups_R, valley_groups_IR)
 
 
        if corr > 0.95:
            print ("Good")

        else:
            return None
        final_valleys = np.concatenate([valley_groups_R, valley_groups_IR])
        final_valleys  = np.unique(final_valleys, axis = 1)
        final_peaks = np.union1d(peaks_R,peaks_IR)    

        if show:
            unravel_val = list(set([x for y in final_valleys for x in y]))
            unravel_val.sort()
            plt.subplot(121)
            plt.title("Final cycles")
            plt.plot(R[0])
            plt.scatter(final_peaks, R[0][final_peaks], c= "r")
            plt.scatter(final_valleys, R[0][final_valleys], c= "g")
            plt.subplot(122)
            plt.title("Final cycles")
            plt.plot(IR[0])
            plt.scatter(final_peaks, IR[0][final_peaks], c= "r")
            plt.scatter(final_valleys, IR[0][final_valleys], c= "g")
            plt.show()

        R_ratio = ac_dc(R[0], final_peaks, final_valleys)
        IR_ratio = ac_dc(IR[0], final_peaks, final_valleys)

        try:
            assert R_ratio.shape == IR_ratio.shape
        except AssertionError:
            minimum= min(R_ratio.shape[0], IR_ratio.shape[0])
            R_ratio = R_ratio[:minimum]
            IR_ratio = IR_ratio[:minimum]                      

        R_value = R_ratio/IR_ratio
        return R_value

def ac_dc(signal,peaks, val):

    """ Get ac and dc component of each signal"""
    dc = []

    ac = []

    for i in range(val.shape[0]):
        try:
            assert signal[peaks[i]] > (signal[val[i][0]]+signal[val[i][1]])/2
        except AssertionError:
            np.delete(val,i)
            continue
        except IndexError:
            break
  
        dc.append((signal[val[i][0]]+signal[val[i][1]])/2)
        ac.append(signal[peaks[i]] - (signal[val[i][0]]+signal[val[i][1]])/2)
 
    print ("SHAPES MATCHING? :", len(dc),len(ac))

    ratio = np.array(ac)/np.array(dc)

  
    return ratio

def calibrate_and_get_model(show = False):
    with open("Calibration_curve.txt", "r") as f:
        points = f.readlines()

    spo2 = np.array([int(x.split("\t")[0]) for x in points])
    R_truth = np.array([float(x.split("\t")[1].rstrip("\n")) for x in points]).reshape((-1,1))
    SPO2_model = LinearRegression()
    SPO2_model.fit(np.concatenate([R_truth, R_truth**2],axis = 1), spo2)

    if show: 
        pred = SPO2_model.predict(np.concatenate([R_truth, R_truth**2],axis = 1))
        plt.plot(R_truth, pred)
        plt.show()

    return SPO2_model

def get_spo2(R_values, SPO2_model):
    R_values = np.asarray(R_values).reshape((-1,1))
    pred = SPO2_model.predict(np.concatenate([R_values,R_values**2], axis = 1))  
    print ("SPO2: ", np.mean(pred))
    return np.mean(pred)
