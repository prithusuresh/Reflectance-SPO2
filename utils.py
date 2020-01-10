import os
import numpy as np 
import pandas as pd
import pickle as pkl
import scipy
import scipy.signal

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams["figure.figsize"] = [30,3]

def detrend(signal):
    R, IR = signal[:,0],signal[:,1]
    R_detrended = fit_trendline(R)[0]
    IR_detrended = fit_trendline(IR)[0]

    signal = np.concatenate([R_detrended,IR_detrended], axis = 1)

    return signal
def fit_trendline(y, fs = 600): 
    '''Fit Trendline for detrending '''
    model = LinearRegression()
    X = np.array([j for j in range(len(y))])

    X= X.reshape((-1,1))
    x = np.concatenate([X,X**2], axis = 1)
    model.fit(x,y)
    pred = model.predict(x)

    t = X.reshape(len(X))/fs
    return [np.array([y[j] - pred[j] + np.mean(y) for j in range(X.shape[0])]).reshape((-1,1)), pred, t]

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

def return_info(signal,wlen, is_smooth):
  """ Get smoothened signal, peak location and valley location 
  Input: 
    signal: RAW SIGNAL"""

  detrended_signal = detrend(signal)
  R_signal = detrended_signal[:,0].reshape((-1,))
  IR_signal = detrended_signal[:,1].reshape((-1,))
  peaks_R, valleys_R = peaks_and_valleys(R_signal, is_smooth)
  peaks_IR, valleys_IR = peaks_and_valleys(IR_signal, is_smooth)
  
  if peaks_R is None or valleys_R is None or peaks_IR is None or valleys_IR is None:
    return [None, None, None], [None, None,None]
  if is_smooth:
    return [smooth(signal[:,0]),peaks_R,valleys_R],[smooth(signal[:,1]),peaks_IR,valleys_IR]
  else:
    return [signal[:,0], peaks_R,valleys_R],[signal[:,1],peaks_IR,valleys_IR]

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
        if len(good_valleys)<3:
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


def valley_matching(R_valleys, IR_valleys):
  R_valleys = np.array(R_valleys, ndmin = 2)
  IR_valleys = np.array(IR_valleys, ndmin = 2)

  minimum = min(R_valleys.shape[0], IR_valleys.shape[0])
  print (minimum)
  dist = {0:[]}
  for r in R_valleys:
    for i in IR_valleys: 
      if np.sum(abs(i-r)) <= 10:
        loc = i

        if not any(loc is x for x in dist[0]):
          
          dist[0].append(loc)
  valleys = dist[0][:minimum]
  print (valleys)
  # /[dist[x][0] for x in np.sort(np.array(list(dist.keys())))[:minimum]]

  return valleys


def peak_matching(R_peaks, IR_peaks):
  R_peaks = np.array(R_peaks)
  IR_peaks = np.array(IR_peaks)
  
  minimum = min(R_peaks.shape[0], IR_peaks.shape[0])
  print (minimum)
  dist = {0:[]}
  for r in R_peaks:
    for i in IR_peaks: 
      if np.sum(abs(i-r)) <= 10:
        loc = i
        if loc not in dist[0]:
          dist[0].append(loc)
  peaks = dist[0][:minimum]
  # [dist[x][0] for x in np.sort(np.array(list(dist.keys())))[:minimum]]
  peaks.sort()

  return peaks




def calculate_R_from_cycle(signal, wlen, show = False):

    """ Calculate Final R value """

    

    R,IR = return_info(signal,wlen, True)
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
        peaks_R,valley_groups_R = extract_cycles(R)
        peaks_IR,valley_groups_IR = extract_cycles(IR)


        final_valleys = np.array(valley_matching(valley_groups_R, valley_groups_IR))
        final_peaks = list(set(peak_matching(peaks_R,peaks_IR)))
        if show:
            unravel_val = list(set(final_valleys.ravel()))
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
        calculate_R_from_cycle.R_components = [ac_dc.ac, ac_dc.dc]
        IR_ratio = ac_dc(IR[0], final_peaks, final_valleys)
        calculate_R_from_cycle.IR_components = [ac_dc.ac, ac_dc.dc]
        
        
        if R_ratio is None or IR_ratio is None:
            return None
        else:
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
    ac_dc.dc = []

    ac_dc.ac = []

    for i in range(val.shape[0]):
        try:
            assert signal[peaks[i]] > (signal[val[i][0]]+signal[val[i][1]])/2
        except AssertionError:
            np.delete(val,i)
            continue
        except IndexError:
            break
  
        ac_dc.dc.append((signal[val[i][0]]+signal[val[i][1]])/2)
        ac_dc.ac.append(signal[peaks[i]] - (signal[val[i][0]]+signal[val[i][1]])/2)
 
    print ("SHAPES MATCHING? :", len(ac_dc.dc),len(ac_dc.ac))
    if len(ac_dc.ac) == 0 or len(ac_dc.dc) == 0:
        return None

    ratio = np.array(ac_dc.ac)/np.array(ac_dc.dc)

  
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
