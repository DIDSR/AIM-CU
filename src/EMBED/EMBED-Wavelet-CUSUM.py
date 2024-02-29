import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import random
sns.set_style('darkgrid')
np.random.seed(1004)

random.seed(58)

import pandas as pd
from datetime import datetime
from matplotlib.ticker import StrMethodFormatter
import os
import sys

import warnings
warnings.filterwarnings("ignore")

def stats(df): #the function displays dataframe size, countings of unique patients and unique exams
    print('Dataframe size: ' + str(df.shape))
    try:
        print('# patients: ' + str(df.patient_id.nunique()))
    except:
        print('# patients: ' + str(df.patient_id.nunique()))
    print('# exams: ' + str(df.acc_anon.nunique()))
    
#Compute CUSUM for the observations in x (specificity in this case)
def compute_cusum(x, mu, k):
    #CUSUM for day0-2000: outcomes are detection delay and #FP, #TP, MTBFA, False alarm rate
    num_rows        = np.shape(x)[0]
    
    x_mean = np.zeros(num_rows,dtype=float)
    #S_hi : for positive changes --------------------------
    S_hi = np.zeros(num_rows,dtype=float)
    S_hi[0] = 0.0 # starts with 0
    #Increase in mean = x-mu-k ----------------------------
    mean_hi = np.zeros(num_rows,dtype=float)

    #Decrease in mean = mu-k-x----------------------------
    mean_lo = np.zeros(num_rows,dtype=float)
    #S_lo : for negative changes --------------------------
    S_lo = np.zeros(num_rows,dtype=float)
    S_lo[0] = 0.0 # starts with 0
    #CUSUM: Cumulative sum of x minus mu ------------------
    cusum = np.zeros(num_rows,dtype=float)
    cusum[0] = 0.0 # initialize with 0
    
    for i in range(0, num_rows):
        x_mean[i]  = x[i] - mu  #x_mean 
        mean_hi[i] = x[i] - mu - k
        S_hi[i]    = max(0, S_hi[i-1] + mean_hi[i])
        mean_lo[i] = mu - k - x[i]
        S_lo[i]    = max(0, S_lo[i-1] + mean_lo[i])
        cusum[i]   = cusum[i-1] + x_mean[i]

    x_mean  = np.round(x_mean,decimals=2)
    S_hi    = np.round(S_hi,decimals=2)
    mean_lo = np.round(mean_lo,decimals=2)
    S_lo    = np.round(S_lo,decimals=2)
    cusum   = np.round(cusum,decimals=2)

    # Construct the tabular CUSUM Chart  
    chart = np.array([])
    chart = np.column_stack((x.T, x_mean.T, mean_hi.T, S_hi.T, mean_lo.T, S_lo.T, cusum.T))
    np.round(chart, 2)

    #d = 2 *(np.log((1-0.01) / (0.0027)))
    #h = d * 0.5 # h= d*k where k=0.5
    #h = 4 # as per the NIST doc on CUSUM

    #l1 =  np.append(num_rows, data_tabular, axis = 1)
    #l1 = np.concatenate(num_rows.T, data_tabular.T)
    #chart = np.column_stack((num_rows.T, data_tabular.T))
    #chart

    np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})
    #print("CUSUM Chart is:\n", np.round(chart,decimals=2))
    #x_mean

    df_out = pd.DataFrame(chart) 
    df_out.columns = ['X','x-mu','Increase in Mean', 'S_hi', 'Decrease-in-mean', 'S_lo', 'CUSUM']
    #filename = "file%d" %runs
    #df_out.to_csv(("CUSUM-out/file%d.csv" %runs), sep='\t')    
    #print(df.to_string())
    #print(chart)
    #Export datafrae to png
    #import dataframe_image as dfi
    #dfi.export(df,'CUSUM-out/CUSUM-run.png')
    
    return S_hi, S_lo, cusum

def cusum_detection(df,df_wavel7,control_limit,delta,simulation_days,samples_per_day):
    #Fetch 100 patients per day (day1-60) from df and days 61-120 from df_medk10 which is the out of control region
    # compute 

    FalsePos         =  np.array([])
    TruePos          =  np.array([])
    DelaytoDetect    =  np.array([])
    FAR              =  np.array([])     #False Alarm Rate
    inSTD_test_sp    =  np.array([])     #Standard deviation of test AUCs
    outSTD_test_sp   =  np.array([]) 
    D                =  np.array([])     #Displacement
    h_1000           =  np.array([]) 
    k_1000           =  np.array([])
    DetectionTimes   =  np.array([],dtype=int)
    Dj               =  np.array([],dtype=int) #save the Dj which are binary values indicating detection MTBFA
    Zj               =  np.array([],dtype=int) #save the Zj = min(Tj,pre-change-days)-MTBFA
    zj               =  np.array([],dtype=int) # ADD - MLE of delays
    cj               =  np.array([],dtype=int) # ADD - binary
    AvgDD            = np.array([])      # Average Detection Delay
    sample_size      = samples_per_day           # samples per day
    pre_change_days  = simulation_days   # pre- or post-change days
    post_change_days = simulation_days
    total_days       = pre_change_days + post_change_days
    patients_in      = df.patient_id.unique()
    patients_o       = df_wavel7.patient_id.unique()
    sp_pre           = np.array([])
    sp_post          = np.array([])
    runs             = 0
    #delta            = 1

    while (runs < 1000):
        days        = 0
        start_in    = 0
        end_in      = samples_per_day
        start_out   = 0 
        end_out     = samples_per_day
        specificity = np.array([])
        while (days < pre_change_days):
            patients100   = patients_in[start_in:end_in]

            #Fetch all the rows for 100 patients
            p100     = df[df['patient_id'].isin(patients100)]

            #print("Checking stats for 100 patients")
            #stats(p100)
    
            #threshold = 0.31
            threshold_pre = 0.0177
            FP = p100[p100['preds'] > threshold_pre]
            TN = p100[p100['preds'] < threshold_pre]

            #print("Total rows:",      p100.index.size)
            #print("#Below Threshold", TN.index.size)
            #print("#Above Threshold", FP.index.size)
            sp = TN.index.size/p100.index.size
            specificity  = np.append(specificity, sp)
            sp_pre       = np.append(sp_pre, sp)
        
            start_in += sample_size
            end_in   += sample_size
            days     += 1 
    
        while (days < total_days):
            patients100_out   = patients_o[start_out:end_out]

            #Fetch all the rows for 100 patients
            p100_out     = df_wavel7[df_wavel7['patient_id'].isin(patients100_out)]

            #print("Checking stats for 100 patients")
            #stats(p100)
    
            #threshold = 0.31
            #threshold_post = 0.0333  
            #threshold_post = 0.03545  
            threshold_post = 0.0177
            FP_o = p100_out[p100_out['preds'] > threshold_post]
            TN_o = p100_out[p100_out['preds'] < threshold_post]

            #print("Total rows:",      p100.index.size)
            #print("#Below Threshold", TN.index.size)
            #print("#Above Threshold", FP.index.size)
            sp_o           = TN_o.index.size/p100_out.index.size
            specificity    = np.append(specificity, sp_o)
            sp_post        = np.append(sp_post, sp_o)
        
            start_out += sample_size
            end_out   += sample_size
            days      += 1
        
        #CUSUM for day0-2000: outcomes are detection delay and #FP, #TP, MTBFA, False alarm rate
        num_rows        = np.shape(specificity)[0]
        in_control_sp   = specificity[:pre_change_days]
        out_control_sp  = specificity[pre_change_days:total_days]
        out_std_sp      = np.std(out_control_sp)
        in_std_sp       = np.std(in_control_sp)
        x               = np.array(specificity)

        mu     = np.mean(in_control_sp)
        mu_1   = np.mean(out_control_sp)
        std    = np.std(in_control_sp)
        std_1  = np.std(out_control_sp)
        displacement = (mu_1-mu)/std
    
        #h      = 0.102       # Upper/lower control limit to detect the changepoint H=0.102, 0.127 
        #k      = 0.03831     # Drift 0.01277 is the 1 sigma change, 0.0255 - one-sigma change, 0.03831 is 3-sigma change, 0.05108
        h      = in_std_sp * control_limit
        k      = (delta * in_std_sp)/2
   
        #Call compute CUSUM function with x (observatoins), in-control mean (mu) and k (drift or reference value)
        S_hi, S_lo, cusum = compute_cusum(x, mu, k)
    
    
        # False positives and Total alarms
        falsePos = 0
        alarms   = 0
        delay    = 0
        avddd    = 0   # this is the delay from the paper: td-ts (z_k-v) where v is the changepoint and z_k is the time of detection
        #MTBFA    = 0
    
        for i in range(0, pre_change_days):
            if ((S_hi[i] > h) or (S_lo[i] > h)):   
                #if (i<pre_change_days):
                falsePos += 1  #False Positives 
                #print("time false alarm",i)
                DetectionTimes= np.append(DetectionTimes, i+1)   #time at which a false positive is detected
                Dj = np.append(Dj, 1)
                Zj = np.append(Zj, min(i,pre_change_days))
                #print("detection times",DetectionTimes)
                #print("detection times size",DetectionTimes.size)
                break
        
        # If there is no false positive, Zj = pre_change_days, Dj = 0
        if falsePos == 0:
            Dj = np.append(Dj, 0)
            #DetectionTimes[runs] = pre_change_days
            Zj = np.append(Zj, pre_change_days) 

        # Delay to detect the first changepoint
        #delay = 0
        for i in range(pre_change_days, total_days):
            if ((S_hi[i] > h) or (S_lo[i] > h)):
                alarms += 1           #True Positive: break after detecting one TP
                #print("alarm at : ", i)
                #delay  = i-1000+1    # ts is 100 because the change starts at day100
                avddd  = i-pre_change_days
                cj = np.append(cj, 1)
                zj = np.append(zj, min(avddd,total_days))
                break
        # If there is no true detection, zj = total simulation days, cj = 0
        if alarms == 0:
            cj = np.append(cj, 0)
            #DetectionTimes[runs] = pre_change_days
            zj = np.append(zj, total_days) 
    
        #Calculate MTBFA(Mean time time between False Alarms)
        #MTBFA = np.mean(DetectionTimes)
        #FlaseAlarmRate = 1/MTBFA
    
        FalsePos       = np.append(FalsePos, falsePos)
        TruePos        = np.append(TruePos, alarms)
        #DelaytoDetect = np.append(DelaytoDetect, delay)   # td-ts+1
        #FAR           = np.append(FAR, FlaseAlarmRate)
        #DetectionTimes= np.append(DetectionTimes, detectionTime)
        AvgDD          = np.append(AvgDD, avddd)   # ADD estimate from the paper
        outSTD_test_sp = np.append(outSTD_test_sp, out_std_sp)
        inSTD_test_sp  = np.append(inSTD_test_sp, in_std_sp)
        D              = np.append(D, displacement)
        h_1000         = np.append(h_1000, h)
        k_1000         = np.append(k_1000, k)
        #print(falsePos)    
    
        
        #Shuffle the patient list for the next simulation
        random.shuffle(patients_in)
        random.shuffle(patients_o)
        runs      += 1  # continue until end of simulation
    
    print("--------------------------------")
    print("Control Limit:", control_limit)
    print("Reference Value:", delta)
    print("Pre/Post Change Days:", change_days)
    print("Samples per day:", samples_per_day)
    print("--------------------------------")
    print("total number of False Positives:",np.sum(FalsePos))
    print("Total True Negatives:",np.sum(TruePos))
    print("Total Missed Detections:",runs-np.sum(TruePos))
    print("Average Detection Delay",np.mean(AvgDD))
    print("Average Detection Delay NEW:",np.sum(zj)/np.sum(cj))
    print("Minimum Delay",np.min(AvgDD))
    print("Maximum Delay",np.max(AvgDD))
    MTBFA = np.mean(DetectionTimes)
    MLP = np.sum(Dj)/np.sum(Zj)
    MTBFA_new = 1/MLP
    FlaseAlarmRate = 1/MTBFA
    print("MTBFA", MTBFA)
    print("MTBFA new", MTBFA_new)
    print("Flase Alarm Rate", FlaseAlarmRate)    
    print("Mean ref. Value", np.mean(k_1000))    
    print("Mean STD_0", np.mean(inSTD_test_sp)) 
    print("Mean STD_1", np.mean(outSTD_test_sp))
    print("in-control mean", mu)
    print("out-of-control mean", mu_1)
    print("Displacement, d:",(mu_1-mu)/std)
    print("Mean Displacement:", np.mean(D))
    
    
if __name__ == "__main__":
    #----in-control---specificity----
    df = pd.read_csv(r"/gpfs_projects/ravi.samala/OUT/2023_MLDrift/20230803-091724__input_list_file_with_output_scores.csv")
    #df = pd.read_csv(r"/gpfs_projects/ravi.samala/OUT/2023_MLDrift/20230803-091724__by_patient_scores.csv")
    
    #----out-of-control---specificity--Median Denoised--
    #df_medk10 = pd.read_csv(r"/gpfs_projects/smriti.prathapan/EMBED/OUT/denoised/p6-12k/medk10/20230804-192403__input_list_file_with_output_scores.csv")
    #df_medk10 = pd.read_csv(r"/gpfs_projects/smriti.prathapan/EMBED/OUT/denoised/p6-12k/medk10/20230804-192403__by_patient_scores.csv")
    
    #----out-of-control---specificity----Wavelet Denoised--(Bayes, level7)-----
    df_wavel7 = pd.read_csv(r"/gpfs_projects/smriti.prathapan/EMBED/OUT/denoised/p6-12k/wavelet-l7-u16/20231017-233737__input_list_file_with_output_scores.csv")
    #df_wavel7 = pd.read_csv(r"/gpfs_projects/ravi.samala/OUT/2023_Smriti/OUT/denoised/cancer-list/wavelet-l7-u16/20231020-152417__by_patient_scores.csv")
   
    #control_limit = 5
    #delta         = 2
    
    print("Number of arguments:", len(sys.argv))
    print("Argument List:", str(sys.argv))
    
    control_limit, delta, change_days, samples_per_day = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    
    #FalsePos, TruePos, AvgDD, DetectionTimes = detection (df,df_wavel7,threshold,delta)
    cusum_detection (df,df_wavel7,control_limit,delta,change_days,samples_per_day)
