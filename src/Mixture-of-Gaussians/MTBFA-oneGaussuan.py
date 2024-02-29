
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd

sns.set_style('darkgrid')
np.random.seed(42)
#MLP Classifier
# Load the required libraries
import sklearn
#print(samples.shape)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings
import glob, os
warnings.filterwarnings('always') 
warnings.filterwarnings('ignore')

def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

# Plot bivariate distribution
def generate_surface(mean, covariance, d):
    """Helper function to generate density surface."""
    nb_of_x = 50 # grid size
    x1s = np.linspace(-6, 10, num=nb_of_x)
    x2s = np.linspace(-6, 10, num=nb_of_x)
    x1, x2 = np.meshgrid(x1s, x2s) # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_x):
        for j in range(nb_of_x):
            pdf[i,j] = multivariate_normal(
                np.matrix([[x1[i,j]], [x2[i,j]]]), 
                d, mean, covariance)
    return x1, x2, pdf  # x1, x2, pdf(x1,x2)

def gaussian_data(n, mean11, mean21, covariance11, covariance21):
    """Draw n samples from the Gaussian Distribution"""
    d = 2 # Number of dimensions per Gaussian
    # Define the mean for Gaussian in Class1
    #mean11 = np.matrix([[0.], [0.]])

    # Define the mean for each Gaussian in Class2
    #mean21 = np.matrix([[1.91], [0.]])  

    # Define the covarience for Gaussian in Class1
    #covariance11 = np.matrix([
    #[1, 0],
    #[0, 1]
    #])

    # Define the covarience for Gaussian in Class2
    #covariance21 = np.matrix([
    #[1, 0],
    #[0, 1]
    #])

    # Create L for each Gaussian and concatenate in Class 1
    L11 = np.linalg.cholesky(covariance11)

    # Create L for each Gaussian and concatenate in Class 2
    L21 = np.linalg.cholesky(covariance21)
    #-----------------------------Class 1--------------------------
    # Sample X from standard normal for Class 1
    #n = 8000 # Samples to draw(initial=50) this was 3000 in all the previous data created so far

    X11 = np.random.normal(size=(d, n))


    # Create a col of 0s for label 0 (Class1)
    label_0 = np.zeros(shape = (n,1), dtype = int)
    #print(label_0.shape)

    # Apply the transformation
    Y11 = L11.dot(X11) + mean11


    #Create Y and append the values for Class 1
    Y11 = np.array(Y11)

    #Y1 = np.concatenate((Y11.T))
    # Add lablels to Class1
    Y_l0 = np.append(Y11.T, label_0, axis = 1)


    # Plot the samples and the distribution for CLass 1
    #fig, ax = plt.subplots(figsize=(6, 4.5))
    # Plot bivariate distribution for Class 1
    x11, x12, p11 = generate_surface(mean11, covariance11, d) # Call this multiple times

    x11 = np.array(x11)
    x12 = np.array(x12)


    X11 = np.concatenate((x11))
    X12 = np.concatenate((x12))

    p11 = np.array(p11)

    p1 = np.concatenate((p11))

    #-----------------------------Class 2--------------------------
    # Sample X from standard normal for Class 2
    X21 = np.random.normal(size=(d, n))

    # Apply the transformation
    Y21 = L11.dot(X21) + mean21


    # Create a col of 0s for label 0 (Class 1)
    label_1 = np.ones(shape = (n,1), dtype = int)

    #Create Y and append the values for Class 2
    Y21 = np.array(Y21)

    #Y2 = np.concatenate((Y21.T))
    #Y2 = np.concatenate((Y21,Y22,Y23,Y24)) #deleted Y24 for sanity check

    #Y2 = np.concatenate((Y21.T,Y22.T,Y23.T)) #sanity check - comment when not
    # Add lablels to Class2
    #Y_l1 = np.append(Y1, label_1, axis = 1) # sanity check
    Y_l1 = np.append(Y21.T, label_1, axis = 1)
    data = np.concatenate((Y_l0,Y_l1))
    #print(data.shape)
    return data

def main():
    # Define the mean for Gaussian in Class1
    mean11 = np.matrix([[0.], [0.]])

    # Define the mean for each Gaussian in Class2
    mean21_83 = np.matrix([[1.931], [0.]])  
    mean21_86 = np.matrix([[2.17], [0.]])  

    # Define the covarience for Gaussian in Class1
    covariance11 = np.matrix([
    [1, 0],
    [0, 1]
    ])

    # Define the covarience for Gaussian in Class2
    covariance21 = np.matrix([
    [1, 0],
    [0, 1]
    ])

    n = 8000
    data = gaussian_data(n, mean11, mean21_86, covariance11, covariance21)

    # Multi Layer Perceptron (MLP) Classifier

    samples = data[:,[0,1]]  #try out the new distribution on classifier
    labels = data[:,2]
    
    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.20, random_state=5)
    #print(X_train.shape)
    #print(X_test.shape)

    # Build the Classifier Model and fit the model to the training data
    mlp = MLPClassifier(hidden_layer_sizes=(2,4,4,1), activation='relu', solver='adam',                  max_iter=2000,learning_rate_init=0.001,learning_rate="constant",random_state=4,shuffle=True,batch_size=8)
    #mlp.fit(X_train,y_train)
    mlp.fit(samples,labels)

    predict_train = mlp.predict(samples)

    #predict_train = mlp.predict(X_train)
    #predict_test = mlp.predict(X_test)

    #Evaluate the model - Training Performance
    #---------------------------------------------------------------------
    #from sklearn.metrics import classification_report,confusion_matrix

    #print the confusion matrix and the confusion report results on the train data
    #print(confusion_matrix(y_train,predict_train))
    #print(classification_report(y_train,predict_train))

    #tn, fp, fn, tp = confusion_matrix(y_train,predict_train).ravel()
    #specificity = tn / (tn+fp)

    #AUC = roc_auc_score(y_train,predict_train)
    #print("Train: Specificity:", specificity)
    #print("Train: AUC:", AUC)
    #print(predict_train.shape)

    #Evaluate the model - Training Performance
    #---------------------------------------------------------------------
    from sklearn.metrics import classification_report,confusion_matrix

    #print the confusion matrix and the confusion report results on the train data
    print(confusion_matrix(labels,predict_train))
    print(classification_report(labels,predict_train))

    tn, fp, fn, tp = confusion_matrix(labels,predict_train).ravel()
    specificity = tn / (tn+fp)

    AUC = roc_auc_score(labels,predict_train)
    print("Train: Specificity:", specificity)
    print("Train: AUC:", AUC)
    #print(predict_train.shape)
    
    
    #os.chdir("/home/smriti.prathapan/note1/data-wo-multiproc")
    #GaussianSamples86 = np.load('Gaussian86-3000days1000Sim.npy')     #in-control  
    #GaussianSamples83 = np.load('Gaussian83-2000days1000Sim.npy')     #out-of-control
    os.chdir("/home/smriti.prathapan/note1")
    GaussianSamples86 = np.load('Gaussian86-150M.npy')     #in-control  
    GaussianSamples83 = np.load('Gaussian83-150M.npy')     #out-of-control

    # Simulate 2000 days - Samples from AUC(0.86) from day 0-999 and AUC(0.83) from day1000-1999
    runs = 0
    FalsePos         =  np.array([])
    TruePos          =  np.array([])
    DelaytoDetect    =  np.array([])
    FAR              =  np.array([])     #False Alarm Rate
    inSTD_test_AUCs  =  np.array([])     #Standard deviation of test AUCs
    outSTD_test_AUCs =  np.array([]) 
    Displacement     =  np.array([])     #Displacement (shift in mean)
    #DetectionTimes=  np.array([])
    DetectionTimes   =  np.array([],dtype=int) #save the FP detection times
    Dj                =  np.array([],dtype=int) #save the Dj which are binary values indicating detection
    Zj                =  np.array([],dtype=int) #save the Zj = min(Tj,pre-change-days)
    AvgDD            = np.array([])       # Average Detection Delay
    #Save the in-control and out-of-control specificities - mean AUCs here
    sp_pre           = np.array([])
    sp_post          = np.array([])
    pre_change_days  = 1000
    post_change_days = 1000
    total_days       = pre_change_days + post_change_days
    sample_size      = 50
    start_86         = 0
    end_86           = 1
    start_83         = 0
    end_83           = 1
    delta            = 1.5 #0.616  #0.1481 #0.318 try 1, 2 and 3

    while (runs < 1000):   #was 8 from the 10S 1000 D data
        test_days   = 0
        test_AUC    =  np.array([])
        #start_86    = 0
        #end_86      = 1
        #start_83    = 0
        #end_83      = 1
        #print("Sim:",runs)
        while (test_days < pre_change_days):     #day0-99 from AUC(0.86) June2->changed to 0-999 of AUC0.86
            test_samples = np.array([])
            test_labels  = np.array([])
            
            #Draw n samples per day-------------------------------------------------------------
            #n = 25
            #data = gaussian_data(n, mean11, mean21_86, covariance11, covariance21) # prallelize, 
            #-----------removed the above lines since it adds a lot of delay--------------------
            start  = start_86 * sample_size
            end    = end_86 * sample_size
            data86 = GaussianSamples86[start:end,:]
            #print("86:Start and end", start, end)
            #print("Sim: day:",runs, test_days)
            
            #Separate the test samples and labels
            test_samples = data86[:,[0,1]]
            test_labels  = data86[:,2]
            
            #Test the Classifier using the new samples and append the AUCs for Changepoint Detection
            predict_test   = mlp.predict(test_samples)
            
            tn, fp, fn, tp = confusion_matrix(test_labels,predict_test).ravel()
            specificity    = tn / (tn+fp)
            AUC            = roc_auc_score(test_labels,predict_test)
            sp_pre         = np.append(sp_pre, AUC)    #Combine for histogram
            test_AUC       = np.append(test_AUC, AUC)
            test_days     += 1
            start_86      += 1
            end_86        += 1
        
        while (test_days < total_days):    #day100-199 from AUC(0.83)  June2->changed to 1000-1999 of AUC0.83
            test_samples83 = np.array([])
            test_labels83 = np.array([])
            
            #Draw n samples per day------------------------------------------------
            #n = 25
            #data = gaussian_data(n, mean11, mean21_83, covariance11, covariance21)
            #----------removed the above lines since it adds a lot of delay--------
            start  = start_83 * sample_size
            end    = end_83 * sample_size
            data83 = GaussianSamples83[start:end,:]
            #print("83:Start and end", start, end)
            #print("Sim: day:",runs, test_days)
            
            #Separate the test samples and labels
            test_samples83 = data83[:,[0,1]]
            test_labels83  = data83[:,2]
            
            #Test the Classifier using the new samples and append the AUCs for Changepoint Detection
            predict_test83         = mlp.predict(test_samples83)
            
            tn83, fp83, fn83, tp83 = confusion_matrix(test_labels83,predict_test83).ravel()
            specificity83          = tn83 / (tn83+fp83)
            AUC83                  = roc_auc_score(test_labels83,predict_test83)
            sp_post                = np.append(sp_post, AUC83)      #Combine for histogram
            test_AUC               = np.append(test_AUC, AUC83)
            test_days             += 1
            start_83              += 1
            end_83                += 1
            
        #CUSUM for day0-2000: outcomes are detection delay and #FP, #TP, MTBFA, False alarm rate
        num_rows        = np.shape(test_AUC)[0]
        in_control_auc  = test_AUC[:pre_change_days]
        out_control_auc = test_AUC[pre_change_days:total_days]
        out_std_auc     = np.std(out_control_auc)
        in_std_auc      = np.std(in_control_auc)
        x               = np.array(test_AUC)

        mu     = np.mean(in_control_auc)
        mu_1   = np.mean(out_control_auc)
        std    = np.std(in_control_auc)
        std_1  = np.std(out_control_auc)
        d      = np.abs((mu_1-mu)/std)

        #h      = 0.25     # Upper/lower control limit to detect the changepoint
        #k      = 0.03    # Drift 0.05 is the 2 sigma change that we wish to detect, 0.025 - one-sigma change
        #delta  = 2
        h      = in_std_auc * 4
        k      = 0.03 
        #k      = (delta * in_std_auc)/2

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

        df = pd.DataFrame(chart) 
        df.columns = ['X','x-mu','Increase in Mean', 'S_hi', 'Decrease-in-mean', 'S_lo', 'CUSUM']
        #filename = "file%d" %runs
        #df.to_csv(("CUSUM-out/file%d.csv" %runs), sep='\t')    
        #print(df.to_string())
        #print(chart)
        #Export datafrae to png
        #import dataframe_image as dfi
        #dfi.export(df,'CUSUM-out/CUSUM-run.png')
        
        
        # False positives and Total alarms
        falsePos = 0
        alarms   = 0
        delay    = 0
        avddd    = 0   # this is the delay from the paper: td-ts (z_k-v) where v is the changepoint and z_k is the time of detection
        #MTBFA    = 0
        #D        =  np.array([],dtype=int) #save the Dj which are binary values indicating detection
        #Z        =  np.array([],dtype=int) #save the Zj = min(Tj,pre-change-days) min(i, S_hi[i-1] + mean_hi[i])
        
        for i in range(0, pre_change_days):
            if (S_lo[i] > h or S_hi[i] > h and i<pre_change_days):   
                falsePos += 1  #False Positives 
                DetectionTimes= np.append(DetectionTimes, i) 
                Dj = np.append(Dj, 1)
                Zj = np.append(Zj, min(i,pre_change_days))
                #print("time false alarm",i)
                #DetectionTimes= np.append(DetectionTimes, i)   #time at which a false positive is detected
                #print("detection times",DetectionTimes)
                #print("detection times size",DetectionTimes.size)
                break
            #if (S_lo[i] > h):   
                #if (i>100):  
                    #alarms += 1        #True Positive: break after detecting one TP
                    #delay   = i-100+1    # ts is 100 because the change starts at day100
                    #avddd   = i-100
                #break
        if falsePos == 0:
            Dj = np.append(Dj, 0)
            #DetectionTimes[runs] = pre_change_days
            Zj = np.append(Zj, pre_change_days)
        # Delay to detect the first changepoint
        #delay = 0
        for i in range(pre_change_days, total_days):
            if ((S_lo[i] > h) or (S_hi[i] > h)):
                alarms += 1        #True Positive: break after detecting one TP
                #delay  = i-1000+1    # ts is 100 because the change starts at day100
                avddd  = i-pre_change_days
                break
        
        #Calculate MTBFA(Mean time time between False Alarms)
        #MTBFA = np.mean(DetectionTimes)
        #FlaseAlarmRate = 1/MTBFA
        
        FalsePos      = np.append(FalsePos, falsePos)
        TruePos       = np.append(TruePos, alarms)
        #DelaytoDetect = np.append(DelaytoDetect, delay)   # td-ts+1
        #FAR           = np.append(FAR, FlaseAlarmRate)
        #DetectionTimes= np.append(DetectionTimes, detectionTime)
        AvgDD         = np.append(AvgDD, avddd)   # ADD estimate from the paper
        outSTD_test_AUCs = np.append(outSTD_test_AUCs, out_std_auc)
        inSTD_test_AUCs  = np.append(inSTD_test_AUCs, in_std_auc)
        Displacement     = np.append(Displacement, d)
        #print(falsePos)
        runs += 1  # continue until 1000 runs 
    print("--------------------------------")
    print("Control Limit:\t", h/np.mean(inSTD_test_AUCs))
    print("Norm.Reference Value:\t", k/np.mean(inSTD_test_AUCs))
    print("Pre/Post Change Days:\t", pre_change_days)
    print("Samples per day:\t", sample_size)
    print("--------------------------------")
    print("total number of False Positives:",np.sum(FalsePos))
    print("Total True Positives:",np.sum(TruePos))
    print("Total False Negatives:",runs-np.sum(TruePos))
    print("Average Detection Delay",np.mean(AvgDD))
    print("Minimum Delay",np.min(AvgDD))
    print("Maximum Delay",np.max(AvgDD))
    MTBFA = np.mean(DetectionTimes)
    MLP = np.sum(Dj)/np.sum(Zj)
    MTBFA_new = 1/MLP
    FlaseAlarmRate = 1/MTBFA
    print("MTBFA", MTBFA)
    print("MTBFA new", MTBFA_new)
    print("Flase Alarm Rate", FlaseAlarmRate)
    print("Mean std of in-control AUCs:",np.mean(inSTD_test_AUCs))
    print("Mean Displacement:", np.mean(Displacement))
    nonZeroAvgDD = AvgDD[np.nonzero(AvgDD)]
    #print("Z", Z)
    #print("D", D)
    #print("Average Detection Delay, ADD from paper is",np.mean(nonZeroAvgDD))
    #print("standard deviations of in-control AUCs:",np.mean(inSTD_test_AUCs))
    #print("standard deviations of out-of-control AUCs:",outSTD_test_AUCs)

if __name__ == "__main__":
    main()

