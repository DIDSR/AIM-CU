
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
import sys
sys.path.append("/home/smriti.prathapan/note1")

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
    #return data
    return Y_l0, Y_l1

def createData(iter):
    # Define the mean1 for Gaussian in Class1
    mean11 = np.matrix([[0.], [0.]])

    # Define the mean2 for each Gaussian in Class2
    mean21_83 = np.matrix([[1.35], [0.]])
    mean21_86 = np.matrix([[1.553], [0.]])


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

    #Define the number of days and sample size (n)
    pre_change_days=10000
    n = 25
    
    #Empty array to append the data
    #GaussianSamples86 = np.empty((0,3))
    GaussianSamples83 = np.empty((0,3))

    #Create data for 1000 pre-change days - this takes about 5 minutes
    for i in range(0, pre_change_days):
        #Y_l0, Y_l1 = gaussian_data(n, mean11, mean21_86, covariance11, covariance21)
        Y_l0, Y_l1 = gaussian_data(n, mean11, mean21_83, covariance11, covariance21)
        #print ("i:",i)
        data = np.concatenate((Y_l0,Y_l1))
        GaussianSamples83 = np.append(GaussianSamples83, np.array(data), axis=0) 
    
    #Save the data
    print ("Saving data for 1000 post-change days, 10Simulations, iteration#:",iter)
    np.save(('/home/smriti.prathapan/note1/GaussData-March0724/D83-Mar10/%d-Gaussian83-10S-1000D.npy' %iter), GaussianSamples83) # save

if __name__ == "__main__":
    #main()
    print("Number of arguments:", len(sys.argv))
    print("Argument List:", str(sys.argv))

    #control_limit, delta, change_days, samples_per_day = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    #cusum_detection (control_limit,delta,change_days,samples_per_day)
    num_iteration = float(sys.argv[1])
    createData(num_iteration)


