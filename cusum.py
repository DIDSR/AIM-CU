"""
CUSUM

@author: smriti.prathapan
"""

import numpy as np
import random
import pandas as pd
import warnings

import plotly.graph_objects as go
import tomli

warnings.filterwarnings("ignore")
random.seed(58)


# CUSUM class and its functionalities [? modify the comment]
class CUSUM:
    """
    [? add docstring for class]
    """

    def __init__(self):  # [? add the required input parameters]
        # [? add required variables here]
        self.df_sp = None

        self.AvgDD = None
        self.h_1000 = None
        self.k_1000 = None
        self.data = None

        self.h = None
        self.in_std = None
        self.S_hi = None
        self.S_lo = None

        self.config = None

    def initialize(self):
        with open("config.toml", "rb") as file_config:
            self.config = tomli.load(file_config)

        self.df_sp = pd.read_csv(self.config["path_input"]["path_df_sp"])
        # AUCs to numpy array
        self.data  = self.df_sp[self.df_sp.columns[1]].to_numpy()

    # Compute CUSUM for the observations in x
    def compute_cusum(self, x, mu_0, k):
        # CUSUM for day0-2000: outcomes are detection delay and #FP, #TP, MTBFA, False alarm rate
        num_rows = np.shape(x)[0]

        x_mean = np.zeros(num_rows, dtype=float)
        # S_hi : for positive changes --------------------------
        self.S_hi = np.zeros(num_rows, dtype=float)
        self.S_hi[0] = 0.0  # starts with 0
        # Increase in mean = x-mu-k ----------------------------
        mean_hi = np.zeros(num_rows, dtype=float)

        # Decrease in mean = mu-k-x----------------------------
        mean_lo = np.zeros(num_rows, dtype=float)
        # S_lo : for negative changes --------------------------
        self.S_lo = np.zeros(num_rows, dtype=float)
        self.S_lo[0] = 0.0  # starts with 0
        # CUSUM: Cumulative sum of x minus mu ------------------
        cusum = np.zeros(num_rows, dtype=float)
        cusum[0] = 0.0  # initialize with 0

        for i in range(0, num_rows):
            x_mean[i] = x[i] - mu_0  # x - mean
            mean_hi[i] = x[i] - mu_0 - k
            self.S_hi[i] = max(0, self.S_hi[i - 1] + mean_hi[i])
            mean_lo[i] = mu_0 - k - x[i]
            self.S_lo[i] = max(0, self.S_lo[i - 1] + mean_lo[i])
            cusum[i] = cusum[i - 1] + x_mean[i]

        x_mean = np.round(x_mean, decimals=2)
        self.S_hi = np.round(self.S_hi, decimals=2)
        mean_lo = np.round(mean_lo, decimals=2)
        self.S_lo = np.round(self.S_lo, decimals=2)
        cusum = np.round(cusum, decimals=2)

        # # Construct the tabular CUSUM Chart
        # chart = np.array([])
        # chart = np.column_stack(
        #     (x.T, x_mean.T, mean_hi.T, self.S_hi.T, mean_lo.T, self.S_lo.T, cusum.T)
        # )
        # np.round(chart, 2)

        # d = 2 *(np.log((1-0.01) / (0.0027)))
        # h = d * 0.5 # h= d*k where k=0.5
        # h = 4 # as per the NIST doc on CUSUM

        # l1 =  np.append(num_rows, data_tabular, axis = 1)
        # l1 = np.concatenate(num_rows.T, data_tabular.T)
        # chart = np.column_stack((num_rows.T, data_tabular.T))
        # chart

        np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})
        # print("CUSUM Chart is:\n", np.round(chart,decimals=2))
        # x_mean

        # df_out = pd.DataFrame(chart)
        # df_out.columns = [
        #     "X",
        #     "x-mu_0",
        #     "Increase in Mean",
        #     "S_hi",
        #     "Decrease-in-mean",
        #     "S_lo",
        #     "CUSUM",
        # ]
        # filename = "file%d" %runs
        # df_out.to_csv(("CUSUM-out/file%d.csv" %runs), sep='\t')
        # print(df.to_string())
        # print(chart)
        # Export datafrae to png
        # import dataframe_image as dfi
        # dfi.export(df,'CUSUM-out/CUSUM-run.png')

        return self.S_hi, self.S_lo, cusum

    # [! provide a proper name]
    def change_detection(self):
        pre_change_days  = 60
        post_change_days = 60
        total_days       = pre_change_days + post_change_days
        ref_val          = 0.5
        control_limit    = 4

        self.h_1000           =  np.array([]) 
        self.k_1000           =  np.array([])
        DetectionTimes   =  np.array([],dtype=int)
        Dj               =  np.array([],dtype=int) #save the Dj which are binary values indicating detection MTBFA
        Zj               =  np.array([],dtype=int) #save the Zj = min(Tj,pre-change-days)-MTBFA
        zj               =  np.array([],dtype=int) # ADD - MLE of delays
        cj               =  np.array([],dtype=int) # ADD - binary
        self.AvgDD            = np.array([])      # Average Detection Delay
        D                =  np.array([])     #Displacement
        FalsePos         =  np.array([])
        TruePos          =  np.array([])



        #CUSUM for day0-60: outcomes are detection delay and #FP, #TP, MTBFA, False alarm rate
        num_rows = np.shape(self.data)[0]
        in_control_data  = self.data[:pre_change_days]
        out_control_data = self.data[pre_change_days:total_days]
        out_std          = np.std(out_control_data)
        self.in_std           = np.std(in_control_data)
        x                = np.array(self.data)

        mu_0   = np.mean(in_control_data)
        mu_1   = np.mean(out_control_data)
        d      = np.abs((mu_1-mu_0)/self.in_std)
            
        #h      = 0.102       # Upper/lower control limit to detect the changepoint H=0.102, 0.127 
        #k      = 0.03831     # Drift 0.01277 is the 1 sigma change, 0.0255 - one-sigma change, 0.03831 is 3-sigma change, 0.05108
        self.h       = control_limit * self.in_std 
        k       = ref_val * self.in_std

        #Call compute CUSUM function with x (observatoins), in-control mean (mu) and k (drift or reference value)
        self.S_hi, self.S_lo, cusum = self.compute_cusum(x, mu_0, k)
        
        # False positives and Total alarms
        falsePos = 0
        alarms   = 0
        delay    = 0
        avddd    = 0   # this is the delay from the paper: td-ts (z_k-v) where v is the changepoint and z_k is the time of detection
        #MTBFA    = 0
            
        for i in range(0, pre_change_days):
            if ((self.S_hi[i] > self.h) or (self.S_lo[i] > self.h)):   
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
            if ((self.S_hi[i] > self.h) or (self.S_lo[i] > self.h)):
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
        self.AvgDD          = np.append(self.AvgDD, avddd)   # ADD estimate from the paper
        #outSTD_test_sp = np.append(outSTD_test_sp, out_std)
        #inSTD_test_sp  = np.append(inSTD_test_sp, in_std)
        D              = np.append(D, d)
        self.h_1000         = np.append(self.h_1000, self.h)
        self.k_1000         = np.append(self.k_1000, k)
        #print(falsePos)    


    # histogram using plotly
    def plot_histogram_plotly(self, data, xlabel, title=""):
        fig = go.Figure(data=[go.Histogram(x=data, nbinsx=30)])
        fig.update_layout(title="[TITLE=?]", xaxis_title=xlabel, yaxis_title="Count")
        fig.update_layout(plot_bgcolor=self.config["color"]["blue_005"])

        return fig

    # Plot the input Specificities using Plotly
    def plot_input_aucs_plotly(self):
        pre_change_days = 60
        post_change_days = 60
        total_days = pre_change_days + post_change_days

        x1 = np.arange(pre_change_days)
        y1 = self.data[:pre_change_days]
        mean_y1 = np.mean(y1)

        x2 = np.arange(pre_change_days, total_days, 1)
        y2 = self.data[pre_change_days:total_days]
        mean_y2 = np.mean(y2)

        fig = go.Figure()

        # add subplots
        fig.add_trace(
            go.Scatter(
                x=x1,
                y=y1,
                mode="markers",
                name=f"""In-control S<sub>p</sub>""",
                marker=dict(color="darkturquoise", size=10),
                opacity=0.4,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x2,
                y=y2,
                mode="markers",
                name=f"""Out-of-control S<sub>p</sub>""",
                marker=dict(color="coral", size=10),
                opacity=0.4,
            )
        )

        # add horizontal lines
        fig.add_trace(
            go.Scatter(
                x=[min(x1), max(x1)],
                y=[mean_y1, mean_y1],
                mode="lines",
                name="In-control mean",
                line=dict(color="darkturquoise", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[min(x2), max(x2)],
                y=[mean_y2, mean_y2],
                mode="lines",
                name="Out-of-control mean",
                line=dict(color="coral", dash="dash"),
            )
        )

        # add vertical line
        fig.add_trace(
            go.Scatter(
                x=[pre_change_days, pre_change_days],
                y=[np.min(self.data), np.max(self.data)],
                mode="lines",
                name="Change-point",
                line=dict(color="grey", dash="dash"),
            )
        )

        fig.update_layout(
            title="Average Specificities for the pre-change and post-change regime",
            xaxis_title="Length of Simulation (days)",
            yaxis_title="AI model Specificity",
            xaxis=dict(dtick=20),
        )

        fig.update_layout(plot_bgcolor=self.config["color"]["blue_005"])

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    # PLOT THE HISTOGRAM OF all Specificities - for the observations from 120 days
    def plot_histogram_aucs_plotly(self):
        # Display the histogram of in-control Sp

        fig = go.Figure()

        # add subplots
        fig.add_trace(
            go.Histogram(
                x=self.data[0:60],
                nbinsx=6,
                name=f"""Pre-change S<sub>p</sub>""",
                marker=dict(color="mediumturquoise"),
                opacity=0.5,
            )
        )

        fig.add_trace(
            go.Histogram(
                x=self.data[60:120],
                nbinsx=6,
                name=f"""Post-change SS<sub>p</sub>""",
                marker=dict(color="coral"),
                opacity=0.5,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[np.mean(self.data[0:60]), np.mean(self.data[0:60])],
                y=[0, 50],  # [! y_max is not working]
                mode="lines",
                name="Reference mean",
                line=dict(color="grey", dash="dash"),
            )
        )

        fig.add_vline(x=np.mean(self.data[0:60]), line_dash="dash", line_color="grey")

        fig.update_layout(
            title="Histograms for the pre-change and post-change specificity",
            xaxis_title="AI model Specificity",
            yaxis_title="Count",
            xaxis=dict(dtick=0.2, range=[0, 1]),
        )

        fig.update_layout(plot_bgcolor=self.config["color"]["blue_005"])

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    # plot CUSUM value using Plotly
    def plot_cusum_plotly(self):
        fig = go.Figure()

        # add subplots
        fig.add_trace(
            go.Scatter(
                x=list(range(len(self.S_hi))),
                y=self.S_hi / self.in_std,
                mode="lines",
                name=f"""Positive changes (S<sub>hi</sub>)""",
                marker=dict(color="greenyellow", size=10),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(self.S_lo))),
                y=self.S_lo / self.in_std,
                mode="lines",
                name=f"""Negative changes (S<sub>lo</sub>)""",
                marker=dict(color="darkcyan", size=10),
            )
        )

        # add horizontal lines
        fig.add_trace(
            go.Scatter(
                x=[0, len(self.S_lo)],
                y=[self.h / self.in_std, self.h / self.in_std],
                mode="lines",
                name="Threshold (H)",
                line=dict(color="firebrick", dash="dash"),
            )
        )

        # add vertical line
        fig.add_trace(
            go.Scatter(
                x=[60, 60],  # [! this is constant!]
                y=[0, np.max(self.S_lo / self.in_std)],
                mode="lines",
                name="Change-point",
                line=dict(color="grey", dash="dash"),
            )
        )

        fig.update_layout(
            title="CUSUM Chart",
            xaxis_title="Length of Simulation (Days)",
            yaxis_title="CUSUM value",
            xaxis=dict(dtick=20),
        )

        fig.update_layout(plot_bgcolor=self.config["color"]["blue_005"])

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig
