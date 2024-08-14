"""
CUSUM

@author: smriti.prathapan
"""

import numpy as np
import random
import pandas as pd
import warnings

# import matplotlib.pyplot as plt
# from matplotlib import rcParams
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
        self.df = None
        self.df_wavel7 = None

        self.AvgDD = None
        self.h_1000 = None
        self.k_1000 = None
        self.data = None
        self.sp_pre = None
        self.sp_post = None

        self.h = None
        self.in_std = None
        self.S_hi = None
        self.S_lo = None

        self.config = None

    def initialize(self):
        with open("config.toml", "rb") as file_config:
            self.config = tomli.load(file_config)

        self.df = pd.read_csv(self.config["path_input"]["path_df"])
        self.df_wavel7 = pd.read_csv(self.config["path_input"]["path_df_wavel7"])

    # The function displays dataframe size, countings of unique patients and unique exams
    def stats(self):
        print("Dataframe size: " + str(self.df.shape))
        try:
            print("# patients: " + str(self.df.patient_id.nunique()))
        except:
            print("# patients: " + str(self.df.patient_id.nunique()))
        print("# exams: " + str(self.df.acc_anon.nunique()))

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

        # Construct the tabular CUSUM Chart
        chart = np.array([])
        chart = np.column_stack(
            (x.T, x_mean.T, mean_hi.T, self.S_hi.T, mean_lo.T, self.S_lo.T, cusum.T)
        )
        np.round(chart, 2)

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

        df_out = pd.DataFrame(chart)
        df_out.columns = [
            "X",
            "x-mu_0",
            "Increase in Mean",
            "S_hi",
            "Decrease-in-mean",
            "S_lo",
            "CUSUM",
        ]
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
        # Fetch 100 patients per day (day1-60) from df and days 61-120 from df_medk10 which is the out of control region
        # compute
        sample_size = 40
        FalsePos = np.array([])
        TruePos = np.array([])
        DelaytoDetect = np.array([])
        FAR = np.array([])  # False Alarm Rate
        inSTD_test_sp = np.array([])  # Standard deviation of test AUCs
        outSTD_test_sp = np.array([])
        D = np.array([])  # Displacement
        self.h_1000 = np.array([])
        self.k_1000 = np.array([])
        DetectionTimes = np.array([], dtype=int)
        Dj = np.array(
            [], dtype=int
        )  # save the Dj which are binary values indicating detection MTBFA
        Zj = np.array([], dtype=int)  # save the Zj = min(Tj,pre-change-days)-MTBFA
        zj = np.array([], dtype=int)  # ADD - MLE of delays
        cj = np.array([], dtype=int)  # ADD - binary
        self.AvgDD = np.array([])  # Average Detection Delay
        start_in = 0
        end_in = sample_size
        start_out = 0
        end_out = sample_size
        # sample_size      = 30
        # days             = 0
        pre_change_days = 60
        post_change_days = 60
        total_days = pre_change_days + post_change_days
        patients_in = self.df.patient_id.unique()
        patients_o = self.df_wavel7.patient_id.unique()
        self.sp_pre = np.array([])
        self.sp_post = np.array([])
        runs = 0
        # delta            = 1 #0.616  #0.1481 #0.318
        ref_val = 0.5
        control_limit = 4
        while runs < 10:  # 1000
            days = 0
            start_in = 0
            end_in = sample_size
            start_out = 0
            end_out = sample_size
            # specificity = np.array([])
            self.data = np.array([])
            while days < pre_change_days:
                patients100 = patients_in[start_in:end_in]

                # Fetch all the rows for 100 patients
                p100 = self.df[self.df["patient_id"].isin(patients100)]

                # print("Checking stats for 100 patients")
                # stats(p100)

                # threshold = 0.31
                threshold_prechange = 0.0177
                FP = p100[p100["preds"] > threshold_prechange]
                TN = p100[p100["preds"] < threshold_prechange]

                # print("Total rows:",      p100.index.size)
                # print("#Below Threshold", TN.index.size)
                # print("#Above Threshold", FP.index.size)
                sp = TN.index.size / p100.index.size
                self.data = np.append(self.data, sp)
                self.sp_pre = np.append(self.sp_pre, sp)

                start_in += sample_size
                end_in += sample_size
                days += 1

            while days < total_days:
                patients100_out = patients_o[start_out:end_out]

                # Fetch all the rows for 100 patients
                p100_out = self.df_wavel7[
                    self.df_wavel7["patient_id"].isin(patients100_out)
                ]

                # print("Checking stats for 100 patients")
                # stats(p100)

                # threshold = 0.31
                # threshold_postchange = 0.0333
                # threshold_postchange = 0.03545
                threshold_postchange = 0.0177
                FP_o = p100_out[p100_out["preds"] > threshold_postchange]
                TN_o = p100_out[p100_out["preds"] < threshold_postchange]

                # print("Total rows:",      p100.index.size)
                # print("#Below Threshold", TN.index.size)
                # print("#Above Threshold", FP.index.size)
                sp_o = TN_o.index.size / p100_out.index.size
                self.data = np.append(self.data, sp_o)  # data = specificity
                self.sp_post = np.append(self.sp_post, sp_o)

                start_out += sample_size
                end_out += sample_size
                days += 1

            # CUSUM for day0-60: outcomes are detection delay and #FP, #TP, MTBFA, False alarm rate
            num_rows = np.shape(self.data)[0]
            in_control_data = self.data[:pre_change_days]
            out_control_data = self.data[pre_change_days:total_days]
            out_std = np.std(out_control_data)
            self.in_std = np.std(in_control_data)
            x = np.array(self.data)

            mu_0 = np.mean(in_control_data)
            mu_1 = np.mean(out_control_data)
            d = np.abs((mu_1 - mu_0) / self.in_std)

            # h      = 0.102       # Upper/lower control limit to detect the changepoint H=0.102, 0.127
            # k      = 0.03831     # Drift 0.01277 is the 1 sigma change, 0.0255 - one-sigma change, 0.03831 is 3-sigma change, 0.05108
            self.h = control_limit * self.in_std
            k = ref_val * self.in_std

            # Call compute CUSUM function with x (observatoins), in-control mean (mu) and k (drift or reference value)
            self.S_hi, self.S_lo, cusum = self.compute_cusum(x, mu_0, k)

            # False positives and Total alarms
            falsePos = 0
            alarms = 0
            delay = 0
            avddd = 0  # this is the delay from the paper: td-ts (z_k-v) where v is the changepoint and z_k is the time of detection
            # MTBFA    = 0

            for i in range(0, pre_change_days):
                if (self.S_hi[i] > self.h) or (self.S_lo[i] > self.h):
                    # if (i<pre_change_days):
                    falsePos += 1  # False Positives
                    # print("time false alarm",i)
                    DetectionTimes = np.append(
                        DetectionTimes, i + 1
                    )  # time at which a false positive is detected
                    Dj = np.append(Dj, 1)
                    Zj = np.append(Zj, min(i, pre_change_days))
                    # print("detection times",DetectionTimes)
                    # print("detection times size",DetectionTimes.size)
                    break

            # If there is no false positive, Zj = pre_change_days, Dj = 0
            if falsePos == 0:
                Dj = np.append(Dj, 0)
                # DetectionTimes[runs] = pre_change_days
                Zj = np.append(Zj, pre_change_days)

            # Delay to detect the first changepoint
            # delay = 0
            for i in range(pre_change_days, total_days):
                if (self.S_hi[i] > self.h) or (self.S_lo[i] > self.h):
                    alarms += 1  # True Positive: break after detecting one TP
                    # print("alarm at : ", i)
                    # delay  = i-1000+1    # ts is 100 because the change starts at day100
                    avddd = i - pre_change_days
                    cj = np.append(cj, 1)
                    zj = np.append(zj, min(avddd, total_days))
                    break

            # If there is no true detection, zj = total simulation days, cj = 0
            if alarms == 0:
                cj = np.append(cj, 0)
                # DetectionTimes[runs] = pre_change_days
                zj = np.append(zj, total_days)

            # Calculate MTBFA(Mean time time between False Alarms)
            # MTBFA = np.mean(DetectionTimes)
            # FlaseAlarmRate = 1/MTBFA

            FalsePos = np.append(FalsePos, falsePos)
            TruePos = np.append(TruePos, alarms)
            # DelaytoDetect = np.append(DelaytoDetect, delay)   # td-ts+1
            # FAR           = np.append(FAR, FlaseAlarmRate)
            # DetectionTimes= np.append(DetectionTimes, detectionTime)
            self.AvgDD = np.append(self.AvgDD, avddd)  # ADD estimate from the paper
            outSTD_test_sp = np.append(outSTD_test_sp, out_std)
            inSTD_test_sp = np.append(inSTD_test_sp, self.in_std)
            D = np.append(D, d)
            self.h_1000 = np.append(self.h_1000, self.h)
            self.k_1000 = np.append(self.k_1000, k)
            # print(falsePos)

            # Shuffle the patient list for the next simulation
            random.shuffle(patients_in)
            random.shuffle(patients_o)
            runs += 1  # continue until end of simulation

        # specificity
        print("H is ", self.h)
        print("Reference Value is", k)
        print("--------------------------------")
        print("Control Limit:\t", control_limit)
        print("Reference Value:\t", ref_val)
        print("Pre/Post Change Days:\t", pre_change_days)
        print("Samples per day:\t", sample_size)
        print("--------------------------------")
        print("total number of False Positives:", np.sum(FalsePos))
        print("Total True Positives:", np.sum(TruePos))
        print("Total False Negatives:", runs - np.sum(TruePos))
        print("Average Detection Delay", np.mean(self.AvgDD))
        print("Average Detection Delay NEW:", np.sum(zj) / np.sum(cj))
        print("Minimum Delay", np.min(self.AvgDD))
        print("Maximum Delay", np.max(self.AvgDD))
        MTBFA = np.mean(DetectionTimes)
        MLP = np.sum(Dj) / np.sum(Zj)
        MTBFA_new = 1 / MLP
        FlaseAlarmRate = 1 / MTBFA
        print("MTBFA", MTBFA)
        print("MTBFA new", MTBFA_new)
        print("Flase Alarm Rate", FlaseAlarmRate)
        nonZeroAvgDD = self.AvgDD[np.nonzero(self.AvgDD)]
        print("Mean ref. Value", np.mean(self.k_1000))
        print("Mean std of in-control data:", np.mean(inSTD_test_sp))
        print("Mean out-of-control data:", np.mean(outSTD_test_sp))
        # print ("mu_0", mu)
        # print ("mu_1", mu_1)
        # print ("std_0", std)
        # print("Displacement, d:",(mu_1-mu)/std)
        print("Mean Displacement:", np.mean(D))

    # histogram using matplotlib
    def plot_histogram_mpl(self, data, xlabel, title=""):
        fig, ax = plt.subplots(figsize=(10, 6))
        rcParams["font.weight"] = "bold"
        count, bins, ignored = plt.hist(data, 30, color="limegreen", alpha=0.5)
        plt.rcParams["axes.facecolor"] = "white"
        plt.grid(visible=None)
        ax.set_xlabel(xlabel, fontsize=18, fontweight="bold")
        ax.set_ylabel("Count", fontsize=18, fontweight="bold")
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.legend(fontsize=16, frameon=False)
        plt.title(title, fontsize=14)
        plt.show()

    # histogram using plotly
    def plot_histogram_plotly(self, data, xlabel, title=""):
        fig = go.Figure(data=[go.Histogram(x=data, nbinsx=30)])
        fig.update_layout(title="[TITLE=?]", xaxis_title=xlabel, yaxis_title="Count")
        fig.update_layout(plot_bgcolor=self.config["color"]["blue_005"])
        # fig.show()

        return fig

    # Plot the input AUCs
    def plot_input_aucs(self):
        fig, ax = plt.subplots(figsize=(13, 6))
        rcParams["font.weight"] = "bold"

        pre_change_days = 60
        post_change_days = 60
        total_days = pre_change_days + post_change_days
        plt.rcParams["axes.facecolor"] = "white"
        plt.grid(b=None)
        y1 = np.arange(pre_change_days)
        x1 = self.data[:pre_change_days]
        plt.scatter(
            y1, x1, c="mediumturquoise", s=100, alpha=0.4, label="In-control $S_p$"
        )
        y2 = np.arange(pre_change_days, total_days, 1)
        x2 = self.data[pre_change_days:total_days]
        plt.scatter(y2, x2, c="coral", s=100, alpha=0.4, label="Out-of-control $S_p$")
        plt.hlines(
            y=np.mean(x1),
            xmin=0,
            xmax=pre_change_days,
            color="darkturquoise",
            alpha=0.9,
            linewidth=4,
            linestyle="--",
            label="In-control mean",
        )
        plt.hlines(
            y=np.mean(x2),
            xmin=pre_change_days,
            xmax=total_days,
            color="coral",
            alpha=0.9,
            linewidth=4,
            linestyle="--",
            label="Out-of-control mean",
        )
        # single vline with specific ymin and ymax
        plt.vlines(
            x=pre_change_days,
            ymin=np.min(self.data),
            ymax=np.max(self.data),
            colors="grey",
            linestyle="--",
            label="Change-point",
        )

        # yline(0.86)
        # plt.plot(test_AUC, 'go')

        ax.legend(fontsize=14)
        rcParams["legend.loc"] = "lower left"
        plt.rcParams["axes.facecolor"] = "white"
        # plt.title('Samples drawn from two Gaussians')
        ax.set_xlabel("Length of Simulation (days)", fontsize=18, fontweight="bold")
        ax.set_ylabel("AI model Specificity", fontsize=18, fontweight="bold")
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(visible=None)
        plt.show()

    # Plot the input AUCs
    def plot_input_aucs_plotly(self):
        pre_change_days = 60
        post_change_days = 60
        total_days = pre_change_days + post_change_days

        # [! NOTE: x and y are interchanged from the original code]
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
                name="In-control $S_p$",
                marker=dict(color="darkturquoise", size=10),
                opacity=0.4,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x2,
                y=y2,
                mode="markers",
                name="Out-of-control $S_p$",
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
            title="[TITLE=?]",
            xaxis_title="Length of Simulation (days)",
            yaxis_title="AI model Specificity",
            xaxis=dict(dtick=20),
        )

        fig.update_layout(plot_bgcolor=self.config["color"]["blue_005"])

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # fig.show()

        return fig

    # PLOT THE HISTOGRAM OF all AUCs - all the AUCs for 1000 simulations * 1000 days
    def plot_histogram_aucs(self):
        rcParams["font.weight"] = "bold"
        fig, ax = plt.subplots(figsize=(10, 6))
        locs = ["upper right", "lower right", "center right"]
        count, bins, ignored = plt.hist(
            self.sp_pre,
            30,
            color="mediumturquoise",
            alpha=0.5,
            label="Pre-change $S_p$",
        )
        plt.hist(self.sp_post, 30, color="coral", alpha=0.5, label="Post-change $S_p$")
        # plt.plot(bins, 1/(std * np.sqrt(2 * np.pi)) *
        #               np.exp( - (bins - mu)**2 / (2 * std**2) ),
        #         linewidth=2, color='r')
        plt.rcParams["axes.facecolor"] = "white"
        plt.grid(visible=None)
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.axvline(
            x=np.mean(self.sp_pre), color="grey", linestyle="--", label="Reference mean"
        )
        ax.set_xlabel("AI model Specificity", fontsize=16, fontweight="bold")
        ax.set_ylabel("Count", fontsize=16, fontweight="bold")
        ax.legend(loc=2, fontsize=12, frameon=False)
        # plt.xticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
        plt.ticklabel_format(style="sci", axis="y", scilimits=(3, 3))
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.show()

    # PLOT THE HISTOGRAM OF all AUCs - all the AUCs for 1000 simulations * 1000 days
    def plot_histogram_aucs_plotly(self):
        fig = go.Figure()

        # add subplots
        fig.add_trace(
            go.Histogram(
                x=self.sp_pre,
                nbinsx=30,
                name="Pre-change $S_p$",
                marker=dict(color="mediumturquoise"),
                opacity=0.5,
            )
        )

        fig.add_trace(
            go.Histogram(
                x=self.sp_post,
                nbinsx=30,
                name="Post-change $S_p$",
                marker=dict(color="coral"),
                opacity=0.5,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[np.mean(self.sp_pre), np.mean(self.sp_pre)],
                y=[0, 100],  # [! y_max is not working]
                mode="lines",
                name="Reference mean",
                line=dict(color="grey", dash="dash"),
            )
        )

        fig.add_vline(x=np.mean(self.sp_pre), line_dash="dash", line_color="grey")

        fig.update_layout(
            title="[TITLE=?]",
            xaxis_title="AI model Specificity",
            yaxis_title="Count",
            xaxis=dict(dtick=0.2, range=[0, 1]),
        )

        fig.update_layout(plot_bgcolor=self.config["color"]["blue_005"])

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # fig.show()

        return fig

    # plot CUSUM value
    def plot_cusum(self):
        # specifying horizontal line type
        fig, ax = plt.subplots(figsize=(12, 6))
        rcParams["font.weight"] = "bold"
        # plt.yticks([0, 5, 10, 15, 20, 25])
        plt.axhline(
            y=self.h / self.in_std,
            color="firebrick",
            linestyle="--",
            linewidth=2.0,
            label="Threshold (H)",
        )
        ax.set_xlabel("Length of Simulation (Days)", fontsize=16, fontweight="bold")
        ax.set_ylabel("CUSUM value", fontsize=16, fontweight="bold")
        plt.plot(
            self.S_hi / self.in_std,
            "greenyellow",
            label="Positive changes ($S_{hi}$)",
            linewidth=2.0,
        )
        plt.plot(
            self.S_lo / self.in_std,
            "darkcyan",
            label="Negative changes ($S_{lo}$)",
            linewidth=2.0,
        )
        ax.axvline(
            x=60, color="grey", linestyle="--", label="Change-point", linewidth=2.0
        )
        # plt.title('Positive and Negative Changes')
        # plt.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45,50, 55, 60])
        # plt.yticks([0, 5, 10, 15, 20, 25])
        # plt.yticks([0, 10, 20, 30, 40, 50, 60])
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)
        plt.legend(fontsize=13, frameon=False)
        rcParams["legend.loc"] = "upper left"
        plt.show()

    # plot CUSUM value
    def plot_cusum_plotly(self):
        fig = go.Figure()

        # add subplots
        fig.add_trace(
            go.Scatter(
                x=list(range(len(self.S_hi))),
                y=self.S_hi / self.in_std,
                mode="lines",
                name="Positive changes ($S_{hi}$)",
                marker=dict(color="greenyellow", size=10),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(self.S_lo))),
                y=self.S_lo / self.in_std,
                mode="lines",
                name="Negative changes ($S_{lo}$)",
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
            title="[TITLE=?]",
            xaxis_title="Length of Simulation (Days)",
            yaxis_title="CUSUM value",
            xaxis=dict(dtick=20),
        )

        fig.update_layout(plot_bgcolor=self.config["color"]["blue_005"])

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # fig.show()

        return fig


# obj_cusum = CUSUM()
# obj_cusum.initialize()
# obj_cusum.stats()
# obj_cusum.change_detection()
# obj_cusum.plot_histogram_mpl(obj_cusum.AvgDD, "ADD")
# obj_cusum.plot_histogram_plotly(obj_cusum.AvgDD, "ADD")
# obj_cusum.plot_histogram_mpl(obj_cusum.h_1000, "H")
# obj_cusum.plot_histogram_plotly(obj_cusum.h_1000, "H")
# obj_cusum.plot_histogram_mpl(obj_cusum.k_1000, "K")
# obj_cusum.plot_histogram_plotly(obj_cusum.k_1000, "K")
# obj_cusum.plot_input_aucs()
# obj_cusum.plot_input_aucs_plotly()
# obj_cusum.plot_histogram_aucs()
# obj_cusum.plot_histogram_aucs_plotly()
# obj_cusum.plot_cusum()
# obj_cusum.plot_cusum_plotly()


# code to add later

# # Save the pre- and pos-change Specificities into dataframe - 60 days of pre- and post-change
# preChange_AUC_1000S_1000D = pd.DataFrame(data)
