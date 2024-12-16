"""
Cumulative Sum (CUSUM)

@author: smriti.prathapan
"""

import os
import sys
import numpy as np
import random
import pandas as pd
import warnings

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tomli

warnings.filterwarnings("ignore")
random.seed(58)


class CUSUM:
    """
    CUSUM class and its functionalities.
    """

    def __init__(self):
        self.df_metric = None
        self.metric_type = None

        self.AvgDD = None
        self.data = None

        self.h = None
        self.in_std = None
        self.S_hi = None
        self.S_lo = None

        self.config = None

        self.total_days = None
        self.pre_change_days = None
        self.post_change_days = None

    def initialize(self) -> None:
        """
        Initialize with the configuration file.
        """
        try:
            with open(os.path.abspath("../../config/config.toml"), "rb") as file_config:
                self.config = tomli.load(file_config)
        except FileNotFoundError:
            print("Error: config.toml not found.")
            sys.exit(1)

    def set_timeline(self, data: np.ndarray) -> None:
        """
        Set the timeline of observations.

        Args:
            data (np.ndarray): Data of the metric values across the observations.
        """
        self.total_days = np.shape(data)[0]

    def set_df_metric_default(self) -> None:
        """
        Read the provided performance metric data to be used for CUSUM for an example.
        """
        try:
            self.df_metric = pd.read_csv(
                os.path.abspath(self.config["path_input"]["path_df_metric"])
            )
        except FileNotFoundError:
            print("Error: CSV file not found.")
            sys.exit(1)
        self.data = self.df_metric[self.df_metric.columns[1]].to_numpy()

        self.set_timeline(self.data)

    def set_df_metric_csv(self, data_csv: pd.DataFrame) -> None:
        """
        Assign the performance metric data to be used for CUSUM.

        Args:
            data_csv (DataFrame or TextFileReader): A comma-separated values (csv) file is returned as two-dimensional data structure with labeled axes.
        """
        self.df_metric = data_csv
        self.data = self.df_metric[self.df_metric.columns[1]].to_numpy()

        self.set_timeline(self.data)

    def compute_cusum(
        self, x: list[float], mu_0: float, k: float
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Compute CUSUM for the observations in x

        Args:
            x (list[float]): Performance metric
            mu_0 (float): In-control mean
            k (float): Reference value related to the magnitude of change that one is interested in detecting

        Returns:
            tuple[list[float], list[float], list[float]]: Positive cumulative sum, negative cumulative sum, and CUSUM
        """
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

        return self.S_hi, self.S_lo, cusum

    def change_detection(
        self,
        pre_change_days: int,
        normalized_ref_value: float = 0.5,
        normalized_threshold: float = 4,
    ) -> None:
        """
        Detects a change in the process.

        Args:
            pre_change_days (int): Number of days for in-control phase.
            normalized_ref_value (float, optional): Normalized reference value for detecting a unit standard deviation change in mean of the process. Defaults to 0.5.
            normalized_threshold (float, optional): Normalized threshold. Defaults to 4.
        """
        self.pre_change_days = pre_change_days
        self.post_change_days = self.total_days - self.pre_change_days

        ref_val = normalized_ref_value
        control_limit = normalized_threshold

        DetectionTimes = np.array([], dtype=int)
        Dj = np.array(
            [], dtype=int
        )  # save the Dj which are binary values indicating detection MTBFA
        Zj = np.array([], dtype=int)  # save the Zj = min(Tj,pre-change-days)-MTBFA
        zj = np.array([], dtype=int)  # ADD - MLE of delays
        cj = np.array([], dtype=int)  # ADD - binary
        self.AvgDD = np.array([])  # Average Detection Delay
        # D = np.array([])  # Displacement
        # FalsePos = np.array([])
        # TruePos = np.array([])

        # CUSUM for day0-60: outcomes are detection delay and #FP, #TP, MTBFA, False alarm rate
        # num_rows = np.shape(self.data)[0]
        in_control_data = self.data[: self.pre_change_days]
        # out_control_data = self.data[self.pre_change_days : self.total_days]
        # out_std = np.std(out_control_data)
        self.in_std = np.std(in_control_data)
        x = np.array(self.data)

        mu_0 = np.mean(in_control_data)
        # mu_1 = np.mean(out_control_data)
        # d = np.abs((mu_1 - mu_0) / self.in_std)

        # h      = 0.102       # Upper/lower control limit to detect the changepoint H=0.102, 0.127
        # k      = 0.03831     # Drift 0.01277 is the 1 sigma change, 0.0255 - one-sigma change, 0.03831 is 3-sigma change, 0.05108
        self.h = control_limit * self.in_std
        k = ref_val * self.in_std

        # Call compute CUSUM function with x (observatoins), in-control mean (mu) and k (drift or reference value)
        self.S_hi, self.S_lo, cusum = self.compute_cusum(x, mu_0, k)

        # False positives and Total alarms
        falsePos = 0
        alarms = 0
        # delay = 0
        avddd = 0  # this is the delay from the paper: td-ts (z_k-v) where v is the changepoint and z_k is the time of detection
        # MTBFA    = 0

        for i in range(0, self.pre_change_days):
            if (self.S_hi[i] > self.h) or (self.S_lo[i] > self.h):
                # if (i<pre_change_days):
                falsePos += 1  # False Positives
                # print("time false alarm",i)
                DetectionTimes = np.append(
                    DetectionTimes, i + 1
                )  # time at which a false positive is detected
                Dj = np.append(Dj, 1)
                Zj = np.append(Zj, min(i, self.pre_change_days))
                # print("detection times",DetectionTimes)
                # print("detection times size",DetectionTimes.size)
                break

        # If there is no false positive, Zj = pre_change_days, Dj = 0
        if falsePos == 0:
            Dj = np.append(Dj, 0)
            # DetectionTimes[runs] = pre_change_days
            Zj = np.append(Zj, self.pre_change_days)

        # Delay to detect the first changepoint
        # delay = 0
        for i in range(self.pre_change_days, self.total_days):
            if (self.S_hi[i] > self.h) or (self.S_lo[i] > self.h):
                alarms += 1  # True Positive: break after detecting one TP
                # print("alarm at : ", i)
                # delay  = i-1000+1    # ts is 100 because the change starts at day100
                avddd = i - self.pre_change_days
                cj = np.append(cj, 1)
                zj = np.append(zj, min(avddd, self.total_days))
                break

        # If there is no true detection, zj = total simulation days, cj = 0
        if alarms == 0:
            cj = np.append(cj, 0)
            # DetectionTimes[runs] = pre_change_days
            zj = np.append(zj, self.total_days)

        # Calculate MTBFA(Mean time time between False Alarms)
        # MTBFA = np.mean(DetectionTimes)
        # FlaseAlarmRate = 1/MTBFA

        # FalsePos = np.append(FalsePos, falsePos)
        # TruePos = np.append(TruePos, alarms)
        # DelaytoDetect = np.append(DelaytoDetect, delay)   # td-ts+1
        # FAR           = np.append(FAR, FlaseAlarmRate)
        # DetectionTimes= np.append(DetectionTimes, detectionTime)
        self.AvgDD = np.append(self.AvgDD, avddd)  # ADD estimate from the paper
        # outSTD_test_sp = np.append(outSTD_test_sp, out_std)
        # inSTD_test_sp  = np.append(inSTD_test_sp, in_std)
        # D = np.append(D, d)
        # self.h_1000 = np.append(self.h_1000, self.h)
        # self.k_1000 = np.append(self.k_1000, k)
        # print(falsePos)

    def plot_histogram_plotly(self, data, xlabel, title="") -> go.Figure:
        """
        histogram using plotly

        Args:
            data (_type_): Data values to show in histogram.
            xlabel (_type_): Title of the label for X-axis.
            title (str, optional): Title of the plot. Defaults to "".

        Returns:
            go.Figure: Histogram as Plotly graph object.
        """
        fig = go.Figure(data=[go.Histogram(x=data, nbinsx=30)])
        fig.update_layout(title="[TITLE=?]", xaxis_title=xlabel, yaxis_title="Count")
        fig.update_layout(plot_bgcolor=self.config["color"]["blue_005"])

        return fig

    def plot_input_metric_plotly(self) -> go.Figure:
        """
        Plot the input metric using Plotly.

        Returns:
            go.Figure: Scatter plot as Plotly graph object.
        """
        x1 = np.arange(self.pre_change_days)
        y1 = self.data[: self.pre_change_days]
        mean_y1 = np.mean(y1)

        x2 = np.arange(self.pre_change_days, self.total_days, 1)
        y2 = self.data[self.pre_change_days : self.total_days]
        mean_y2 = np.mean(y2)

        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.7, 0.3],
            shared_yaxes=True,
            horizontal_spacing=0.02,
        )

        font_size_title = 20
        font_size_legend = 18

        # add subplots
        fig.add_trace(
            go.Scatter(
                x=x1,
                y=y1,
                mode="markers",
                name=f"""In-control S<sub>p</sub>""",
                marker=dict(color="darkturquoise", size=10),
                opacity=0.4,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x2,
                y=y2,
                mode="markers",
                name=f"""Out-of-control S<sub>p</sub>""",
                marker=dict(color="coral", size=10),
                opacity=0.4,
            ),
            row=1,
            col=1,
        )

        # add horizontal lines
        fig.add_trace(
            go.Scatter(
                x=[min(x1), max(x1)],
                y=[mean_y1, mean_y1],
                mode="lines",
                name="In-control mean",
                line=dict(color="darkturquoise", dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[min(x2), max(x2)],
                y=[mean_y2, mean_y2],
                mode="lines",
                name="Out-of-control mean",
                line=dict(color="coral", dash="dash"),
            ),
            row=1,
            col=1,
        )

        # add vertical line
        fig.add_trace(
            go.Scatter(
                x=[self.pre_change_days, self.pre_change_days],
                y=[np.min(self.data), np.max(self.data)],
                mode="lines",
                name="Change-point",
                line=dict(color="grey", dash="dash"),
                # textfont=dict(size=18)
            ),
            row=1,
            col=1,
        )

        fig.update_layout(
            title={
                "text": "Average metric for pre- and post-change regime, and histogram",
                "font": {"size": font_size_title, "weight": "bold"},
            },
            xaxis_title={
                "text": "Length of Simulation (days)",
                "font": {"size": font_size_legend, "weight": "bold"},
            },
            yaxis_title={
                "text": "AI model metric",
                "font": {"size": font_size_legend, "weight": "bold"},
            },
            xaxis=dict(dtick=20),
        )

        fig.update_layout(plot_bgcolor=self.config["color"]["blue_005"])

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # add histogram (like marginal histogram)
        nbinsx = 15  # 6

        # add subplots
        fig.add_trace(
            go.Histogram(
                y=self.data[: self.pre_change_days],
                nbinsy=nbinsx,
                # name=f"""Pre-change S<sub>p</sub>""",
                showlegend=False,
                marker=dict(color="mediumturquoise"),
                opacity=0.4,
                orientation="h",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Histogram(
                y=self.data[self.pre_change_days : self.total_days],
                nbinsy=nbinsx,
                # name=f"""Post-change S<sub>p</sub>""",
                showlegend=False,
                marker=dict(color="coral"),
                opacity=0.4,
                orientation="h",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 20],  # [! y_max can should be used]
                y=[
                    np.mean(self.data[: self.pre_change_days]),
                    np.mean(self.data[: self.pre_change_days]),
                ],
                mode="lines",
                # name="Reference mean",
                showlegend=False,
                line=dict(color="mediumturquoise", dash="dash"),
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(
            title_text="Count",
            title_font_size=font_size_legend,
            title_font_weight="bold",
            row=1,
            col=2,
        )

        # update layout
        fig.update_layout(barmode="overlay")

        if self.config["control"]["save_figure"] == "true":
            fig.write_image(
                os.path.join(
                    self.config["path_output"]["path_figure"], "fig_plot_1.png"
                ),
                scale=3,
            )

        return fig

    def plot_cusum_plotly(self) -> go.Figure:
        """
        Plot CUSUM value using Plotly

        Returns:
            go.Figure: CUSUM plot using Plotly graph object.
        """
        fig = go.Figure()

        font_size_title = 20
        font_size_legend = 18

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
                x=[self.pre_change_days, self.pre_change_days],
                y=[
                    0,
                    np.max(self.S_lo / self.in_std),
                ],  # [! np.max(self.S_lo / self.in_std)?]
                mode="lines",
                name="Change-point",
                line=dict(color="grey", dash="dash"),
            )
        )

        fig.update_layout(
            title={
                "text": "CUSUM Chart",
                "font": {"size": font_size_title, "weight": "bold"},
            },
            xaxis_title={
                "text": "Length of Simulation (Days)",
                "font": {"size": font_size_legend, "weight": "bold"},
            },
            yaxis_title={
                "text": "CUSUM value",
                "font": {"size": font_size_legend, "weight": "bold"},
            },
            xaxis=dict(dtick=20),
        )

        fig.update_layout(plot_bgcolor=self.config["color"]["blue_005"])

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        if self.config["control"]["save_figure"] == "true":
            fig.write_image(
                os.path.join(
                    self.config["path_output"]["path_figure"], "fig_plot_2.png"
                ),
                scale=3,
            )

        return fig
