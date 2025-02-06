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

        self.H = None
        self.in_std = None
        self.in_mu = None
        self.S_hi = None
        self.S_lo = None

        self.config = None

        self.total_days = None
        self.pre_change_days = None
        self.post_change_days = None

        self.init_days = None

    def initialize(self) -> None:
        """
        Initialize with the configuration file.
        """
        try:
            path_file_config = os.path.abspath("../../config/config.toml")

            with open(path_file_config, "rb") as file_config:
                self.config = tomli.load(file_config)
        except FileNotFoundError:
            print("Error: config.toml not found at", path_file_config)
            sys.exit(1)

    def set_init_stats(self, init_days: int) -> None:
        """
        Use initial days to calculate in-control mean and standard deviation.

        Args:
            init_days (int, optional): Initial days when observations are considered stable. Defaults to 30.
        """
        self.init_days = init_days

        in_control_data = self.data[
            : self.init_days
        ]  # Assume the input data has more than 30 observations
        self.in_std = np.std(in_control_data)
        self.in_mu = np.mean(in_control_data)  # In-control mean

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
            path_csv = os.path.abspath(
                os.path.join("../../", self.config["path_input"]["path_df_metric"])
            )
            self.df_metric = pd.read_csv(path_csv)
        except FileNotFoundError:
            print("Error: CSV file not found at", path_csv)
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
            x (list[float]): Performance metric to be monitored
            mu_0 (float)   : In-control mean of the observations/performance metric
            k (float)      : Reference value related to the magnitude of change that one is interested in detecting

        Returns:
            tuple[list[float], list[float], list[float]]: Positive cumulative sum, negative cumulative sum, and CUSUM
        """
        num_rows = np.shape(x)[0]

        x_mean = np.zeros(num_rows, dtype=float)
        # S_hi : sum of positive changes --------------------------
        self.S_hi = np.zeros(num_rows, dtype=float)
        self.S_hi[0] = 0.0  # starts with 0
        # Increase in mean = x-mu-k ----------------------------
        mean_hi = np.zeros(num_rows, dtype=float)

        # Decrease in mean = mu-k-x----------------------------
        mean_lo = np.zeros(num_rows, dtype=float)
        # S_lo : sum of negative changes --------------------------
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
        normalized_ref_value: float = 0.5,
        normalized_threshold: float = 4,
    ) -> None:
        """
        Detects a change in the process.

        Args:
            pre_change_days (int)                 : Number of days for in-control phase.
            normalized_ref_value (float, optional): Normalized reference value for detecting a unit standard deviation change in mean of the process. Defaults to 0.5.
            normalized_threshold (float, optional): Normalized threshold. Defaults to 4.
        """
        self.pre_change_days = self.init_days  # This is the #initial days that we assume to be in-control - user enters or default = 30

        ref_val = normalized_ref_value
        control_limit = normalized_threshold

        DetectionTimes = np.array([], dtype=int)
        Dj = np.array(
            [], dtype=int
        )  # save the Dj which are binary values indicating detection MTBFA
        Zj = np.array([], dtype=int)  # save the Zj = min(Tj,pre-change-days)-MTBFA
        zj = np.array([], dtype=int)  # ADD - Maximum likelihood estimate of delays
        cj = np.array(
            [], dtype=int
        )  # ADD - binary - whether there is a change-detection (1) or not (0)
        self.AvgDD = np.array([])  # Average Detection Delay

        self.H = control_limit * self.in_std  # Threhold
        k = ref_val * self.in_std  # Reference value

        x = np.array(self.data)

        # Call compute CUSUM function with x (observatoins), in-control mean (mu) and k (drift or reference value)
        self.S_hi, self.S_lo, cusum = self.compute_cusum(x, self.in_mu, k)

        # Check the variations in self.S_hi and self.S_lo to determine whether there was a change in the data
        S_hi_last_known_zero = np.where(self.S_hi == 0)[
            0
        ]  # Find all the indices where self.S_hi was 0
        S_hi_start_of_change = (
            S_hi_last_known_zero[-1] + 1
        )  # Fetch the last entry where self.S_hi was 0

        S_lo_last_known_zero = np.where(self.S_lo == 0)[
            0
        ]  # Find all the indices where self.S_lo was 0
        S_lo_start_of_change = (
            S_lo_last_known_zero[-1] + 1
        )  # Fetch the last entry where self.S_lo was 0

        # Display the print messages in the UI
        if (S_lo_start_of_change < S_hi_start_of_change) and (
            self.S_lo[S_lo_start_of_change + 10] > self.H
        ):  # check if the changes in the next 10 observations exceed the threshold
            print(
                f"Change-point with respect to S_lo is: {S_lo_start_of_change}"
            )  # Use this change-point to generate histograms
            self.pre_change_days = S_lo_start_of_change

        elif (S_hi_start_of_change < S_lo_start_of_change) and (
            self.S_hi[S_hi_start_of_change + 10] > self.H
        ):
            print(f"Change-point with respect to S_hi is: {S_hi_start_of_change}")
            self.pre_change_days = S_hi_start_of_change
        else:
            print(f"No change")

        # False positives and Total alarms
        falsePos = 0
        alarms = 0
        avddd = 0  # this is the delay from the paper: td-ts (z_k-v) where v is the changepoint and z_k is the time of detection

        for i in range(0, self.pre_change_days):
            if (self.S_hi[i] > self.H) or (self.S_lo[i] > self.H):
                falsePos += 1  # False Positives
                DetectionTimes = np.append(
                    DetectionTimes, i + 1
                )  # time at which a false positive is detected
                Dj = np.append(Dj, 1)
                Zj = np.append(Zj, min(i, self.pre_change_days))
                break

        # If there is no false positive, Zj = pre_change_days, Dj = 0
        if falsePos == 0:
            Dj = np.append(Dj, 0)
            Zj = np.append(Zj, self.pre_change_days)

        # Delay to detect the first changepoint
        # delay = 0
        for i in range(self.pre_change_days, self.total_days):
            if (self.S_hi[i] > self.H) or (self.S_lo[i] > self.H):
                alarms += 1  # True Positive: break after detecting one TP
                cj = np.append(cj, 1)
                zj = np.append(zj, min(i, self.total_days) - self.pre_change_days)
                break

        # If there is no true detection, zj = total simulation days, cj = 0
        if alarms == 0:
            cj = np.append(cj, 0)
            zj = np.append(zj, self.total_days)

        self.AvgDD = np.append(self.AvgDD, avddd)  # ADD estimate from the paper

    def plot_input_metric_plotly_raw(self) -> go.Figure:
        """
        Plot AI output using Plotly.

        Returns:
            go.Figure: Scatter plot as Plotly graph object.
        """
        x1 = np.arange(self.init_days)
        y1 = self.data[: self.init_days]

        x2 = np.arange(self.init_days, self.total_days, 1)
        y2 = self.data[self.init_days : self.total_days]

        fig = make_subplots(
            rows=1,
            cols=1,
            shared_yaxes=True,
            horizontal_spacing=0.02,
        )

        font_size_title = 20
        font_size_legend = 18

        # separate in sublots
        fig.add_trace(
            go.Scatter(
                x=x1,
                y=y1,
                mode="markers",
                marker=dict(color="lime", size=10),
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
                marker=dict(color="lime", size=10),
                opacity=0.2,
            ),
            row=1,
            col=1,
        )

        fig.add_vrect(
            x0=0,
            x1=self.init_days,
            annotation_text="Initial days",
            annotation_position="top right",
            fillcolor="palegreen",
            opacity=0.25,
            line_width=0,
        )

        fig.update_layout(
            title={
                "text": "AI output",
                "font": {"size": font_size_title, "weight": "bold"},
            },
            xaxis_title={
                "text": "Number of baseline observations",
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

        fig.update_layout(showlegend=False)

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
                name=f"""In-control data""",
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
                name=f"""Out-of-control data""",
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
                "text": "Pre- and post-change observations and their respective histograms",
                "font": {"size": font_size_title, "weight": "bold"},
            },
            xaxis_title={
                "text": "Number of baseline observations",
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
                    os.path.abspath(
                        os.path.join(
                            "../../", self.config["path_output"]["path_figure"]
                        )
                    ),
                    "fig_plot_average_metric.png",
                ),
                scale=3,
            )
            print(
                "Created",
                os.path.join(
                    os.path.abspath(
                        os.path.join(
                            "../../", self.config["path_output"]["path_figure"]
                        )
                    ),
                    "fig_plot_average_metric.png",
                ),
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
                y=[self.H / self.in_std, self.H / self.in_std],
                mode="lines",
                name="Threshold (h)",
                line=dict(color="skyblue", dash="dash"),
            )
        )

        # add vertical line
        fig.add_trace(
            go.Scatter(
                x=[self.pre_change_days, self.pre_change_days],
                y=[
                    0,
                    np.max(self.S_lo / self.in_std),
                ],
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
                "text": "Number of baseline observations",
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
                    os.path.abspath(
                        os.path.join(
                            "../../", self.config["path_output"]["path_figure"]
                        )
                    ),
                    "fig_plot_cusum_chart.png",
                ),
                scale=3,
            )
            print(
                "Created",
                os.path.join(
                    os.path.abspath(
                        os.path.join(
                            "../../", self.config["path_output"]["path_figure"]
                        )
                    ),
                    "fig_plot_cusum_chart.png",
                ),
            )

        return fig
