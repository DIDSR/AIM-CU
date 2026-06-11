"""
AIM-CU is a statistical tool for AI monitoring based on a cumulative sum (AIM-CU) approach.

AIM-CU computes:
- The parameter choices for change-point detection based on an acceptable false alarm rate
- Detection delay estimates for a given displacement of the performance metric from the target for those parameter choices.

Notes:
- Requires: rpy2, an R installation, and R package `spc` (plus optional ggplot2/hexbin/lazyeval/cusumcharter/RcppCNPy).
- If R packages are missing, this code will attempt to install them (CRAN mirror must be reachable).
"""
#Packages
import os
import sys
import random
import warnings
from collections import OrderedDict
from typing import Tuple, List

import numpy as np
import pandas as pd
import tomli
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.resources import files

#Read config.toml 
def load_package_config():
    config_path = files("aim_cu").joinpath("config.toml")
    with config_path.open("rb") as f:
        return tomli.load(f)

_CONFIG = load_package_config()
#print("_CONFIG =", _CONFIG)
shift_in_mean = _CONFIG["CUSUM_params"]["shift_in_mean"]
# ---------------------------
# R / rpy2 setup (spc package)
# ---------------------------
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages

#---new imports
# import R's utility package
utils = rpackages.importr('utils')
spc = rpackages.importr('spc')

# Suppress all R warnings globally
ro.r["options"](warn=-1)

#warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module=r"rpy2\..*")
random.seed(58)

# --------------------------------------------------------
# I Compute CUSUM parameters: h,k and average run length
# --------------------------------------------------------
def get_ref_value(h: float, ARL_0: float) -> float:
    """
    Compute the normalized reference value (k) for a given threshold (h) and in-control Average Run Length (ARL₀).

    Parameters
    ----------
    h : float
        Normalized decision threshold of the CUSUM chart.
    ARL_0 : float
        Target in-control Average Run Length (ARL₀).

    Returns
    -------
    float
        Normalized reference value (k) corresponding to the given h and ARL₀.
    """
    k = np.round(spc.xcusum_crit_L0h(ARL_0, h), decimals=4).tolist()[0]
    return k


def get_threshold(k: float, ARL_0: float) -> float:
    """
    Compute the normalized threshold (h) for a given reference value (k) 
    and target in-control Average Run Length (ARL₀).

    Parameters
    ----------
    k : float
        Normalized reference value used in the CUSUM chart.
    ARL_0 : float
        Target in-control Average Run Length (ARL₀).

    Returns
    -------
    float
        Normalized threshold (h) corresponding to the given k and ARL₀.
    """
    h = np.round(
        spc.xcusum_crit_(k, ARL_0, mu0=0, hs=0, sided="one", r=30),
        decimals=4,
    ).tolist()[0]
    return h


def get_ref_values(h: float, list_ARL_0: List[float]) -> Tuple[pd.DataFrame, OrderedDict]:
    """
    Compute normalized reference values (k) for a list of target ARL₀ values 
    given a fixed threshold (h).

    Parameters
    ----------
    h : float
        Normalized decision threshold of the CUSUM chart.
    list_ARL_0 : List[float]
        List of target in-control Average Run Length (ARL₀) values.

    Returns
    -------
    Tuple[pandas.DataFrame, collections.OrderedDict]
        A tuple containing:
        - DataFrame: Table with columns ['ARL_0', 'k'] listing each ARL₀ and its corresponding k.
        - OrderedDict: Mapping of ARL₀ to k
    """
    dict_ARL0_k: OrderedDict = OrderedDict()
    summary_table_df_ARL0_k = pd.DataFrame(columns=["ARL_0", "k"])

    for n, ARL_0 in enumerate(list_ARL_0):
        k = np.round(spc.xcusum_crit_L0h(ARL_0, h), decimals=4).tolist()[0]
        summary_table_df_ARL0_k.loc[n] = [ARL_0, k]
        dict_ARL0_k[ARL_0] = k

    return summary_table_df_ARL0_k, dict_ARL0_k


def compute_ARL1(h: float, k: float, mu1: float) -> float:
    """
    Compute the out-of-control Average Run Length (ARL₁) for a given 
    threshold (h), reference value (k), and mean shift (μ₁).

    Parameters
    ----------
    h : float
        Normalized decision threshold.
    k : float
        Normalized reference value.
    mu1 : float
        Change in the process mean (standard deviations from the target value).

    Returns
    -------
    float
        Average Run Length (ARL₁), is the estimate of steady state ARL (expected detection delay) to detect the change in mean.
    """
    ARL_1 = np.round(
        spc.xcusum_ad_(k=k, h=h, mu1=mu1, mu0=0, sided="two", r=20),
        decimals=2,
    ).tolist()[0]
    return ARL_1


def compute_ARL1_table(h: float, shift_in_mean: List[float], dict_ARL0_k: OrderedDict) -> pd.DataFrame:
    """
    Compute a table of Average Run Length (ARL₁) values across multiple change in mean and ARL₀-reference value (k) combinations.

    Parameters
    ----------
    h : float
        Normalized decision threshold.
    shift_in_mean : List[float]
        List of mean shifts (μ₁), expressed in standardized units.
    dict_ARL0_k : OrderedDict
        Mapping of target in-control ARL₀ values to their corresponding
        normalized reference values (k).

    Returns
    -------
    pandas.DataFrame
        Table of ARL₁ values for mean shifts and corresponding ARL₀-k combinations.
    """
    list_ARL_0 = list(dict_ARL0_k.keys())

    dict_data_ARL1_k: OrderedDict = OrderedDict()
    dict_data_ARL1_k["Shift in mean"] = shift_in_mean

    for ARL_0 in list_ARL_0:
        k = dict_ARL0_k[ARL_0]
        list_ARL_1 = []
        for mu1 in shift_in_mean:
            ARL_1 = np.round(
                spc.xcusum_ad_(k=k, h=h, mu1=mu1, mu0=0, sided="two", r=20),
                decimals=2,
            ).tolist()[0]
            list_ARL_1.append(ARL_1)

        #Format column name: "50 (0.16)"
        col_name = f"{int(ARL_0)} ({k:.2f})"
        dict_data_ARL1_k[col_name] = list_ARL_1

        #dict_data_ARL1_k[ARL_0] = list_ARL_1

    return pd.DataFrame(dict_data_ARL1_k)


# -------------------------------------------
# II Performance drift detection using CUSUM
# -------------------------------------------
class CUSUM:
    """
    CUSUM-based performance drift detection.

    This class provides methods to initialize baseline parameters, compute
    positive and negative CUSUM statistics, detect performance drifts, and
    generate CUSUM charts.
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
        Load and initialize configuration settings from the configuration file (`config.toml`).

        This method populates the 'config' attribute with parameters such as plotting options, and default values.

        Returns
        -------
        None
        """
        try:
            path_file_config = os.path.abspath("config.toml")
            with open(path_file_config, "rb") as file_config:
                self.config = tomli.load(file_config)
        except FileNotFoundError:
            print("Error: config.toml not found at", path_file_config)
            sys.exit(1)

    def set_init_stats(self, init_days: int) -> None:
        """
        Compute in-control parameters from baseline observations.

        Uses the first 'init_days' observations to estimate the in-control mean and standard deviation.

        Parameters
        ----------
        init_days : int
            Number of observations assumed to be in-control.

        Returns
        -------
        None
        """
        self.init_days = init_days
        in_control_data = self.data[: self.init_days]
        self.in_std = float(np.std(in_control_data))
        self.in_mu = float(np.mean(in_control_data))

    def set_timeline(self, data: np.ndarray) -> None:
        """
        Set the timeline of the performance metric.

        Determines the total number of observations in the input data.

        Parameters
        ----------
        data : np.ndarray
            1D Array of the AI performance metric.

        Returns
        -------
        None
        """
        self.total_days = int(np.shape(data)[0])

    def set_df_metric_default(self) -> None:
        """
        Read the AI performance metric from the default CSV file and initializes the internal 
        data and set the timeline.

        Raises
        ------
        SystemExit
            If the CSV file is not found at the specified path.

        Returns
        -------
        None
        """
        try:
            path_csv = os.path.abspath(os.path.join("../../", self.config["path_input"]["path_df_metric"]))
            self.df_metric = pd.read_csv(path_csv)
        except FileNotFoundError:
            print("Error: CSV file not found at", path_csv)
            sys.exit(1)

        self.data = self.df_metric[self.df_metric.columns[1]].to_numpy()
        self.set_timeline(self.data)

    def set_df_metric_csv(self, data_csv: pd.DataFrame) -> None:
        """
        Assign the performance metric data read fom the input csv file to a datafram .
        Parameters
        ----------
        data_csv : pandas.DataFrame
            DataFrame containing the performance metric data. The values are expected in the second column (index 1).

        Returns
        -------
        None
        """
        self.df_metric = data_csv
        self.data = self.df_metric[self.df_metric.columns[1]].to_numpy()
        self.set_timeline(self.data)

    def compute_cusum(
        self, x: List[float], mu_0: float, ref_val: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute CUSUM statistics for the performance metric.

        Calculates the positive (S_hi) and negative (S_lo) cumulative sums,
        along with the cumulative deviation from the in-control mean, to
        monitor for shifts in the process.

        Parameters
        ----------
        x : List[float]
            Sequence of observed performance metric values.
        mu_0 : float
            In-control mean (baseline) of the performance metric.
        ref_val : float
            Reference value (k) that determines sensitivity to shifts.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - S_hi : Positive CUSUM values (detects upward shifts)
            - S_lo : Negative CUSUM values (detects downward shifts)
            - cusum : Cumulative sum of deviations from the mean
        """
        x = np.asarray(x, dtype=float)
        num_rows = x.shape[0]

        x_mean = np.zeros(num_rows, dtype=float)
        self.S_hi = np.zeros(num_rows, dtype=float)
        self.S_lo = np.zeros(num_rows, dtype=float)
        cusum = np.zeros(num_rows, dtype=float)

        # Starts with 0
        self.S_hi[0] = 0.0
        self.S_lo[0] = 0.0
        cusum[0] = 0.0

        mean_hi = np.zeros(num_rows, dtype=float)
        mean_lo = np.zeros(num_rows, dtype=float)

        
        x_mean[0] = x[0] - mu_0
        mean_hi[0] = x[0] - mu_0 - ref_val
        mean_lo[0] = mu_0 - ref_val - x[0]

        for i in range(1, num_rows):
            x_mean[i] = x[i] - mu_0
            mean_hi[i] = x[i] - mu_0 - ref_val
            self.S_hi[i] = max(0.0, self.S_hi[i - 1] + mean_hi[i])
            mean_lo[i] = mu_0 - ref_val - x[i]
            self.S_lo[i] = max(0.0, self.S_lo[i - 1] + mean_lo[i])
            cusum[i] = cusum[i - 1] + x_mean[i]

        return (
            np.round(self.S_hi, 2),
            np.round(self.S_lo, 2),
            np.round(cusum, 2),
        )

    def change_detection(
        self,
        normalized_ref_value: float = 0.5,
        normalized_threshold: float = 4,
    ) -> None:
        """
        Detect changes in the process using CUSUM statistics.

        Computes the CUSUM values and identifies the first point at which
        the process deviates significantly from the in-control state based
        on the specified reference value and threshold.

        Parameters
        ----------
        normalized_ref_value : float, optional
            Normalized reference value (k) that controls sensitivity to
            shifts in the process mean. Default is 0.5.
        normalized_threshold : float, optional
            Normalized decision threshold (h) used to signal a change.
            Default is 4.
        Returns
        -------
        None
        """
        self.pre_change_days = None

        control_limit = normalized_threshold
        self.H = control_limit * self.in_std
        ref_val = normalized_ref_value * self.in_std

        x = np.array(self.data, dtype=float)
        self.S_hi, self.S_lo, _ = self.compute_cusum(x, self.in_mu, ref_val)
        
        # Find first occurrence where the threshold is exceeded
        S_hi_exceeds = np.where(self.S_hi > self.H)[0]
        S_lo_exceeds = np.where(self.S_lo > self.H)[0]
        
        # Detect whether S_hi or S_lo exceeds threshold
        if len(S_hi_exceeds) > 0 and len(S_lo_exceeds) > 0:
            if S_hi_exceeds[0] < S_lo_exceeds[0]:
                self.pre_change_days = int(S_hi_exceeds[0])
                print(f"Detected upward drift at: {self.pre_change_days}")
            else:
                self.pre_change_days = int(S_lo_exceeds[0])
                print(f"Detected downward drift at: {self.pre_change_days}")
        elif len(S_hi_exceeds) > 0:
            self.pre_change_days = int(S_hi_exceeds[0])
            print(f"Detected upward drift at: {self.pre_change_days}")
        elif len(S_lo_exceeds) > 0:
            self.pre_change_days = int(S_lo_exceeds[0])
            print(f"Detected downward drift at: {self.pre_change_days}")
        else:
            print("No performance drift detected")
        
    def plot_input_data(self):
        """
        Plot the AI performance metric with baseline region highlighted.

        Generates a scatter plot of the performance metric over time

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure containing the scatter plot of the data.
        """
        x1 = np.arange(self.init_days)
        y1 = self.data[: self.init_days]

        x2 = np.arange(self.init_days, self.total_days, 1)
        y2 = self.data[self.init_days : self.total_days]
        
        fig, ax = plt.subplots(figsize=(10, 5))

        # Scatter plots
        ax.scatter(x1, y1, color="lime", s=20, alpha=0.4)
        ax.scatter(x2, y2, color="lime", s=20, alpha=0.2)

        # Highlight baseline 
        ax.axvspan(0, self.init_days, color="palegreen", alpha=0.25)

        # Labels and title
        ax.set_title("AI output", fontsize=16, fontweight="bold")
        ax.set_xlabel("Time", fontsize=14, fontweight="bold")
        ax.set_ylabel("AI model metric", fontsize=14, fontweight="bold")

        # Tick spacing (similar to dtick=20)
        ax.set_xticks(np.arange(0, self.total_days, 20))

        # Background color (optional)
        try:
            ax.set_facecolor(self.config["color"]["blue_005"])
        except Exception:
            pass  # fallback if config not set

        # Remove legend 
        ax.legend().set_visible(False) if ax.get_legend() else None

        plt.tight_layout()

        return fig

    def plot_changepoint(self):
        """
        Plot the AI performance metric with the detected change pointhighlighting in-control and out-of-control regions 
        and marking the detected change point.

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure containing the change-point plot.
        """

        # Determine change-point
        if self.pre_change_days is None:
            split = self.init_days
        else:
            split = self.pre_change_days

        # Split data
        x1 = np.arange(split)
        y1 = self.data[:split]
        mean_y1 = np.mean(y1) if len(y1) else 0.0

        x2 = np.arange(split, self.total_days)
        y2 = self.data[split:self.total_days]
        mean_y2 = np.mean(y2) if len(y2) else 0.0

        fig, ax = plt.subplots(figsize=(10, 5))

        # Scatter plots
        ax.scatter(x1, y1, color="darkturquoise", s=20, alpha=0.4, label="In-control data")
        ax.scatter(x2, y2, color="coral", s=20, alpha=0.4, label="Test data")

        # Mean lines
        if len(x1):
            ax.plot([x1.min(), x1.max()], [mean_y1, mean_y1],
                    linestyle="--", color="darkturquoise", label="In-control mean")

        if len(x2):
            ax.plot([x2.min(), x2.max()], [mean_y2, mean_y2],
                    linestyle="--", color="coral", label="Test mean")

        # Change-point line
        ax.axvline(x=split, color="grey", linestyle="--", label="Detected change-point")

        # Labels and title
        ax.set_title("AI model performance versus time", fontsize=16, fontweight="bold")
        ax.set_xlabel("Time", fontsize=14, fontweight="bold")
        ax.set_ylabel("AI model performance", fontsize=14, fontweight="bold")

        # Tick spacing similar to dtick=20
        ax.set_xticks(np.arange(0, self.total_days, 20))

        # Background color (optional)
        try:
            ax.set_facecolor(self.config["color"]["blue_005"])
        except Exception:
            pass

        # Legend (top center like Plotly)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True)

        plt.tight_layout()
    
        return fig

    

    def plot_cusum_chart(self):
        """
        Plot the CUSUM chart of the performance metric: the positive (S_hi) and negative (S_lo) CUSUM statistics 
        along with the decision threshold 
        used for change detection.

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure with the CUSUM chart.
        """

        fig, ax = plt.subplots(figsize=(10, 5))

        # X axis
        x = np.arange(len(self.S_hi))

        # Plot S_hi and S_lo (normalized)
        ax.plot(x, self.S_hi / self.in_std,
                label="Positive changes (S_hi)",
                color="cyan")

        ax.plot(x, self.S_lo / self.in_std,
                label="Negative changes (S_lo)",
                color="darkcyan")

        # Threshold line
        threshold = self.H / self.in_std
        ax.plot([0, len(self.S_lo)],
                [threshold, threshold],
                linestyle="--",
                color="magenta",
                label="Threshold (h)")

        # Determine split point
        split = self.pre_change_days if self.pre_change_days is not None else self.init_days

        # Background shading (like Plotly rectangles)
        try:
            ax.axvspan(0, split,
                    color=self.config["color"]["blue_005"],
                    alpha=0.8)

            ax.axvspan(split, len(self.S_lo),
                    color="rgb(253, 243, 235)",
                    alpha=0.8)
        except Exception:
            # fallback if config missing
            ax.axvspan(0, split, color="lightblue", alpha=0.3)
            ax.axvspan(split, len(self.S_lo), color="mistyrose", alpha=0.3)

        # Labels and title
        ax.set_title("CUSUM Chart", fontsize=16, fontweight="bold")
        ax.set_xlabel("Time", fontsize=14, fontweight="bold")
        ax.set_ylabel("CUSUM value", fontsize=14, fontweight="bold")

        # Tick spacing similar to dtick=20
        ax.set_xticks(np.arange(0, len(self.S_lo), 20))

        # Legend (top center)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True)

        plt.tight_layout()

        return fig