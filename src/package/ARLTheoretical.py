"""
ARLTheoretical

@author: smriti.prathapan
"""

import pandas as pd
import numpy as np
from collections import OrderedDict

# import rpy2's package module
import rpy2.robjects.packages as rpackages

# R vector of strings
from rpy2.robjects.vectors import StrVector

import rpy2.robjects as ro

# Suppress all R warnings globally
ro.r["options"](warn=-1)


def get_ref_value_k(h: float, ARL_0: float) -> float:
    """
    Calculation for the reference value for given h and ARL_0.

    Args:
        h (float): Normalized threshold.
        ARL_0 (float): ARL0 value.

    Returns:
        float: Normalized reference value k.
    """

    k = np.round(spc.xcusum_crit_L0h(ARL_0, h), decimals=4).tolist()[0]

    return k


def get_ref_value(
    h: float, list_ARL_0: list[float]
) -> tuple[pd.DataFrame, OrderedDict]:
    """
    provides normalized reference values k for provided list of ARL0, given the value of normalized threshold h.

    Args:
        h (float): Normalized threshold.
        list_ARL_0 (list): List of ARL0 values.

    Returns:
        tuple[pd.Dataframe, OrderedDict]: Dataframe of ARL0 and k, Data dictionary of ARL0 and k; where k is normalized reference value.
    """

    # data in Ordered dictionary to use it other functions instead of using DataFrame
    dict_ARL0_k = OrderedDict()

    summary_table_df_ARL0_k = pd.DataFrame(columns=["ARL_0", "k"])
    for n, ARL_0 in enumerate(list_ARL_0):
        k = np.round(spc.xcusum_crit_L0h(ARL_0, h), decimals=4).tolist()[0]
        summary_table_df_ARL0_k.loc[n] = [
            ARL_0,
            k,
        ]
        dict_ARL0_k[ARL_0] = k

    return summary_table_df_ARL0_k, dict_ARL0_k


def get_ARL_1_h_mu1_k(h: float, k: float, mu1: float) -> float:
    """
    Calculate ARL_1 with given Shift in Mean (mu1) and k.

    Args:
        h (float): Normalized threshold.
        k (float): Normalized reference value.
        mu1 (float): Intended shift in mean.

    Returns:
        float: Detection delay (ARL1).
    """
    
    ARL_1 = np.round(
        spc.xcusum_ad_(k=k, h=h, mu1=mu1, mu0=0, sided="two", r=20), decimals=2).tolist()[0] #Changing decimals to 2 digits to match the results from paper

    return ARL_1


def get_ARL_1(
    h: float, shift_in_mean: list[float], dict_ARL0_k: OrderedDict
) -> pd.DataFrame:
    """
    Get the ARL1 along with k values.

    Args:
        h (float): Normalized threshold.
        shift_in_mean (list[float]): List of the values of shift in mean.
        dict_ARL0_k (OrderedDict): Data dictionary of ARL0 and k

    Returns:
        pd.DataFrame: Table for ARL1 and k values.
    """

    list_ARL_0 = [ARL_0 for ARL_0 in dict_ARL0_k.keys()]

    dict_data_ARL1_k = OrderedDict()
    dict_data_ARL1_k["Shift in mean"] = shift_in_mean

    for ARL_0 in list_ARL_0:
        k = dict_ARL0_k[ARL_0]
        list_ARL_1 = []

        for mu1 in shift_in_mean:
            ARL_1 = np.round(
                spc.xcusum_ad_(k=k, h=h, mu1=mu1, mu0=0, sided="two", r=20), decimals=2).tolist()[0] #Changing decimals to 2 digits to match the results from paper
            list_ARL_1.append(ARL_1)

        dict_data_ARL1_k[ARL_0] = list_ARL_1

    summary_table_df_ARL1_k = pd.DataFrame(dict_data_ARL1_k)

    return summary_table_df_ARL1_k


# import R's utility package
utils = rpackages.importr("utils")
spc = rpackages.importr("spc")
# select a mirror for R packages
utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

# R package names
packnames = ("ggplot2", "hexbin", "lazyeval", "cusumcharter", "RcppCNPy", "spc")

# Selectively install what needs to be install
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
