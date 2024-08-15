"""
ARLTheoretical

@author: smriti.prathapan
"""

import rpy2

# import rpy2's package module
import rpy2.robjects.packages as rpackages

# R vector of strings
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.lib.ggplot2 as gp
import pandas as pd
import numpy as np
from collections import OrderedDict

# import R's utility package
utils = rpackages.importr("utils")
spc = rpackages.importr("spc")
# select a mirror for R packages
utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

# R package names
packnames = ("ggplot2", "hexbin", "lazyeval", "cusumcharter", "RcppCNPy", "spc")

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))


def get_ref_value(h: float, list_ARL_0: list):
    """
    # Code for table 4
    # Find k given h

    Example:
        h = 4 # Default choice for threshold
        list_ARL_0 = [50, 100, 150, 200, 300, 400, 500, 1000]
    """

    # data in Ordered dictionary to use it other functions instead of using DataFrame
    dict_ARL0_k = OrderedDict()

    # Print the reference values for an intended ARL_0 with normalized threshold, h=4
    print("Reference value, k: for an intended ARL_0 and h")
    summary_table_df_ARL0_k = pd.DataFrame(columns=["ARL_0", "k"])
    for n, ARL_0 in enumerate(list_ARL_0):
        k = np.round(spc.xcusum_crit_L0h(ARL_0, h), decimals=4).tolist()[0]
        summary_table_df_ARL0_k.loc[n] = [
            ARL_0,
            k,
        ]
        dict_ARL0_k[ARL_0] = k

    return summary_table_df_ARL0_k, dict_ARL0_k


def get_ARL_1(
    h: float, shift_in_mean: list, dict_ARL0_k: OrderedDict
):
    """
    # Code for table 5
    # xcusum.ad
    # Find ARL_1 for the k from above with h=4 for various shifts in mean

    Example:
        h = 4
        shift_in_mean = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6] # mu1
    """

    list_ARL_0 = [ARL_0 for ARL_0 in dict_ARL0_k.keys()]

    dict_data_ARL1_k = OrderedDict()
    dict_data_ARL1_k["Shift in mean"] = shift_in_mean

    for ARL_0 in list_ARL_0:
        k = dict_ARL0_k[ARL_0]
        list_ARL_1 = []

        for mu1 in shift_in_mean:
            ARL_1 = np.round(
                spc.xcusum_ad_(k=k, h=h, mu1=mu1, mu0=0, sided="two", r=20), decimals=4
            ).tolist()[0]
            list_ARL_1.append(ARL_1)

        dict_data_ARL1_k[ARL_0] = list_ARL_1

    summary_table_df_ARL1_k = pd.DataFrame(dict_data_ARL1_k)

    return summary_table_df_ARL1_k


# summary_table_ARL0_k, dict_ARL0_k = get_ref_value(
#     h=4, list_ARL_0=[50, 100, 150, 200, 300, 400, 500, 1000]
# )

# summary_table_df_ARL1_k = get_ARL_1(
#     h=4,
#     k=0.159,
#     mu1=0.1,
#     dict_ARL0_k=dict_ARL0_k,
#     shift_in_mean=[
#         0.1,
#         0.2,
#         0.3,
#         0.4,
#         0.5,
#         0.6,
#         0.7,
#         0.8,
#         0.9,
#         1.0,
#         1.1,
#         1.2,
#         1.3,
#         1.4,
#         1.5,
#         1.6,
#     ],  # mu1
# )
