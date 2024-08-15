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


def get_ref_value(h: float):
    """
    # Code for table 4
    # Find k given h

    Example:
        # L0 = 50  #ARL0
        h  =  4  #Default choice for threshold
    """

    # Try out this code
    # k = spc.xcusum_crit_L0h(L0, h)  # .crit.L0h(L0,h)  #xcusum.crit.L0h(L0, h)

    # Print the reference values for an intended ARL_0 with normalized threshold, h=4
    print("Reference value, k: for an intended ARL_0 and h")
    summary_table = pd.DataFrame(columns=["ARL_0", "k"])
    summary_table.loc[1] = [50, np.round(spc.xcusum_crit_L0h(50, h), decimals=4)]
    summary_table.loc[2] = [100, np.round(spc.xcusum_crit_L0h(100, h), decimals=4)]
    summary_table.loc[3] = [150, np.round(spc.xcusum_crit_L0h(150, h), decimals=4)]
    summary_table.loc[4] = [200, np.round(spc.xcusum_crit_L0h(200, h), decimals=4)]
    summary_table.loc[5] = [300, np.round(spc.xcusum_crit_L0h(300, h), decimals=4)]
    summary_table.loc[6] = [400, np.round(spc.xcusum_crit_L0h(400, h), decimals=4)]
    summary_table.loc[7] = [500, np.round(spc.xcusum_crit_L0h(500, h), decimals=4)]
    summary_table.loc[8] = [1000, np.round(spc.xcusum_crit_L0h(1000, h), decimals=4)]

    return summary_table


def get_ARL_1(h: float, k: float, mu1: float, ref_val: list, shift_in_mean: list):
    """
    # Code for table 5
    # xcusum.ad
    # Find ARL_1 for the k from above with h=4 for various shifts in mean

    Example:
        h = 4
        k = 0.159
        mu1 = 0.1
        ref_val = [0.159] # k
        shift_in_mean = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6] # mu1
    """

    ARL_1 = spc.xcusum_ad_(k=k, h=h, mu1=mu1, mu0=0, sided="two", r=20)

    # Print the ARL_1s for k computed from above for a certain ARL_0 with normalized threshold, h=4
    print("ALR_1s for an intended ARL_0 and h")

    summary_table = pd.DataFrame(columns=["Shift in mean", "k"])

    # ref_val = [0.159, 0.299, 0.3713, 0.4191, 0.4830, 0.5264, 0.5591, 0.6567] #k

    i = 1
    for k in ref_val:
        for mu1 in shift_in_mean:
            ARL_1 = spc.xcusum_ad_(k=k, h=h, mu1=mu1, mu0=0, sided="two", r=20)
            summary_table.loc[i] = [mu1, np.round(ARL_1, decimals=4)]
            i += 1

    return summary_table

# table_4 = get_ref_value(
#     h=4
# )
d = {}
d['first_level'] = pd.DataFrame(columns=['idx', 'a', 'b', 'c'],
                                         data=[[10, 0.89, 0.98, 0.31],
                                               [20, 0.34, 0.78, 0.34]]).set_index('idx')
table_4 = pd.concat(d, axis=1)
table_4.columns = ['_'.join(col) for col in table_4.columns]
print(table_4.to_string())

table_5 = get_ARL_1(
    h = 4,
    k = 0.159,
    mu1 = 0.1,
    ref_val = [0.159], # k
    shift_in_mean = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6] # mu1
)

# print(table_5.to_string())