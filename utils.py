"""
Utilities to handle different operations
"""

import pandas as pd
import great_tables as gt
from collections import OrderedDict


# Get the great_table as HTML from Pandas dataframe
def get_greattable_as_html(df: pd.DataFrame):
    table_great_table = gt.GT(data=df)

    return table_great_table.as_raw_html()


# Populate ARLTheoretical.summary_table_df_ARL0_k
def populate_summary_table_ARL0_k(summary_table_df_ARL0_k: pd.DataFrame):
    table_great_table_ARL0_k = (
        gt.GT(summary_table_df_ARL0_k)
        .tab_header(
            title=gt.html(
                "Reference Values for an intended ARL<sub>0</sub> with normalized threshold, h = 4"
            )
        )
    )
    return table_great_table_ARL0_k.as_raw_html()


# Populate Multiindex table specific for ARLTheoretical.summary_table_df_ARL1_k
def populate_summary_table_ARL1_k(
    summary_table_df_ARL1_k: pd.DataFrame, dict_ARL0_k: OrderedDict
):
    list_ARL_0 = [str(ARL_0) for ARL_0 in dict_ARL0_k.keys()]
    list_k = [k for k in dict_ARL0_k.values()]

    format_k_ARL_0 = lambda k, ARL_0: gt.html(str(k) + "<br>" + "(" + str(ARL_0) + ")")

    column_label_dict = {
        ARL_0: format_k_ARL_0(k, ARL_0) for ARL_0, k in zip(list_ARL_0, list_k)
    }

    table_great_table_ARL1_k = (
        gt.GT(summary_table_df_ARL1_k)
        .tab_header(
            title=gt.html(
                "Estimate of steady state ARL (ARL<sub>1</sub>) based on the computed reference values and intended zero-state ARL (ARL<sub>0</sub>) with normalized threshold, h = 4)"
            )
        )
        .tab_stubhead(label="Shift in mean")
        .tab_spanner(
            label=gt.html("Reference Values<br>(Intended ARL<sub>0</sub>)"),
            columns=list_ARL_0,
        )
        .cols_move_to_start(columns=["Shift in mean"])
        .cols_label(**column_label_dict)
        .data_color(palette=["#D0E4F0", "#A3CAE1", "#77B0D2"])
    )
    return table_great_table_ARL1_k.as_raw_html()
