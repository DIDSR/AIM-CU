'''
Utilities to handle different operations
'''

import pandas as pd
import great_tables as gt

def get_greattable_as_html(df: pd.DataFrame):
    table_great_table = gt.GT(data=df)

    return table_great_table.as_raw_html()