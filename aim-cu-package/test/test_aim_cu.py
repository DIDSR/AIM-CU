#Usage:
#  python test_aim_cu.py --csv /scratch/smriti.prathapan/fromOHPC/note1/AIM-CU-package/spec-60-60.csv  --norm_h 5

import argparse
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")
import aim_cu
import matplotlib
from collections import OrderedDict
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="input the CSV path")
    parser.add_argument("--init_days", type=int, default=30)
    parser.add_argument("--norm_k", type=float, default=0.5)
    parser.add_argument("--norm_h", type=float, default=4.0)
    #parser.add_argument("--show_plots", action="store_true")
    args = parser.parse_args()
    
    # --------------------------------------------------------
    # I Compute CUSUM parameters and average run length
    # --------------------------------------------------------
    h = float(args.norm_h)
    print("\n=== SPC helper functions for computing CUSUM parameters ===")
    k_for_arl100 = aim_cu.get_ref_value(h=h, ARL_0=100.0)
    #print(f"k (for h={h}, ARL0=100): {k_for_arl100}")
    
    h_back = aim_cu.get_threshold(k=k_for_arl100, ARL_0=100.0)
    print(f"h (for k={k_for_arl100}, ARL0=100): {h_back}")
    
    df_k, dict_k = aim_cu.get_ref_values(h=4.0, list_ARL_0=[50.0, 100.0, 200.0])
    print(f"list of k values  h={h} : \n")
    print(df_k)
    
    #print(aim_cu.shift_in_mean)
    #print(aim_cu.dict_ARL0_k)

    ARL_1 = aim_cu.compute_ARL1(h=h_back,k=k_for_arl100, mu1=0.65)
    print(f"ARL_1 : {ARL_1}")
    
    df_arl_1 = aim_cu.compute_ARL1_table(h=4, shift_in_mean=[0.1, 0.2, 0.3], dict_ARL0_k=dict_k)
    print(f"ARL_1 table : {df_arl_1}")

    # -------------------------------
    # II Performance drift detection
    # -------------------------------
    print("\n=== Performance drift detection using CUSUM ===")
    cusum = aim_cu.CUSUM() 
    

    # If config.toml is not found, initialize() will sys.exit(1)
    try:
        cusum.initialize()
    except SystemExit:
        print("WARNING: config.toml not found. Using fallback config.")
        cusum.config = {
            "color": {"blue_005": "white"},
            "control": {"save_figure": "false"},
            "path_output": {"path_figure": "."},
        }

    if args.csv:
        df = pd.read_csv(args.csv)
        cusum.set_df_metric_csv(df)
    else:
        cusum.set_df_metric_default()

    cusum.set_init_stats(args.init_days)
    print(f"Total observations: {cusum.total_days}")
    print(f"Baseline observations: {cusum.init_days}")

    cusum.change_detection(normalized_ref_value=args.norm_k, normalized_threshold=args.norm_h)
    
    fig1 = cusum.plot_input_data()
    #fig1.write_image("input_data.png") 
    fig1.savefig("input_data.png", dpi=200, bbox_inches="tight")

    fig2 = cusum.plot_changepoint()
    fig2.savefig("changepoint.png", dpi=200, bbox_inches="tight")
    
    fig3 = cusum.plot_cusum_chart()
    fig3.savefig("cusum_chart.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
