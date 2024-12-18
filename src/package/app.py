"""
Gradio user interface for AIM-CU
"""

import os
import sys
import pandas as pd
import gradio as gr
import tomli
from cusum import CUSUM
from ARLTheoretical import get_ref_value, get_ref_value_k, get_ARL_1, get_ARL_1_h_mu1_k
from utils import (
    populate_summary_table_ARL0_k,
    populate_summary_table_ARL1_k,
)
import great_tables as gt
import plotly.graph_objects as go


def populate_table(h: str) -> tuple[gt.GT, gt.GT]:
    """
    Populate tables for ARL0 and ARL1 given the value of h

    Args:
        h (str): Normalized threshold.

    Returns:
        tuple[gt.GT, gt.GT]: Table for ARL0 and k in HTML format; table for ARL1 and k in HTML format.
    """
    h = float(h)

    summary_table_df_ARL0_k, dict_ARL0_k = get_ref_value(
        h=h,
        list_ARL_0=config["params_cusum"]["list_ARL_0"],
    )

    summary_table_df_ARL1_k = get_ARL_1(
        h=h,
        shift_in_mean=config["params_cusum"]["shift_in_mean"],
        dict_ARL0_k=dict_ARL0_k,
    )

    return populate_summary_table_ARL0_k(
        summary_table_df_ARL0_k
    ), populate_summary_table_ARL1_k(summary_table_df_ARL1_k, dict_ARL0_k)


def calculate_reference_value_k(h: str, arl_0: str) -> float:
    """
    Gets the reference value for given h and ARL_0.

    Args:
        h (str): Normalized threshold.
        arl_0 (str): ARL0 value.

    Returns:
        float: Normalized reference value k.
    """
    h = float(h)
    arl_0 = float(arl_0)

    k = get_ref_value_k(h=h, ARL_0=arl_0)

    return k


def calculate_arl1_h_k_mu1(h: str, k: str, mu1: str) -> float:
    """
    Get the ARL_1 with given Shift in Mean (mu1) and k.

    Args:
        h (str): Normalized threshold.
        k (str): Normalized reference value.
        mu1 (str): Intended shift in mean.

    Returns:
        float: Detection delay (ARL1).
    """
    h = float(h)
    k = float(k)
    mu1 = float(mu1)

    arl_1 = get_ARL_1_h_mu1_k(h=h, k=k, mu1=mu1)

    return arl_1


def populate_cusum_plots(
    file_csv_metric: gr.File,
    ref_value: str,
    normalized_threshold: str,
    pre_change_days: str,
) -> tuple[go.Figure, go.Figure]:
    """
    Populate CUSUM plots

    Args:
        file_csv_metric (gr.File): CSV file with metric data
        ref_value (str): Normalized reference value for detecting a unit standard deviation change in mean of the process.
        normalized_threshold (str): Normalized threshold.
        pre_change_days (str): Number of days for in-control phase.

    Returns:
        tuple[go.Figure, go.Figure]: Scatter plot as Plotly graph object; CUSUM plot using Plotly graph object.
    """
    ref_value = float(ref_value)
    normalized_threshold = float(normalized_threshold)
    pre_change_days = int(pre_change_days)

    if file_csv_metric is not None:
        # upload CSV file
        data_csv_metric = pd.read_csv(file_csv_metric.name)
        obj_cusum.set_df_metric_csv(data_csv_metric)
    else:
        # use the example CSV data
        obj_cusum.set_df_metric_default()

    obj_cusum.change_detection(
        pre_change_days=pre_change_days,
        normalized_ref_value=ref_value,
        normalized_threshold=normalized_threshold,
    )

    return (obj_cusum.plot_input_metric_plotly(), obj_cusum.plot_cusum_plotly())


with gr.Blocks(
    theme=gr.themes.Base(
        neutral_hue=gr.themes.Color(
            c50="#e5f1f8",
            c100="#e5f1f8",
            c200="#cce4f1",
            c300="#b2d7ea",
            c400="#7fbddc",
            c500="#4ca3ce",
            c600="#007cba",
            c700="#006394",
            c800="#004a6f",
            c900="#00314a",
            c950="#001825",
        ),
    )
) as demo:
    gr.Markdown(f"""
                # AIM-CU
                ## AIM-CU: A statistical tool for AI monitoring using cumulative sum (AIM-CU).
                """)  # noqa: F541

    gr.Markdown(f"""
                ### AIM-CU Input:
                AI output (e.g. metrics such as Accuracy, F1-score, Sensitivity etc.)
                """)  # noqa: F541

    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
                        ### Phase I:
                        Parameter choices for detecting change and detection delay estimates (theoretical calculations).
                        """)  # noqa: F541

            gr.Markdown(f"""
                ### Enter h value:
                """)  # noqa: F541

            h_phase1 = gr.Textbox(
                label="h value =",
                placeholder="h = normalized threshold, default = 4",
                value="4",
                autofocus=True,
            )

            dataframe_gt_ref_value = gr.HTML(
                label="Reference Values for an intended ARL0 with normalized threshold h",
                show_label=True,
                visible=False,
            )

            gr.Markdown(f"""
                ### Calculate reference value k for a specific value for ARL<sub>0</sub>:
                """)  # noqa: F541

            with gr.Row():
                arl_0 = gr.Textbox(
                    label="ARL_0 value =", placeholder="ARL_0", value="100"
                )

                button_calculate_k = gr.Button("Calculate k")

                output_k = gr.Textbox(label="Calculated k =", visible=False)

            dataframe_gt_ARL0 = gr.HTML(
                label="Estimate of steady state ARL (ARL_1 based on the computed reference values and intended zero-state ARL (ARL_0) with normalized threshold h)",
                show_label=True,
                visible=False,
            )

            gr.Markdown(f"""
                ### Calculate ARL<sub>1</sub> for reference value h, value k and shift in mean:
                """)  # noqa: F541

            with gr.Row():
                k_phase1 = gr.Textbox(
                    label="k value =", placeholder="k", value="0.2996"
                )
                mu1 = gr.Textbox(
                    label="Shift in mean value =",
                    placeholder="Shift in mean value",
                    value="1.2",
                )

                button_calculate_ARL_1 = gr.Button("Calculate ARL_1")

                output_ARL_1 = gr.Textbox(label="Calculated ARL_1 =", visible=False)

            button_populate_table = gr.Button(
                "Populate Reference Values and ARL_1 tables for the given h value"
            )

            gr.Markdown(f"""
                ### Upload the CSV file with metric. Or use the default example CSV file by directly clicking the button below.
                """)  # noqa: F541

            # load the CSV file with specifities across days
            csv_file_metric = gr.File(
                file_types=["csv"],
                label="Upload CSV file with metric across days",
            )
        with gr.Column():
            gr.Markdown(f"""
                        ### Phase II:
                        Performance drift detection plots, pre- and post-change distribution with respect to the performance drift detected.
                        """)  # noqa: F541

            gr.Markdown(f"""
                ### Enter required values:
                """)  # noqa: F541

            with gr.Row():
                h_phase2 = gr.Textbox(
                    label="h value =",
                    placeholder="normalized threshold, default = 4",
                    value="4",
                )

                k_phase2 = gr.Textbox(
                    label="k value =",
                    placeholder="normalized reference value, default = 0.5",
                    value="0.5",
                )

            pre_change_days = gr.Textbox(
                label="In-control days =",
                placeholder="Number of days for in-control phase, default = 60",
                value="60",
            )

            button_csv_metric = gr.Button("Show CUSUM plots")

            plot_avg_metric = gr.Plot(
                label="Average metric for the pre-change and post-change regime",
                visible=False,
            )
            plot_cusum_chart = gr.Plot(label="CUSUM Chart", visible=False)

    # Get the CSV file and populate tables
    button_populate_table.click(
        fn=populate_table,
        inputs=[h_phase1],
        outputs=[dataframe_gt_ref_value, dataframe_gt_ARL0],
    )
    button_populate_table.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=dataframe_gt_ref_value
    )
    button_populate_table.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=dataframe_gt_ARL0
    )

    # Calculate specific k for ARL_0
    button_calculate_k.click(
        fn=calculate_reference_value_k, inputs=[h_phase1, arl_0], outputs=[output_k]
    )
    button_calculate_k.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=output_k
    )

    # Calculate specific ARL_1 for value h, value k and shift in mean
    button_calculate_ARL_1.click(
        fn=calculate_arl1_h_k_mu1, inputs=[h_phase1, k_phase1, mu1], outputs=[output_ARL_1]
    )
    button_calculate_ARL_1.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=output_ARL_1
    )

    # Get the CSV file and populate plots
    button_csv_metric.click(
        fn=populate_cusum_plots,
        inputs=[csv_file_metric, k_phase2, h_phase2, pre_change_days],
        outputs=[plot_avg_metric, plot_cusum_chart],
    )

    button_csv_metric.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=plot_avg_metric
    )
    button_csv_metric.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=plot_cusum_chart
    )

try:
    path_file_config = os.path.abspath("../../config/config.toml")

    with open(path_file_config, "rb") as file_config:
        config = tomli.load(file_config)
except FileNotFoundError:
    print("Error: config.toml not found at", path_file_config)
    sys.exit(1)


if config["control"]["save_figure"] == "true":
    path_check = os.path.abspath(
        os.path.join("../../", config["path_output"]["path_figure"])
    )
    if not os.path.exists(path_check):
        os.mkdir(path_check)
        print("Created", path_check)

obj_cusum = CUSUM()
obj_cusum.initialize()

demo.launch(server_name="0.0.0.0", server_port=7860)
