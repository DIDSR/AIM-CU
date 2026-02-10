"""
Gradio user interface for AIM-CU

Disclaimer:
This software and documentation was developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.
"""

import os
import sys
import pandas as pd
import gradio as gr
import tomli
from cusum import CUSUM
from ARLTheoretical import (
    get_ref_value,
    get_ref_value_k,
    get_ARL_1,
    get_ARL_1_h_mu1_k,
    get_threshold_h,
)
from utils import (
    populate_summary_table_ARL0_k,
    populate_summary_table_ARL1_k,
)
import great_tables as gt
import plotly.graph_objects as go


def set_init_days(
    file_csv_metric: gr.File, init_days: str
) -> tuple[float, float, go.Figure]:
    """
    Set number of baseline observations and get in-control mean and standard deviation.

    Args:
        file_csv_metric (gr.File): CSV file with metric data
        init_days (str): Number of baseline observations to calculate in-control mean and standard deviation

    Returns:
        tuple[float, float, go.Figure]: In-control mean and standard deviation, and observation data plot.
    """
    init_days = int(init_days)

    data_csv_metric = pd.read_csv(file_csv_metric.name)
    obj_cusum.set_df_metric_csv(data_csv_metric)

    obj_cusum.set_init_stats(init_days=init_days)

    return (
        "{:.2f}".format(obj_cusum.in_mu),
        "{:.2f}".format(obj_cusum.in_std),
        obj_cusum.plot_input_metric_plotly_raw(),
    )


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
    summary_table_df_ARL0_k = summary_table_df_ARL0_k.applymap(
        lambda x: "{:.2f}".format(x) if isinstance(x, (float)) else x
    )

    summary_table_df_ARL1_k = get_ARL_1(
        h=h,
        shift_in_mean=config["params_cusum"]["shift_in_mean"],
        dict_ARL0_k=dict_ARL0_k,
    )
    summary_table_df_ARL1_k = summary_table_df_ARL1_k.applymap(
        lambda x: "{:.2f}".format(x) if isinstance(x, (float)) else x
    )

    return populate_summary_table_ARL0_k(
        summary_table_df_ARL0_k, h
    ), populate_summary_table_ARL1_k(summary_table_df_ARL1_k, dict_ARL0_k, h)


def calculate_reference_value_k(h: str, arl_0: str) -> tuple[str, str, str, str]:
    """
    Gets the reference value for given h and ARL_0.

    Args:
        h (str): Normalized threshold.
        arl_0 (str): ARL0 value.

    Returns:
        tuple[str, str, str, str]: Normalized reference value k (for output, k_phase1, k_phase2, h_phase2).
    """
    h = float(h)
    arl_0 = float(arl_0)

    k = get_ref_value_k(h=h, ARL_0=arl_0)
    k = "{:.2f}".format(k)

    return k, k, k, h


def calculate_threshold_h(k: str, arl_0: str) -> tuple[str, str, str, str]:
    """
    Gets the threshold h for given k and ARL_0.

    Args:
        k (str): Normalized reference value.
        arl_0 (str): ARL0 value.

    Returns:
        tuple[str, str, str, str]: Normalized threshold h (for output, h_phase1, h_phase2, k_phase2).
    """
    k_val = float(k)
    arl_0 = float(arl_0)

    h = get_threshold_h(k=k_val, ARL_0=arl_0)
    h = "{:.2f}".format(h)

    return h, h, h, k


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
    arl_1 = "{:.2f}".format(arl_1)

    return arl_1


def populate_cusum_plots(
    ref_value: str, normalized_threshold: str
) -> tuple[go.Figure, go.Figure]:
    """
    Populate CUSUM plots

    Args:
        ref_value (str): Normalized reference value for detecting a unit standard deviation change in mean of the process.
        normalized_threshold (str): Normalized threshold.

    Returns:
        tuple[go.Figure, go.Figure]: Scatter plot as Plotly graph object; CUSUM plot using Plotly graph object.
    """
    ref_value = float(ref_value)
    normalized_threshold = float(normalized_threshold)

    obj_cusum.change_detection(
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
                AI output metric (e.g. AUROC, F1-score, Sensitivity, Test Positive Rate, etc.)
                """)  # noqa: F541

    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
                        ### Initialization:
                        """)  # noqa: F541

            gr.Markdown(f"""
                ### Upload the AI output metric.
                """)  # noqa: F541

            # load the CSV file with specifities across days
            csv_file_metric = gr.File(
                label="Upload the AI output (CSV file)",
            )

            with gr.Row():
                with gr.Column():
                    init_days = gr.Textbox(
                        label="Number of baseline observations",
                        placeholder="30",
                        value="30",
                    )
                with gr.Column():
                    button_calculate_incontrol_params = gr.Button(
                        "Calculate parameters"
                    )

            with gr.Row():
                with gr.Column():
                    in_control_mean = gr.Textbox(
                        label="In-control mean", interactive=False
                    )
                with gr.Column():
                    in_control_std = gr.Textbox(
                        label="In-control standard deviation", interactive=False
                    )

            plot_observation_data = gr.Plot(label="AI output", visible=False)

            gr.Markdown(f"""
                        Parameter choices for detecting change and detection delay estimates (theoretical calculations).
                        """)  # noqa: F541

            # gr.Markdown(f"""
            #     ### Enter h value:
            #     """)  # noqa: F541

            dataframe_gt_ref_value = gr.HTML(
                label="Reference Values for an intended ARL0 with normalized threshold h",
                show_label=True,
                visible=False,
            )

            gr.Markdown(f"""
                ### Calculate parameters:
                """)  # noqa: F541

            with gr.Tabs():
                with gr.Tab("Calculate k from h"):
                    gr.Markdown(f"""
                        Calculate reference value k for specific values of h and ARL<sub>0</sub>:
                        """)  # noqa: F541

                    with gr.Row():
                        h_phase1 = gr.Textbox(
                            label="h value =",
                            placeholder="h = normalized threshold, default = 4. Range: between 4 and 5 ([4, 5])",
                            value="3",
                            autofocus=True,
                        )

                        arl_0 = gr.Textbox(
                            label="ARL₀ value =", placeholder="ARL₀", value="100"
                        )

                        button_calculate_k = gr.Button("Calculate k")

                        output_k = gr.Textbox(label="Calculated k =", visible=False)

                with gr.Tab("Calculate h from k"):
                    gr.Markdown(f"""
                        Calculate threshold h for specific values of k and ARL<sub>0</sub>:
                        """)  # noqa: F541

                    with gr.Row():
                        k_for_h = gr.Textbox(
                            label="k value =", placeholder="k", value="0.5"
                        )
                        arl_0_for_h = gr.Textbox(
                            label="ARL₀ value =", placeholder="ARL₀", value="100"
                        )

                        button_calculate_h = gr.Button("Calculate h")

                        output_h = gr.Textbox(label="Calculated h =", visible=False)

            dataframe_gt_ARL0 = gr.HTML(
                label="Estimate of steady state ARL (ARL₁ based on the computed reference values and intended zero-state ARL (ARL₀) with normalized threshold h)",
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

                # example: if std_in=0.03 and shift in mean (in original data)=0.045, then the value that the user enter will be 0.045/0.03=1.5
                # Shift in mean value is the absolute differece of in-control mean and test mean
                mu1 = gr.Textbox(
                    label="Shift in mean value (expressed in term of in-control standard deviation) =",
                    placeholder="Shift in mean value",
                    value="1.2",
                )

                button_calculate_ARL_1 = gr.Button("Calculate ARL₁")

                output_ARL_1 = gr.Textbox(label="Calculated ARL₁ =", visible=False)

            button_populate_table = gr.Button(
                "Populate Reference Values and ARL₁ tables for the given h value"
            )

            gr.Markdown(f"""
                ### Workflow:
                Phase I:
                - Upload the AI output (CSV file).
                - Enter number of baseline observations.
                - Calculate parameters.
                - Check parameter choices in Phase 1, for Phase 2. (optional)
                
                Phase II:
                - Enter h and k values.
                - Get CUSUM plots.
                """)  # noqa: F541
        with gr.Column():
            gr.Markdown(f"""
                        ### Monitoring:
                        Performance drift detection plots, pre- and post-change distribution with respect to the performance drift detected.
                        """)  # noqa: F541

            gr.Markdown(f"""
                ### Enter required values:
                """)  # noqa: F541

            with gr.Row():
                h_phase2 = gr.Textbox(
                    label="h value =",
                    placeholder="normalized threshold, default = 4. Range: between 4 and 5 ([4, 5])",
                    value="3",
                )

                k_phase2 = gr.Textbox(
                    label="k value =",
                    placeholder="normalized reference value, default = 0.5",
                    value="0.5",
                )

            button_csv_metric = gr.Button("Show CUSUM plots")

            plot_cusum_chart = gr.Plot(label="CUSUM Chart", visible=False)

            plot_avg_metric = gr.Plot(
                label="AI model performance",
                visible=False,
            )

    with gr.Row():
        table_param_description = gr.Dataframe(
            value=pd.read_csv("../../assets/params.csv"),
        )

    button_calculate_incontrol_params.click(
        fn=set_init_days,
        inputs=[csv_file_metric, init_days],
        outputs=[in_control_mean, in_control_std, plot_observation_data],
    )

    button_calculate_incontrol_params.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=plot_observation_data
    )

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

    # Calculate specific h for k and ARL_0
    button_calculate_h.click(
        fn=calculate_threshold_h,
        inputs=[k_for_h, arl_0_for_h],
        outputs=[output_h, h_phase1, h_phase2, k_phase2],
    )
    button_calculate_h.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=output_h
    )

    # Calculate specific k for ARL_0
    button_calculate_k.click(
        fn=calculate_reference_value_k,
        inputs=[h_phase1, arl_0],
        outputs=[output_k, k_phase1, k_phase2, h_phase2],
    )
    button_calculate_k.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=output_k
    )

    # Calculate specific ARL_1 for value h, value k and shift in mean
    button_calculate_ARL_1.click(
        fn=calculate_arl1_h_k_mu1,
        inputs=[h_phase1, k_phase1, mu1],
        outputs=[output_ARL_1],
    )
    button_calculate_ARL_1.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=output_ARL_1
    )

    # Get the CSV file and populate plots
    button_csv_metric.click(
        fn=populate_cusum_plots,
        inputs=[k_phase2, h_phase2],
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
