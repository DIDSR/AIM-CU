"""
Gradio user interface for AIM-CU
"""

import os
import pandas as pd
import gradio as gr
import tomli
from cusum import CUSUM
from ARLTheoretical import get_ref_value, get_ARL_1
from utils import (
    populate_summary_table_ARL0_k,
    populate_summary_table_ARL1_k,
)

with open("config.toml", "rb") as file_config:
    config = tomli.load(file_config)

if not os.path.exists(config["path_output"]["path_figure"]):
    os.mkdir(config["path_output"]["path_figure"])

obj_cusum = CUSUM()
obj_cusum.initialize()

# Populate tables for ARL0 and ARL1 given the value of h
def populate_table(h):
    h = float(h)

    summary_table_df_ARL0_k, dict_ARL0_k = get_ref_value(
        h=h, list_ARL_0=[50, 100, 150, 200, 300, 400, 500, 1000]
    )

    summary_table_df_ARL1_k = get_ARL_1(
        h=h,
        shift_in_mean=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
        ],
        dict_ARL0_k=dict_ARL0_k,
    )

    return populate_summary_table_ARL0_k(
        summary_table_df_ARL0_k
    ), populate_summary_table_ARL1_k(summary_table_df_ARL1_k, dict_ARL0_k)


# Populate CUSUM plots
def populate_cusum_plots(file_csv_specificity):
    if file_csv_specificity is not None:
        # upload CSV file
        data_csv_specificity = pd.read_csv(file_csv_specificity.name)
        obj_cusum.set_df_spec_csv(data_csv_specificity)
    else:
        # use the example CSV data
        obj_cusum.set_df_spec_default()

    obj_cusum.change_detection()

    return (obj_cusum.plot_input_specificities_plotly(), obj_cusum.plot_cusum_plotly())


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
    format = '<a href="{link}">{text}</a>'
    text_with_link = format.format

    gr.Markdown(f"""
                # AIM-CU: A CUSUM-based tool for AI Monitoring
                """)  # noqa: F541

    gr.Markdown(f"""
                ### AIM-CU is a statistical tool for AI monitoring using cumulative sum (AIM-CU). AIM-CU computes:
                * the parameter choices for change-point detection based on an acceptable false alarm rate
                * detection delay estimates for a given displacement of the performance metric from the target for those parameter choices.
                """)

    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
                        ## Phase I: Initialization
                        """)

            with gr.Row():
                with gr.Column():
                    h = gr.Textbox(
                        label="h value =",
                        placeholder="h = normalized threshold, default = 4",
                    )
                with gr.Column():
                    k = gr.Textbox(
                        label="k value =",
                        placeholder="k = reference value, default = 0.5",
                    )

            dataframe_gt_ref_value = gr.HTML(
                label="Reference Values for an intended ARL0 with normalized threshold h",
                show_label=True,
                visible=False,
            )
            dataframe_gt_ARL0 = gr.HTML(
                label="Estimate of steady state ARL (ARL1 based on the computed reference values and intended zero-state ARL (ARL0) with normalized threshold h)",
                show_label=True,
                visible=False,
            )

            button_populate_table = gr.Button(
                "Populate Reference Values and ARL1 tables for the given h value"
            )

        with gr.Column():
            gr.Markdown(f"""
                        ## Phase II: Monitoring
                        """)
            gr.Markdown(f"""
                Upload the CSV file with specificities. Or use the default example CSV file by directly clicking the button below.
                """)  # noqa: F541
            # load the CSV file with specifities across days
            csv_file_specificity = gr.File(
                file_types=["csv"],
                label="Upload CSV file with specificities across days",
            )
            button_csv_specificity = gr.Button("Show CUSUM plots")

            plot_avg_specificity = gr.Plot(
                label="Average Specificities for the pre-change and post-change regime",
                visible=False,
            )
            plot_cusum_chart = gr.Plot(label="CUSUM Chart", visible=False)

            # details about the tool
            gr.Markdown(f"""
                        ### Potential users who are concerned about safe and reliable medical AI tools:
                        * AI developers
                        * Healthcare professionals
                        * Patients
                        * Regulators
                        * Policymakers
                        """)

    # Get the CSV file and populate tables
    button_populate_table.click(
        fn=populate_table,
        inputs=[h],
        outputs=[dataframe_gt_ref_value, dataframe_gt_ARL0],
    )
    button_populate_table.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=dataframe_gt_ref_value
    )
    button_populate_table.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=dataframe_gt_ARL0
    )

    # Get the CSV file and populate plots
    button_csv_specificity.click(
        fn=populate_cusum_plots,
        inputs=[csv_file_specificity],
        outputs=[plot_avg_specificity, plot_cusum_chart],
    )

    button_csv_specificity.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=plot_avg_specificity
    )
    button_csv_specificity.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=plot_cusum_chart
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
