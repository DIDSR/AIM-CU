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

    return (
        obj_cusum.plot_input_specificities_plotly(),
        obj_cusum.plot_cusum_plotly()
    )


with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50=config["color"]["blue_005"],
            c100=config["color"]["blue_005"],
            c200=config["color"]["blue_020"],
            c300=config["color"]["blue_020"],
            c400=config["color"]["blue_040"],
            c500=config["color"]["blue_040"],
            c600=config["color"]["blue_060"],
            c700=config["color"]["blue_060"],
            c800=config["color"]["blue_080"],
            c900=config["color"]["blue_080"],
            c950=config["color"]["blue_100"],
        ),
        secondary_hue=gr.themes.Color(
            c50=config["color"]["blue_005"],
            c100=config["color"]["blue_005"],
            c200=config["color"]["blue_020"],
            c300=config["color"]["blue_020"],
            c400=config["color"]["blue_040"],
            c500=config["color"]["blue_040"],
            c600=config["color"]["blue_060"],
            c700=config["color"]["blue_060"],
            c800=config["color"]["blue_080"],
            c900=config["color"]["blue_080"],
            c950=config["color"]["blue_100"],
        ),
    ).set(
        button_primary_background_fill="*primary_200",
        button_primary_background_fill_hover="*primary_100",
    )
) as demo:
    format = '<a href="{link}">{text}</a>'
    text_with_link = format.format

    gr.Markdown(f"""
                # AIM-CU: A CUSUM-based tool for AI Monitoring
                """)  # noqa: F541

    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
                h is referred to as the normalized threshold. In typical applications, h may default to 4.
                """)  # noqa: F541
            h = gr.Textbox(
                label="h value",
                placeholder="h is referred to as the normalized threshold. In typical applications, h may default to 4.",
            )

            dataframe_gt_ref_value = gr.HTML(label="Reference Values for an intended ARL0 with normalized threshold h", show_label=True, visible=False)
            dataframe_gt_ARL0 = gr.HTML(label="Estimate of steady state ARL (ARL1 based on the computed reference values and intended zero-state ARL (ARL0) with normalized threshold h)", show_label=True, visible=False)

            button_populate_table = gr.Button(
                "Populate Reference Values and ARL1 tables for the given h value"
            )

        with gr.Column():
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
            # plot_histogram = gr.Plot(
            #     label="Histograms for the pre- and post-change specificity",
            #     visible=False,
            # )
            plot_cusum_chart = gr.Plot(label="CUSUM Chart", visible=False)

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
    # button_csv_specificity.click(
    #     fn=lambda: gr.update(visible=True), inputs=[], outputs=plot_histogram
    # )
    button_csv_specificity.click(
        fn=lambda: gr.update(visible=True), inputs=[], outputs=plot_cusum_chart
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
