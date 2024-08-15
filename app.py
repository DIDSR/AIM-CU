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
obj_cusum.stats()
obj_cusum.change_detection()


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
            h = gr.Textbox(label="h", placeholder="Enter the value h")
            # k = gr.Textbox(label="k", placeholder="Enter the value k")
            # mu1 = gr.Textbox(label="mu1", placeholder="Enter the value mu1")
            # ref_val = gr.Textbox(label="ref_val", placeholder="Enter the value ref_val")
            # hshift_in_mean_start = gr.Textbox(
            #     label="shift_in_mean: start", placeholder="Enter the value"
            # )
            # hshift_in_mean_increament = gr.Textbox(
            #     label="shift_in_mean: increament", placeholder="Enter the value"
            # )
            # hshift_in_mean_end = gr.Textbox(
            #     label="shift_in_mean: end", placeholder="Enter the value"
            # )

            dataframe_gt_ref_value = gr.HTML(label="ARL0")
            dataframe_gt_ARL0 = gr.HTML(label="ARL1")

            button_populate_table = gr.Button("Populate Tables")

        with gr.Column():
            plot1 = gr.Plot(value=obj_cusum.plot_input_aucs_plotly(), label='Average Specificities for the pre-change and post-change regime')
            plot2 = gr.Plot(value=obj_cusum.plot_histogram_aucs_plotly(), label='Histograms for the pre- and post-change specificity')
            plot3 = gr.Plot(value=obj_cusum.plot_cusum_plotly(), label='CUSUM Chart')

    button_populate_table.click(
        fn=populate_table,
        inputs=[h],
        outputs=[dataframe_gt_ref_value, dataframe_gt_ARL0],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
