import gradio as gr
import tomli
from cusum import CUSUM
from ARLTheoretical import get_ref_value, get_ARL_1, table_4, table_5

with open("config.toml", "rb") as file_config:
    config = tomli.load(file_config)

obj_cusum = CUSUM()
obj_cusum.initialize()
obj_cusum.stats()
obj_cusum.change_detection()


def populate_table(
    h,
    k,
    mu1,
    ref_val,
    hshift_in_mean_start,
    hshift_in_mean_increament,
    hshift_in_mean_end,
):
    return table_4, table_5
    return get_ref_value(h), get_ARL_1(
        h,
        k,
        mu1,
        [ref_val],
        range(
            hshift_in_mean_start,
            hshift_in_mean_end + hshift_in_mean_increament,
            hshift_in_mean_increament,
        ),
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
            h = gr.Textbox(label="h", placeholder="Enter the value h")
            k = gr.Textbox(label="k", placeholder="Enter the value k")
            mu1 = gr.Textbox(label="mu1", placeholder="Enter the value mu1")
            ref_val = gr.Textbox(label="ref_val", placeholder="Enter the value ref_val")
            hshift_in_mean_start = gr.Textbox(
                label="shift_in_mean: start", placeholder="Enter the value"
            )
            hshift_in_mean_increament = gr.Textbox(
                label="shift_in_mean: increament", placeholder="Enter the value"
            )
            hshift_in_mean_end = gr.Textbox(
                label="shift_in_mean: end", placeholder="Enter the value"
            )

            dataftame_ref_value = gr.Dataframe(
                label="Reference Values",
                col_count=2,
                headers=["ARL0", "k"],
            )

            dataftame_ARL0 = gr.Dataframe(
                label="ARL1",
                col_count=2,
                headers=["Shift in mean", "k"],
            )

            button_populate_table = gr.Button("Populate Tables")

        with gr.Column():
            plot1 = gr.Plot(value=obj_cusum.plot_input_aucs_plotly())
            plot2 = gr.Plot(value=obj_cusum.plot_histogram_aucs_plotly())
            plot3 = gr.Plot(value=obj_cusum.plot_cusum_plotly())

    button_populate_table.click(
        fn=populate_table,
        inputs=[
            h,
            k,
            mu1,
            ref_val,
            hshift_in_mean_start,
            hshift_in_mean_increament,
            hshift_in_mean_end,
        ],
        outputs=[dataftame_ref_value, dataftame_ARL0],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
