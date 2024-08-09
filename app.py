import gradio as gr
import tomli
from cusum import CUSUM

with open("config.toml", "rb") as file_config:
    config = tomli.load(file_config)

obj_cusum = CUSUM()
obj_cusum.initialize()
obj_cusum.stats()
obj_cusum.change_detection()

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
                {text_with_link(link='mailto:Ravi.Samala@fda.hhs.gov', text='Ravi Samala')} and {text_with_link(link='mailto:Smriti.Prathapan@fda.hhs.gov', text='Smriti Prathapan')}; DIDSR/OSEL/CDRH/FDA.
                """)

    with gr.Row():
        with gr.Column():
            intro = gr.Markdown(f"""
                           # Introduction:
                           In this study, we investigate how a change in the performance of an AI model caused by an abrupt data drift can be detected by using a cumulative sum (CUSUM) control chart
                           
                            CUSUM, which is a well-studied statistical process control method, can easily be adapted for monitoring the performance of AI models targeted for assisting in medical diagnosis. We demonstrate that the sensitivity of CUSUM can be controlled to balance between the mean time between false alarms versus the delay in detecting a true change in performance. The analysis method proposed in this study to monitor the performance of AI models applied to cancer detection on screening mammography may be generalized to practical situations where case-based information about whether a patient is cancer-positive or cancer-negative is not known.
                           """)
            
            plot1 = gr.Plot(value=obj_cusum.plot_histogram_plotly(obj_cusum.AvgDD, "ADD"))
            plot2 = gr.Plot(value=obj_cusum.plot_histogram_plotly(obj_cusum.h_1000, "H"))
            plot3 = gr.Plot(value=obj_cusum.plot_histogram_plotly(obj_cusum.k_1000, "K"))

        with gr.Column():
            plot4 = gr.Plot(value=obj_cusum.plot_input_aucs_plotly())
            plot5 = gr.Plot(value=obj_cusum.plot_histogram_aucs_plotly())
            plot6 = gr.Plot(value=obj_cusum.plot_cusum_plotly())

demo.launch()