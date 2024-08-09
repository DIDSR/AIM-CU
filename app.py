import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt


# load CSV file or use the example, and prepare plots
def process_plot_csv(file, dataframe):
    if file is not None:
        # upload CSV file
        df = pd.read_csv(file.name)
    else:
        # use the example CSV data
        df = dataframe

    # convert the data numeric (required when dataframe is edited)
    df = df.map(lambda x: pd.to_numeric(x))

    list_figure = []

    plt.figure(figsize=(10, 10))
    plt.plot(df[df.columns[0]], df[df.columns[1]])
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title("plot_1")
    plt.savefig("plot_1.png")
    plt.close()
    list_figure.append("plot_1.png")

    plt.figure(figsize=(10, 10))
    plt.bar(df[df.columns[0]], df[df.columns[1]])
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title("plot_2")
    plt.savefig("plot_2.png")
    plt.close()
    list_figure.append("plot_2.png")

    return df, list_figure


df_example = pd.read_csv("example.csv")

# theme/colors
color_theme = "#d0e4f0"
color_button = "#4F96C4"
color_title = "#007CBA"

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#F3F9FC",
            c100="#F3F9FC",
            c200="#D0E4F0",
            c300="#D0E4F0",
            c400="#A3CAE1",
            c500="#A3CAE1",
            c600="#77B0D2",
            c700="#77B0D2",
            c800="#4F96C4",
            c900="#4F96C4",
            c950="#007CBA",
        ),
        secondary_hue=gr.themes.Color(
            c50="#F3F9FC",
            c100="#F3F9FC",
            c200="#D0E4F0",
            c300="#D0E4F0",
            c400="#A3CAE1",
            c500="#A3CAE1",
            c600="#77B0D2",
            c700="#77B0D2",
            c800="#4F96C4",
            c900="#4F96C4",
            c950="#007CBA",
        ),
    ).set(
    button_primary_background_fill="*primary_200",
    button_primary_background_fill_hover="*primary_100",
)
) as demo:
    with gr.Row():
        with gr.Column():
            csv_file = gr.File(file_types=["csv"], label="Upload CSV File")
            editable_dataframe = gr.Dataframe(
                value=df_example,
                label="Editable Dataframe",
                col_count=2,
                headers=["x", "y"],
            )
            submit_button = gr.Button('Submit')

        with gr.Column():
            selected_data = gr.Dataframe(
                value=df_example, label="Selected Data", col_count=2, headers=["x", "y"]
            )
            output_gallery = gr.Gallery(label="Outputs")

    submit_button.click(
        fn=process_plot_csv,
        inputs=[csv_file, editable_dataframe],
        outputs=[selected_data, output_gallery]
    )

demo.launch()