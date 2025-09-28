import gradio as gr
import spaces
from definers import (
    install_ffmpeg,
    apt_install, pip_install,
    google_drive_download,
    train, infer,
    css, theme
)

install_ffmpeg()
apt_install()

google_drive_download("1jkmEQxSHYff05opUQ4czyDNj-2uQIK68", "faiss-1.12.0-py3-none-manylinux_2_35_x86_64.whl", False)
pip_install("faiss-1.12.0-py3-none-manylinux_2_35_x86_64.whl")

pip_install("numpy==1.26.4")

@spaces.GPU(duration=240)
def handle_training(
    features, labels, model_path, remote_src, dataset_label_columns, revision, url_type, drop_list, selected_rows
):
    return train(
        model_path,
        remote_src, revision, url_type,
        features, labels,
        dataset_label_columns, drop_list,
        selected_rows
    )

@spaces.GPU(duration=180)
def handle_prediction(pred,model):
    return infer(model, pred)

def main():
    with gr.Blocks(theme=theme(), css=css()) as app:
            gr.Markdown("# Model Training and Prediction")

            with gr.Tabs():

                with gr.TabItem("Train"):
                    with gr.Row():
                        with gr.Column():
                            model_train = gr.File(label="Upload Model (for re-training)")
                            remote = gr.Textbox(placeholder="Remote Dataset", label="HuggingFace name or URL")
                            revision = gr.Textbox(placeholder="Dataset Revision", label="Revision")
                            tpe = gr.Dropdown(label="Remote Dataset Type", choices=["parquet", "json", "csv", "arrow", "webdataset", "txt"])
                            drop_list = gr.Textbox(placeholder="Ignored Columns (semi-colon separated)", label="Drop List")
                            label_columns = gr.Textbox(placeholder="Label Columns (semi-colon separated)", label="Label Columns")
                            selected_rows = gr.Textbox(placeholder="Single rows and ranges (space separated, use a hyphen to choose a range or rows)", label="Selected Rows")

                        with gr.Column():
                            local_features = gr.File(label="Local Features",file_count="multiple",allow_reordering=True)
                            local_labels = gr.File(label="Local Labels (for supervised training)",file_count="multiple",allow_reordering=True)
                            train_button = gr.Button("Train", elem_classes="btn")
                            train_output = gr.File(label="Trained Model Output")

                    train_button.click(
                        fn=handle_training,
                        inputs=[local_features, local_labels, model_train, remote, label_columns, revision, tpe, drop_list, selected_rows],
                        outputs=[train_output]
                    )

                with gr.TabItem("Predict"):
                    with gr.Row():
                        with gr.Column():
                            model_predict = gr.File(label="Upload Model (for prediction)")
                            prediction_data = gr.File(label="Prediction Data")

                        with gr.Column():
                            predict_button = gr.Button("Predict", elem_classes="btn")
                            predict_output = gr.File(label="Prediction Output")

                    predict_button.click(
                        fn=handle_prediction,
                        inputs=[prediction_data, model_predict],
                        outputs=[predict_output]
                    )

    app.queue().launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
