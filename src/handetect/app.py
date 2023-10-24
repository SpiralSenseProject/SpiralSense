import gradio as gr
import predict as predict
import extract_gradcam as extract_gradcam
import extract_lime as extract_lime


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def process_file(
    upload_filepath,
    gradcam_toggle,
    lime_toggle,
):
    print("Upload filepath:", upload_filepath)
    print("GradCAM toggle:", gradcam_toggle)
    print("LIME toggle:", lime_toggle)
    result = []
    sorted_classes = predict.predict_image(upload_filepath)
    for class_label, class_prob in sorted_classes:
        class_prob = class_prob.item().__round__(2)
        result.append(f"{class_label}: {class_prob}%")
    result = result[:4]
    if gradcam_toggle == True:
        cam = extract_gradcam.extract_gradcam(upload_filepath, save_path="gradcam.jpg")
        result.append("gradcam.jpg")
    else:
        result.append(None)
    if lime_toggle == True:
        lime = extract_lime.generate_lime(upload_filepath, save_path="lime.jpg")
        result.append("lime.jpg")
    else:
        result.append(None)
    return result
    # else:
    #     sorted_classes = predict.predict_image(upload_filepath)
    #     for class_label, class_prob in sorted_classes:
    #         class_prob = class_prob.item().__round__(2)
    #         result.append(f"{class_label}: {class_prob}%")
    #     result = result[:4]
    #     if gradcam_toggle == 1:
    #         cam = extract.extract_gradcam(upload_filepath, save_path="gradcam.jpg")
    #         result.append("gradcam.jpg")
    #     if lime_toggle == 1:
    #         lime = lime_eval.generate_lime(upload_filepath, save_path="lime.jpg")
    #         result.append("lime.jpg")
    #     return result


# Prerun to innitialize the model
# process_file(None, r"data\test\Task 1\Dystonia\0c08d2ea-8e1c-4ac6-92db-a752388b30cf.png")

css = """
.block {
    margin-left: auto;
    margin-right: auto;
    width: 100%;
}
#image_input {
    width: 300px !important;
    height: 300px !important;
}
#image_input img {
    width: 300px !important;
    height: 300px !important;
}
.output-image {
    width: 70% !important;
    text-align: -webkit-center !important;
}
.output-image img {
    width:  300px !important;
}
.toggle {
    width: 17% !important;
}
.show-api {
    visibility: hidden !important;
}

.built-with {
    visibility: hidden !important;
}

#title-label {
    font-size: 35px !important;
    text-align: -webkit-center !important;
    margin-block-end: -55px;
}
#desc-label {
    font-size: 15px !important;
    text-align: -webkit-center !important;
}

.output-class.svelte-75gm11.svelte-75gm11.svelte-75gm11 {
    font-size: unset !important;
}

"""

block = gr.Blocks(title="SpiralSense", css=css, theme="gradio/soft")

block.queue()

with block as demo:
    with gr.Column():
        gr.Label("SpiralSense", elem_id="title-label", show_label=False)
        gr.Label(
            "Cost-Effective, Portable And Stressless Spiral Drawing Analysing Web Application for Early Detection of Multiple Neurological Disorders with 96% Accuracy",
            elem_id="desc-label",
            show_label=False
        )
        # gr.Markdown(
        #     """
        #     <h1 style="text-align: center;">SpiralSense</h1>
        #     <h4 style="text-align: center;">Cost-Effective, Portable And Stressless Spiral Drawing Analysing Web Application for Early Detection of Multiple Neurological Disorders with 96% Accuracy</h4>
        #     """
        # )
        # gr.Markdown(
        #     """
        #     <h4 style="text-align: center;">------------------------------------------</h4>
        #     """
        # )
        with gr.Row():
            image_input = gr.Image(
                type="filepath",
                label="Choose Image",
                source="upload",
                elem_id="image_input",
            )
            with gr.Column():
                gr.Markdown(
                    """
                    <h4>Feature Explanations</h4>
                    """
                )
                gradcam_toggle = gr.Checkbox(label="GradCAM++")
                lime_toggle = gr.Checkbox(label="LIME")
                warning_of_slow = gr.Label(
                    "Warning: Feature Explanation may take a very long time to load.",
                    elem_id="warning_of_slow",
                    color="red",
                    show_label=False,
                )
        with gr.Row():
            submit_button = gr.Button(value="Submit")
        gr.Markdown("<br>")
            # cancel_button = gr.Button(value="Cancel")
        # theme="gradio/soft",
        # fn=process_file,
        # title="HANDETECT",
        # outputs=[
        with gr.Row():
            prob1_textbox = gr.outputs.Textbox(label="Probability 1")
            prob2_textbox = gr.outputs.Textbox(label="Probability 2")
            prob3_textbox = gr.outputs.Textbox(label="Probability 3")
            prob4_textbox = gr.outputs.Textbox(label="Probability 4")
            # GradCAM
        with gr.Row():
            gradcam_output = gr.Image(
                label="GradCAM++",
                type="filepath",
                elem_classes=["output-image"],
            )
            lime_output = gr.Image(
                label="LIME",
                type="filepath",
                elem_classes=["output-image"],
            )

        submit_button.click(
            process_file,
            [image_input, gradcam_toggle, lime_toggle],
            [
                prob1_textbox,
                prob2_textbox,
                prob3_textbox,
                prob4_textbox,
                gradcam_output,
                lime_output,
            ],
            show_progress="minimal",
            preprocess=upload_file,
            scroll_to_output=True,
            # cancels=[cancel_button],
        )


demo.launch()
