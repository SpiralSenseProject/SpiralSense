import gradio as gr
import predict as predict


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def process_file(webcam_filepath, upload_filepath):
    result = []
    if webcam_filepath == None:
        sorted_classes = predict.predict_image(upload_filepath)
        for class_label, class_prob in sorted_classes:
            class_prob = class_prob.item().__round__(2)
            result.append(f"{class_label}: {class_prob}%")
        return result
    elif upload_filepath == None:
        sorted_classes = predict.predict_image(webcam_filepath)
        for class_label, class_prob in sorted_classes:
            class_prob = class_prob.item().__round__(2)
            result.append(f"{class_label}: {class_prob}%")
        return result
    else:
        sorted_classes = predict.predict_image(upload_filepath)
        for class_label, class_prob in sorted_classes:
            class_prob = class_prob.item().__round__(2)
            result.append(f"{class_label}: {class_prob}%")
        return result


demo = gr.Interface(
    theme='gradio/soft',
    fn=process_file,
    title="HANDETECT",
    description="An innovative AI-powered system that facilitates early detection and monitoring of movement disorders through handwriting assessment",
    inputs=[
        gr.inputs.Image(
            source="upload", type="filepath", label="Choose Image"
        ),
    ],
    outputs=[
        gr.outputs.Textbox(label="Prediction 1"),
        gr.outputs.Textbox(label="Prediction 2"),
        gr.outputs.Textbox(label="Prediction 3"),
    ],
)

demo.launch()
