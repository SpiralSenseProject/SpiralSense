import gradio as gr
import predict as predict
import extract as extract

def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def process_file(webcam_filepath, upload_filepath, ):
    result = []
    if webcam_filepath == None:
        sorted_classes = predict.predict_image(upload_filepath)
        for class_label, class_prob in sorted_classes:
            class_prob = class_prob.item().__round__(2)
            result.append(f"{class_label}: {class_prob}%")
        cam = extract.extract_gradcam(upload_filepath, save_path="gradcam.jpg")
        result = result[:3]
        result.append("gradcam.jpg")
        return result
    elif upload_filepath == None:
        sorted_classes = predict.predict_image(webcam_filepath)
        for class_label, class_prob in sorted_classes:
            class_prob = class_prob.item().__round__(2)
            result.append(f"{class_label}: {class_prob}%")
        cam = extract.extract_gradcam(webcam_filepath, save_path="gradcam.jpg")
        result = result[:3]
        result.append("gradcam.jpg")
        return result
    else:
        sorted_classes = predict.predict_image(upload_filepath)
        for class_label, class_prob in sorted_classes:
            class_prob = class_prob.item().__round__(2)
            result.append(f"{class_label}: {class_prob}%")
        cam = extract.extract_gradcam(upload_filepath, save_path="gradcam.jpg")
        # Only keep the first 3 results
        result = result[:3]
        result.append("gradcam.jpg")
        return result



demo = gr.Interface(
    theme="gradio/soft",
    fn=process_file,
    title="HANDETECT",
    
    inputs=[
        gr.components.Image(type="filepath", label="Choose Image", source="upload"),
    ],
    outputs=[
        gr.outputs.Textbox(label="Probability 1"),
        gr.outputs.Textbox(label="Probability 2"),
        gr.outputs.Textbox(label="Probability 3"),
        # GradCAM
        gr.outputs.Image(label="GradCAM++", type="filepath"),
        
        
    ],
    
)

demo.launch()
