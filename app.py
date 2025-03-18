import torch
import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image

MODEL_PATH = "best.pt"  
model = torch.hub.load("yolov5", "custom", path=MODEL_PATH, source="local")

class_names = ["RBC", "WBC", "Platelets"]

def predict(image):
    results = model(image)  
    boxes = results.xyxy[0].cpu().numpy()  
    confs = boxes[:, 4]  
    cls_ids = boxes[:, 5].astype(int)  
    processed_image = Image.fromarray(results.render()[0])

    total_detections = len(cls_ids)
    class_counts = {name: 0 for name in class_names}
    class_precision = {name: 0 for name in class_names}

    for cls_id, conf in zip(cls_ids, confs):
        class_name = class_names[cls_id]
        class_counts[class_name] += 1
        class_precision[class_name] += conf  

    for key in class_precision.keys():
        if class_counts[key] > 0:
            class_precision[key] /= class_counts[key]

    df = pd.DataFrame({
        "Class": list(class_counts.keys()),
        "Detections": list(class_counts.values()),
        "Precision": list(class_precision.values()),
    })
    
    return processed_image, df

custom_css = """
h1 {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #FF5733;
    margin-bottom: 10px;
}
.gradio-container {
    background: linear-gradient(to right, #f7f7f7, #ffffff);
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
}
.gr-button {
    background-color: #FF5733 !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    transition: 0.3s ease-in-out;
}
.gr-button:hover {
    background-color: #E74C3C !important;
    transform: scale(1.05);
}
.gr-input {
    border: 2px solid #FF5733 !important;
    border-radius: 8px !important;
}
.gr-image {
    border: 2px solid #ddd !important;
    border-radius: 10px;
    padding: 5px;
}
.gr-dataframe {
    font-size: 15px !important;
    border-radius: 8px !important;
    overflow: hidden;
    box-shadow: 0px 3px 8px rgba(0, 0, 0, 0.1);
}
"""

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Dataframe()],
    title=" YOLOv5 Blood Cell Detection with Precision & Recall ",
    description="Upload an image to detect **RBC, WBC, and Platelets**. The table on right shows precision per class.",
    theme="default",
    css=custom_css
)

app.launch(share=True)
