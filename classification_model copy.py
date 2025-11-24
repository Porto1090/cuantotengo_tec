import gradio as gr
import cv2
import numpy as np
from inference.main_inference import run_inference  # adapt this import

def process(image):
    # image is a numpy array (RGB)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # run your full pipeline
    result, annotated = run_inference(img_bgr, sender_phone=None)

    # convert annotated back to RGB for display
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return result, annotated_rgb

gr.Interface(
    fn=process,
    inputs=gr.Image(type="numpy", label="Upload Shelf Image"),
    outputs=[
        gr.JSON(label="Brand Counts & Output"),
        gr.Image(label="Annotated Image")
    ],
    title="CuantoTengo Shelf Counter",
    description="Upload a photo of the shelf and the algorithm will detect bottles, align columns, and count each brand."
).launch()

