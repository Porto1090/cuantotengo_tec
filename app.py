import os
import re
import cv2
import uuid
import time
import string
import secrets
import numpy as np
import gradio as gr
import pandas as pd
import azure_blob_storage as azure_bs
from datetime import datetime, timezone
from inference.main_inference import InferencePipeline
from inference.config import (
    MX_CLASS_NAMES_DICT, LAB_CLASS_NAMES_DICT,
)

import base64
from PIL import Image
from io import BytesIO

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# PLACEHOLDER:
dummy_df = pd.DataFrame({"MARCA": [], "TOTAL": []})
SESSION_REGEX = re.compile(r"^[0-9]{1,3}$")

from dotenv import load_dotenv
load_dotenv()

BRAND_DETECTION_VERSION = os.getenv("BRAND_DETECTION_VERSION")

BRAND_MAP = {
    **MX_CLASS_NAMES_DICT,
    **LAB_CLASS_NAMES_DICT,
}

# === HELPERS ===
def generate_short_hex():
    letters = [secrets.choice(string.ascii_uppercase) for _ in range(2)]
    numbers = [secrets.choice(string.digits) for _ in range(4)]
    return "".join(letters + numbers)
    #return uuid.uuid4().hex[:6].upper()

def format_output_for_df(brand_totals):
    rows = []
    for key, value in brand_totals.items():
        pretty_name = BRAND_MAP.get(key, key)
        rows.append({"MARCA": pretty_name or key, "TOTAL": value})
        print(f"Marca: {pretty_name}, Total: {value}")
    
        df = pd.DataFrame(rows)
        df = df.sort_values(by="MARCA").reset_index(drop=True)
    return df

def format_output_html(brand_totals):
    rows = ""
    bg_colors = ["#222222", "#333333"]

    sorted_items = sorted(
        brand_totals.items(),
        key=lambda x: BRAND_MAP.get(x[0], x[0])
    )

    for idx, (key, value) in enumerate(sorted_items):
        transformed_key = "_".join(key.split('_')[1:])
        print(f"Transformed key: {transformed_key}")
        pretty_name = BRAND_MAP.get(transformed_key, key.split("_", 1)[-1].replace("_", " ").title())
        img_path = f"./images/{transformed_key}.jpg"

        # Convertir imagen local a base64
        try:
            img_base64 = image_to_base64(img_path)
            img_src = f"data:image/jpeg;base64,{img_base64}"
        except:
            # Fallback a una URL por defecto si falla
            img_src = "https://via.placeholder.com/64"
    
        bg_color = bg_colors[idx % 2]

        rows += f"""
        <tr style="background-color:{bg_color}; border-bottom:1px solid #e5e7eb;">
            <td style="background-color:#FFFFFF; margin: auto; text-align:center; padding:10px; width:80px;">
                <img src="{img_src}" color="black" width="64" height="64" style="object-fit: contain; align:center;" alt="NO IMAGE" />
            </td>
            <td style="padding:8px; text-align:left; font-size:24px; height:80px;">
                {pretty_name}
            </td>
            <td style="padding:8px; font-weight:600; text-align:center; font-size:24px;">
                {value}
            </td>
        </tr>
        """
        print(f"Marca: {pretty_name}, Total: {value}")

    return f"""
    <table style="
        width:100%;
        border-collapse: collapse;
        font-size: 16px;
        background-color:#FFFFFF;
    ">
        <thead>
            <tr style="background-color:#000000; color:white;">
                <th style="padding:12px; width:80px;"></th>
                <th style="padding:12px; text-align:left;">MARCA</th>
                <th style="padding:12px; text-align:center;">TOTAL</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """

def preprocess_image(image_bytes, max_size=1024, jpeg_quality=70):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Error: No se logró decodificar la imagen.")

    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, jpg = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR)

# === MAIN PROCESSING FUNCTION ===
def process(file, session_id, enter_time, progress=gr.Progress()):
    print(f"\nSession ID: {session_id}")
    print(f"Algorithm version: {BRAND_DETECTION_VERSION}")
    elapsed_time = None
    if enter_time is not None:
        now = datetime.now(timezone.utc)
        elapsed_time = (now - enter_time).total_seconds()
    progress(0, desc="Leyendo imagen...")

    try:
        if isinstance(file, dict):
            image_bytes = file["data"]
        else:
            with open(file, "rb") as f:
                image_bytes = f.read()

        progress(0.2, desc="Preprocesando imagen...")
        img_bgr = preprocess_image(image_bytes)
    except Exception:
        print("Failed to decode the image.")
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True, value="# **Error:** No se pudo decodificar la imagen. Vuelve a intentarlo."),
        )
        
    progress(0.6, desc="Detectando productos...")
    pipeline = InferencePipeline()
    brand_totals, annotated, cap_data, front_bottles, bottle_brand_mapping, lane_totals, processing_time = pipeline.run(img_bgr)

    if not brand_totals:
        print("No products detected in the image.")
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True, value="# **Error:** No se detectaron productos en la imagen. Vuelve a intentarlo."),
        )
        
    progress(0.9, desc="Preparando resultados...")

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    real_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()[:-6]
    # === SAVE TO AZURE BLOB STORAGE ===
    # SAVE ORIGINAL IMAGE
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    azure_bs.save_image_to_blob(img_rgb, session_id, real_timestamp + "_original", BRAND_DETECTION_VERSION + "_images")
    # SAVE ALGORITHM IMAGE
    azure_bs.save_image_to_blob(annotated_rgb, session_id, real_timestamp + "_bounding_boxes", BRAND_DETECTION_VERSION + "_images")
    # SAVE ALGORITHM LOG
    log_dict = {
        "session_id": session_id,
        "version": BRAND_DETECTION_VERSION,
        "timestamp": real_timestamp,
        "brand_totals": brand_totals,
        "processing_time": f"{processing_time:.2f}",
        "elapsed_time_in_session": f"{elapsed_time:.2f}",
    }
    azure_bs.save_log_to_blob(log_dict, session_id, real_timestamp, BRAND_DETECTION_VERSION + "_logs")
    
    print(f"Timestamp Photo Session: {log_dict['elapsed_time_in_session']} seconds")
    print(f"Process Timestamp: {log_dict['processing_time']} seconds")

    SHOW_IMAGES = True
    if (SHOW_IMAGES):
        html_table = format_output_html(brand_totals)
        img_url = azure_bs.get_blob_url(session_id, real_timestamp + "_bounding_boxes", BRAND_DETECTION_VERSION + "_images")
        time.sleep(0.5)
        progress(1.0, desc="Completado")
        return html_table, img_url, gr.update(visible=False)
    else: 
        df = format_output_for_df(brand_totals)
        img_url = azure_bs.get_blob_url(session_id, real_timestamp + "_bounding_boxes", BRAND_DETECTION_VERSION + "_images")
        time.sleep(0.5)
        progress(1.0, desc="Completado")
        return df, img_url, gr.update(visible=False)

# === SESSION CONTROLLERS ===
def set_session_id(user_input):
    if not user_input:
        return (
            None, # session_id
            gr.update(visible=True, value="ID requerido"), # session_error
            gr.update(visible=False),  # dashboard
            gr.update(visible=True),  # gate
            gr.update(value="", visible=False),  # session_text
            None  # enter_time = None
        )

    sid = user_input.strip().upper()

    if not SESSION_REGEX.match(sid):
        return (
            None, # session_id
            gr.update(visible=True, value="ID inválido (1-999)"), # session_error
            gr.update(visible=False),  # dashboard
            gr.update(visible=True),  # gate
            gr.update(value="", visible=False),  # session_text
            None  # enter_time = None
        )
        
    # timestamp for session start
    timestamp = datetime.now(timezone.utc)

    return (
        sid, # session_id
        gr.update(visible=False), # session_error
        gr.update(visible=True),   # mostrar dashboard
        gr.update(visible=False),  # gate
        gr.update(value="", visible=True),  # session_text
        timestamp  # enter_time
    )
    
def render_session_text(session_id):
    return (
        gr.update(value=f"## ID Sesión: **{session_id if session_id != '0' else 'TEST'}**", visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )
    
# === UI HELPERS ===
def reset_ui():
    return (
        None,  # file_input
        gr.update(visible=False),  # output_json
        gr.update(visible=False),  # output_img
        gr.update(visible=True, value=""),  # exception_message
        gr.update(visible=False),  # reset_btn
    )
    
def on_file_upload(file):
    if file is None:
        return (
            gr.update(visible=True), # file_input
            gr.update(visible=False), # output_json
            gr.update(visible=False), # output_img
            gr.update(visible=False),  # reset_btn
        )
    else:
        return (
            gr.update(visible=False), # file_input
            gr.update(visible=True), # output_json
            gr.update(visible=True), # output_img
            gr.update(visible=True),  # reset_btn
        )

with gr.Blocks(title="CuantoTengo") as main:
    session_id = gr.State(value=None)
    enter_time = gr.State(value=None)
    
    gr.Markdown(f"<div style='display:flex; align-items:center; justify-content:space-between;'>\
        <h1 style='font-size: 40px; color:#F59E0B'>CuantoTengo</h1> \
        <h3 style='font-size: 16px; color:#FFF'><b>BY LIFT LAB</b></h3>\
    </div>")
    
    #=== SESSION GATE ===
    with gr.Column(visible=True) as session_gate:
        gate_title= gr.Markdown("## Ingresa tu ID de usuario")
        session_input = gr.Textbox(
            label="ID de sesión",
            placeholder="EJEMPLO: 000",
        )

        session_error = gr.Markdown(visible=False)
        enter_btn = gr.Button("ENTRAR", size="lg")
        test_btn = gr.Button("ENTRAR COMO TEST", size="lg")

    #=== MAIN APP ===
    with gr.Column(visible=False) as dashboard:
        session_text = gr.Markdown(visible=False)

        # === INPUTS ===
        file_input = gr.File(label="PULSA AQUÍ PARA TOMAR LA FOTO", file_types=["image"])
        # === OUTPUTS ===
        reset_btn = gr.Button(
            "TOMAR OTRA FOTO",
            visible=False,
            size="lg"
        )
        # output_json = gr.DataFrame(value=dummy_df, visible=False)
        output_json = gr.HTML(visible=False)
        output_img = gr.Image(label="IMAGEN GENERADA", visible=False)
        
        # === ERROR MESSAGES ===      
        exception_message = gr.Markdown(visible=True, value="")
        
        # logout_btn = gr.Button(
        #     "CERRAR SESIÓN",
        #     visible=True,
        #     size="md"
        # )

        # === EVENTS ===
        file_input.change(
            fn=on_file_upload,
            inputs=file_input,
            outputs=[file_input, output_json, output_img, reset_btn],
        )
        
        file_input.upload(
            fn=process,
            inputs=[file_input, session_id, enter_time],
            outputs=[output_json, output_img, exception_message],
        )
        
        reset_btn.click(
            fn=reset_ui,
            inputs=None,
            outputs=[
                file_input,
                output_json,
                output_img,
                exception_message,
                reset_btn
            ]
        )

    # === GLOBAL EVENTS (SCENE MANAGER) ===
    enter_btn.click(
        fn=set_session_id,
        inputs=session_input,
        outputs=[
            session_id,
            session_error,
            dashboard,
            session_gate,
            session_text,
            enter_time
        ],
    )
    
    test_btn.click(
        fn=set_session_id,
        inputs=gr.Textbox(value=000, visible=False),
        outputs=[
            session_id,
            session_error,
            dashboard,
            session_gate,
            session_text,
            enter_time
        ],
    )
    
    session_id.change(
        fn=render_session_text,
        inputs=session_id,
        outputs=[
            session_text,
            gate_title,
            session_input,
            session_error,
            enter_btn,
            test_btn
        ],
    )

import os
# ===LAUNCHING THE APP===

port = int(os.environ.get("PORT", 7860))
main.launch(theme=gr.themes.Citrus(), server_name="0.0.0.0", server_port=port, share=False)
