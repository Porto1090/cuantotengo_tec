import os
import re
import cv2
import uuid
import string
import secrets
import numpy as np
import gradio as gr
import pandas as pd
import azure_blob_storage as azure_bs
from datetime import datetime, timezone
from inference.main_inference import InferencePipeline

# PLACEHOLDER:
dummy_df = pd.DataFrame({"MARCA": [], "TOTAL": []})
SESSION_REGEX = re.compile(r"^[0-9]{1,3}$")

from dotenv import load_dotenv
load_dotenv()

BRAND_DETECTION_VERSION = os.getenv("BRAND_DETECTION_VERSION")

BRAND_MAP = {
    "can - Dos Equis - Lager": "Dos Equis Lager",
    "can - Manzanita Sol - Original": "Manzanita Sol Original",
    "can - Modelo - Especial": "Modelo Especial",
    "can - Modelo - Negra": "Negra Modelo",
    "can - New Mix - Jimador Paloma Lata": "New Mix Jimador Paloma Lata",
    "can - Pepsi - Black": "Pepsi Black",
    "can - Pepsi - Light": "Pepsi Light",
    "can - Pepsi - Regular": "Pepsi Regular",
    "can - Canada Dry - Ginger Ale": "Canada Dry Ginger Ale",
    "can - Coca-Cola - Diet Coke": "Coca-Cola Diet Coke",
    "can - Seltzer Water - Lime": "Seltzer Water Lime",
}

BRAND_MAP_IMAGES = {
    "can - Dos Equis - Lager": "https://encrypted-tbn0.gstatic.com/shopping?q=tbn:ANd9GcR3wQ_D2ZrU3JoDUa0lm1zfnOFAm2PIjZnzj9D4Z0w-NebHoFjV8nePrTk1oq7GWK-VNnQeBlrlV7YAJq3R2mt396fQGSYh7w",
    "can - Manzanita Sol - Original": "https://ss302.liverpool.com.mx/xl/1046946100.jpg",
    "can - Modelo - Especial": "https://californiaranchmarket.com/cdn/shop/products/modelo.jpg?v=1659716608",
    "can - Modelo - Negra": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRF3eefN4QK5YEPSFFQMnYCxAoy2lF_WaVNAA&s",
    "can - New Mix - Jimador Paloma Lata": "https://cdn11.bigcommerce.com/s-0ddlsmhg83/products/10243/images/10525/new-mix-el-jimador-paloma-ready-to-drink-cocktail__57304.1752496204.1280.1280__00174.1755243376.386.513.jpg?c=1",
    "can - Pepsi - Black": "https://missionaryexpressbrazil.com/cdn/shop/files/pepsiblackzerosugar12oz2.49_992e9e25-3fa0-48e9-99c5-2b7c1698923a.webp?v=1747544034",
    "can - Pepsi - Light": "https://store.haciendaencantada.com/images/virtuemart/product/PEPSI%20LIGHT%20LATA.png",
    "can - Pepsi - Regular": "https://www.pepsicopartners.com/medias/300Wx300H-1-JVX42-1.jpeg?context=bWFzdGVyfHJvb3R8MjQ1ODN8aW1hZ2UvanBlZ3xhREZrTDJnNE5TOHhNREEwT1RVeU56UTBOelU0TWk4ek1EQlhlRE13TUVoZk1TMUtWbGcwTWkweExtcHdaV2N8MTI0YjExNmM1ZDA4YTQ5Yjk1OWE4NzI1YWE4YWU4MTQ5MTIwMGQyODQxODBlZTkyNDc4MGFkZGQyYTc4MWM2Nw",
    "can - Canada Dry - Ginger Ale": "https://i5.walmartimages.com/asr/db765423-e5ac-40e2-bb9e-b8f1c0cd1cfe.03628c2784f31baac3bc8bee88e311c8.jpeg?odnHeight=768&odnWidth=768&odnBg=FFFFFF",
    "can - Coca-Cola - Diet Coke": "https://boxncase.com/cdn/shop/files/bevcc18.original.jpg?v=1737915612&width=1920",
    "can - Seltzer Water - Lime": "https://i5.walmartimages.com/asr/02b5fd40-539c-4711-a7a6-7a6bde2ca476.787e2d88cadbf76a8b3c52a73ff07c3f.jpeg?odnHeight=768&odnWidth=768&odnBg=FFFFFF",
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
        pretty_name = BRAND_MAP.get(key, key)
        img = BRAND_MAP_IMAGES.get(key, "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQQmcXdOVEha1F9byhaNFhdHJkThCbyLM8g2g&s") 

        bg = bg_colors[idx % 2]

        rows += f"""
        <tr style="background-color:{bg}; border-bottom:1px solid #e5e7eb;">
            <td style="background-color:#FFFFFF; margin: auto; text-align:center; padding:10px; width:80px;">
                <img src="{img}" width="64" alt="{pretty_name}" />
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
    
    real_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    # === SAVE TO AZURE BLOB STORAGE ===
    azure_bs.save_image_to_blob(annotated_rgb, session_id, real_timestamp)
    log_dict = {
        "session_id": session_id,
        "timestamp": real_timestamp,
        "brand_totals": brand_totals,
        "processing_time": f"{processing_time:.2f}",
        "elapsed_time_in_session": f"{elapsed_time:.2f}",
    }
    azure_bs.save_log_to_blob(log_dict, session_id, real_timestamp)
    
    print(f"Timestamp Photo Session: {log_dict['elapsed_time_in_session']} seconds")
    print(f"Process Timestamp: {log_dict['processing_time']} seconds")

    SHOW_IMAGES = True
    if (SHOW_IMAGES):
        html_table = format_output_html(brand_totals)
        progress(1.0, desc="Completado")
        return html_table, annotated_rgb, gr.update(visible=False)
    else: 
        df = format_output_for_df(brand_totals)
        progress(1.0, desc="Completado")
        return df, annotated_rgb, gr.update(visible=False)

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
        gr.update(value=f"## ID Sesión: **{session_id}**", visible=True),
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
    
    dashboard_header = gr.Markdown(f"<div style='display:flex; align-items:center; justify-content:space-between;'>\
        <h1 style='font-size: 48px; color:#FFF'>CuantoTengo</h1> \
        <h3 style='font-size: 24px; color:#FFF'>VERSION <b>{BRAND_DETECTION_VERSION}</b></h3>\
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
    
    session_id.change(
        fn=render_session_text,
        inputs=session_id,
        outputs=[
            session_text,
            gate_title,
            session_input,
            session_error,
            enter_btn
        ],
    )

import os
# ===LAUNCHING THE APP===

port = int(os.environ.get("PORT", 7860))
main.launch(theme=gr.themes.Citrus(), server_name="0.0.0.0", server_port=port, share=False)
