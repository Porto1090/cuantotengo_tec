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
        img = f"public/{key}.png"

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

def process(file, session_id, progress=gr.Progress()):
    print(f"\nSession ID: {session_id}")
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
        "processing_time": f"{processing_time:.4f}",
    }
    azure_bs.save_log_to_blob(log_dict, session_id, real_timestamp)

    SHOW_IMAGES = False
    if (SHOW_IMAGES):
        html_table = format_output_html(brand_totals)
        progress(1.0, desc="Completado")
        return html_table, annotated_rgb, gr.update(visible=False)
    else: 
        df = format_output_for_df(brand_totals)
        progress(1.0, desc="Completado")
        return df, annotated_rgb, gr.update(visible=False)

with gr.Blocks(title="CuantoTengo") as demo:
    session_id = gr.State()
    gr.Markdown("<h1 style='font-size: 48px;'>CuantoTengo</h1>")
    session_text = gr.Markdown("### ID Sesión: generando..._")
    
    def init_show_session():
        sid = generate_short_hex()
        return sid, f"## ID Sesión: **{sid}**"

    demo.load(fn=init_show_session, inputs=None, outputs=[session_id, session_text])

    # === INPUTS ===
    file_input = gr.File(label="PULSA AQUÍ PARA TOMAR LA FOTO", file_types=["image"])

    # === OUTPUTS ===
    reset_btn = gr.Button(
        "TOMAR OTRA FOTO",
        visible=False,
        size="lg"
    )
    output_json = gr.DataFrame(value=dummy_df, visible=False)
    # output_json = gr.HTML(visible=False)
    output_img = gr.Image(label="IMAGEN GENERADA", visible=False)
    
    # === ERROR MESSAGES ===      
    exception_message = gr.Markdown(visible=True, value="")
    
    # === INTERACTIONS ===
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

    file_input.change(
        fn=on_file_upload,
        inputs=file_input,
        outputs=[file_input, output_json, output_img, reset_btn],
    )
    
    file_input.upload(
        fn=process,
        inputs=[file_input, session_id],
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

import os
# LAUNCHING THE APP

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port, share=False)
