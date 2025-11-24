import requests
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from io import BytesIO
from requests.auth import HTTPBasicAuth
from PIL import Image
import numpy as np
from datetime import datetime, timezone

from keys import twilio_keys
from services.database import update_value, fetch_query, create_record
from services.whatsapp import send_message
from services.blob_storage import upload_image_to_blob, generate_blob_url
from inference.main_inference import run_inference
from inference.config import TOPS_MODEL_PATH, MODEL_VERSIONS

import logging, time, uuid
from contextlib import contextmanager
from flask import g

logging.basicConfig(level=logging.INFO)  # or DEBUG
log = logging.getLogger("cuantotengo")

@contextmanager
def stage(name):
    t0 = time.time()
    rid = getattr(g, "rid", "no-rid")
    log.info("[%s] ▶ %s: start", rid, name)
    try:
        yield
        dt = (time.time() - t0) * 1000
        log.info("[%s] ✅ %s: done in %.1f ms", rid, name, dt)
    except Exception:
        dt = (time.time() - t0) * 1000
        log.exception("[%s] ❌ %s: failed after %.1f ms", rid, name, dt)
        raise


app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    """Runs "process_image" when an image is sent to the corresponding WhatsApp number."""

    g.rid = uuid.uuid4().hex[:8]
    log.info("[%s] Webhook hit. form=%s", g.rid, dict(request.form))

    print(f"Received a message.")
    #print(">>> Webhook entered.")

    # Get data
    media_url = request.form.get("MediaUrl0")
    media_type = request.form.get("MediaContentType0")
    sender_phone = request.form.get("From")
    #print(f">>> Incoming form: media_url={media_url}, media_type={media_type}, sender_phone={sender_phone}")

    try:
        # Check if user is registered
        phone = sender_phone.split(":")[-1]
        user_row = fetch_query("SELECT * FROM users WHERE phone_number = '{}'".format(phone))
        print("SELECT * FROM users WHERE phone_number = '{}'".format(phone))
        print(user_row)

        if user_row == []:
            send_message(sender_phone, "Your phone number is not registered. Please ask someone to add your data to the database.")
        else:
            user_id = user_row[0][0]
            if media_url and media_type.startswith("image/"):
                run_pipeline(media_url, sender_phone, user_id)
            else:
                send_message(sender_phone, "Please send a picture.")
            
            response = MessagingResponse()
            return str(response)
    # TODO: Differentiate between exceptions that are because of the picture vs. exceptions that are due to other reasons
    except Exception as e:
         print(e)
         send_message(sender_phone, "Please try taking another picture.")


def run_pipeline(image_url, sender_phone, user_id):

    response = requests.get(image_url, stream=True, auth=HTTPBasicAuth(twilio_keys["account_sid"], twilio_keys["auth_token"]))
    print("At here.")

    if response.status_code == 200:

        # Preprocess image
        with stage("preprocess"):
            processed_image = preprocess_image(response)
            log.info("[%s] image shape=%s", g.rid, getattr(processed_image, "shape", None))

        # Run inference
        with stage("inference"):
            class_counts, annotated_image, cap_data, front_bottles, bottle_brand_mapping, lane_totals = run_inference(processed_image, sender_phone)
            log.info("[%s] classes=%s", g.rid, list(class_counts.keys()))

        # Upload image metadata
        #image_data = {'user_id': user_id} | MODEL_VERSIONS
        image_data = {**{'user_id': user_id}, **MODEL_VERSIONS}
        image_id = create_record(table_name="images", data=image_data, id_column='image_id')

        # Upload annotated image to blob 
        # TODO: Make the annotated image show the lanes found as well 
        # (we skip this step for now for faster inference)
        # upload_image_to_blob(annotated_image, f"annotated_images/{image_id}.png")
        # annotated_image_url = generate_blob_url(f"annotated_images/{image_id}.png")

        # # Send results
        #send_results_to_user(class_counts, sender_phone, annotated_image_url)

        # just send the results without image
        send_results_to_user(class_counts, sender_phone, None)

        # Upload original image to blob
        upload_image_to_blob(processed_image, f"raw_images/{image_id}.png")

        # Upload caps data
        for cap in cap_data:
            cap_data_db = {
                "image_id": image_id,
                "object_type": int(cap["class"]),
                "x_center": float(cap["x1"]), # TODO: Is this actually the center?
                "y_center": float(cap["y1"]),
                "width": float(cap["x2"]),
                "height": float(cap["y2"]),
                "confidence": float(cap["confidence"])
            }
            create_record(table_name="ProductTops", data=cap_data_db, id_column="product_top_id")

        # Upload fronts  abd lanes data
        for front in front_bottles:
            bottle_brand_mapping_key = (front["x1"], front["y1"], front["x2"], front["y2"], front["confidence"])
            brand_string = bottle_brand_mapping[bottle_brand_mapping_key]
            if brand_string == "Unknown Drink":
                brand, flavor = "", ""
            else:
                brand, flavor = brand_string.split(" - ", 1) if " - " in brand_string else (brand_string, "")

            # Fronts
            front_data_db = {
                "image_id": image_id,
                "object_type": 1, # TODO
                "x_center": float(front["x1"]), # TODO: Is this actually the center?
                "y_center": float(front["y1"]),
                "width": float(front["x2"]),
                "height": float(front["y2"]),
                "confidence": float(front["confidence"]),
                "brand": brand,
                "product_name": flavor,
            }
            product_front_id = create_record(table_name="ProductFronts", data=front_data_db, id_column="product_front_id")

            # Lanes
            lane_data_db = {
                "image_id": image_id,
                "product_front_id": product_front_id,
                "model_count": lane_totals[bottle_brand_mapping_key[0:4]]
            }
            create_record(table_name="Lanes", data=lane_data_db, id_column="lane_id")


        return jsonify({'message': "Inference has run!"}), 200
    else:
        return f"Failed to download image. Status code: {response.status_code}"


def preprocess_image(response):
    print("Preprocessing image.")
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(image)[:, :, ::-1]


# TODO: The inference should return a dictionary with attributes rather than a string
# Version to send image and results
# def send_results_to_user(class_counts, sender_phone, annotated_image_url):
#     """Formats and sends WhatsApp message with results."""
#     print("Replying to user.")

#     def format_label(label, count):
#         try:
#             item, brand, variant = label.split(' - ', 2)
#             item_label = f"{item}s" if count > 1 else item
#             return f"{count} {item_label} of {brand} ({variant})"
#         except ValueError:
#             return f"{count} {label}"

#     items = [f"- {format_label(label, count)}" for label, count in class_counts.items()]

#     # If no items, send a different message
#     if not items:
#         body = "Found nothing."
#     else:
#         body = "\n".join(items)
    
#     send_message(sender_phone, body, annotated_image_url)

# Version to send just text results
def send_results_to_user(class_counts, sender_phone, annotated_image_url):
    """Formats and sends WhatsApp message with results."""
    print("Replying to user.")

    def format_label(label, count):
        try:
            item, brand, variant = label.split(' - ', 2)
            item_label = f"{item}s" if count > 1 else item
            return f"{count} {item_label} of {brand} ({variant})"
        except ValueError:
            return f"{count} {label}"

    items = [f"- {format_label(label, count)}" for label, count in class_counts.items()]

    if not items:
        body = "Found nothing."
    else:
        body = "\n".join(items)

    #Only send text, skip image attachment
    send_message(sender_phone, body)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
