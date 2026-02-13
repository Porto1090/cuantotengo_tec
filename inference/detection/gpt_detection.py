import os
import json
import cv2
import base64
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def encode_crop(crop):
	_, buffer = cv2.imencode(".jpg", crop)
	b64 = base64.b64encode(buffer).decode("utf-8")
	return f"data:image/jpeg;base64,{b64}"

def llm_detect_brand(crop):
	image_data_url = encode_crop(crop)
	response = client.responses.create(
	model="gpt-4.1-mini",
		input=[
			{
				"role": "user",
				"content": [
					{
						"type": "input_text",
						"text": "Identify beverage CAN OR BOTTLE brand and flavor. Reply ONLY in format: BRAND_FLAVOR (e.g. pepsi_regular, dosequis_lager, etc). Do not add additional `_` characters."
					},
					{
						"type": "input_image",
						"image_url": image_data_url
					}
				]
			}
		]
	)

	return response.output_text.strip().lower()