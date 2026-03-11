import os
import json
import cv2
import base64
from openai import OpenAI
from inference.config import (
	SKU_CATALOG
)

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def encode_crop(crop):
	_, buffer = cv2.imencode(".jpg", crop)
	b64 = base64.b64encode(buffer).decode("utf-8")
	return f"data:image/jpeg;base64,{b64}"

def build_sku_list():
    skus = []
    for brand, flavors in SKU_CATALOG.items():
        for flavor in flavors:
            skus.append(f"{brand} {flavor}")
    return skus
  
def llm_detect_brand(crop, nearby_crops):
	main_image_data_url = encode_crop(crop)
	nearby_images = []
	for n_crop in nearby_crops:
		nearby_images.append({
				"type": "input_image",
				"image_url": encode_crop(n_crop)
		})

	sku_list = build_sku_list()
	sku_list_text = "\n".join([f"- {sku}" for sku in sku_list])

	response = client.responses.create(
	model="gpt-5.2",
		input=[
			{
				"role": "user",
				"content": [
					{
						"type": "input_text",
						"text": f"""
      				You are a product recognition system.
							Identify the beverage SKU shown in the image.

							You MUST choose EXACTLY one option from the SKU list.

							SKU LIST:
							{sku_list_text}

							Rules:
							- Return ONLY the exact SKU text.
							- Do not add explanations.
							- Do not invent SKUs.
							- If uncertain, choose the closest visual match.

							Main product image:
							"""
					},
					{
						"type": "input_image",
						"image_url": main_image_data_url
					},
					{
						"type": "input_text",
						"text": "Nearby products for context:"
					},
					*nearby_images
				]
			}
		]
	)

	result = response.output_text.strip()

	if result not in sku_list:
		return "unknown"

	return result.lower()