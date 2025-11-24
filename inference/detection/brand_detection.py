import cv2
import base64
import re
from PIL import Image
from openai import OpenAI
from inference.config import (
    GPT_SYSTEM_PROMPT,
    GPT_USER_PROMPT_TEMPLATE,
    GPT_MODEL,
    GPT_TEMPERATURE,
    GPT_MAX_TOKENS,
    standard_drinks,
)
from keys import openai_key
from inference.image_utils import encode_image


def get_brands_from_image(front_bottles, image):
    """
    Sends each front bottle image to an API (e.g., GPT-4o) individually 
    and extracts brand + flavor for more precise identification.

    Args:
        front_bottles (list): List of bounding boxes [(x1, y1, x2, y2)] for front bottles.
        image (numpy.ndarray): Original shelf image in BGR format.

    Returns:
        list: Identified brands in left-to-right order as strings (e.g., ["Coca-Cola - Original", ...])
    """
    if image is None:
        raise ValueError("Error: The provided image array is None.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    brands_list = []
    client = OpenAI(api_key=openai_key)

    for i, front in enumerate(front_bottles):
        cropped = image_rgb[int(front["y1"]):int(front["y2"]), int(front["x1"]):int(front["x2"])]
        cropped_pil = Image.fromarray(cropped)
        cropped_path = f"/tmp/bottle_{i}.jpg"
        cropped_pil.save(cropped_path)

        base64_image = encode_image(cropped_path)
        user_prompt = GPT_USER_PROMPT_TEMPLATE.format(standard_drinks=standard_drinks)

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=GPT_MAX_TOKENS,
            temperature=GPT_TEMPERATURE,
        )

        brand_detail = response.choices[0].message.content.strip()
        brands_list.append(brand_detail)

    return brands_list


def match_brands_to_bottles(front_bottles, brands_list):
    """
    Assigns brands from GPT-4o's `brands_list` to detected front bottles.

    Args:
        front_bottles (list): List of bounding boxes [(x1, y1, x2, y2)] for front-facing bottles.
        brands_list (list): List of strings containing predicted brand-flavor combinations.

    Returns:
        dict: Mapping from bottle bounding boxes to brand strings.
    """
    if len(front_bottles) != len(brands_list):
        print(f"⚠️ Detected {len(front_bottles)} bottles but got {len(brands_list)} brand predictions.")
        min_len = min(len(front_bottles), len(brands_list))
        front_bottles = front_bottles[:min_len]
        brands_list = brands_list[:min_len]
    
    return {tuple(bbox.values()): brand for bbox, brand in zip(front_bottles, brands_list)}
    #return {bbox: brand for bbox, brand in zip(front_bottles, brands_list)}
