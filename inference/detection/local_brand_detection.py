import os
import cv2
import torch
from PIL import Image
import torch.nn.functional as tnnf
from torchvision import transforms, models
from inference.config import(
    LAB_BRAND_MODEL_PATH, LAB_CLASS_NAMES,
    MX_BRAND_MODEL_PATH, MX_CLASS_NAMES
)
from inference.detection.gpt_detection import llm_detect_brand

CLASS_NAMES = []
BRAND_MODEL_PATH = ""

from dotenv import load_dotenv
load_dotenv()

BRAND_DETECTION_VERSION = os.getenv("BRAND_DETECTION_VERSION")
if BRAND_DETECTION_VERSION == "MEX":
    CLASS_NAMES = MX_CLASS_NAMES
    BRAND_MODEL_PATH = MX_BRAND_MODEL_PATH
elif BRAND_DETECTION_VERSION == "LAB":
    CLASS_NAMES = LAB_CLASS_NAMES
    BRAND_MODEL_PATH = LAB_BRAND_MODEL_PATH

model = None

if BRAND_DETECTION_VERSION != "GPT":        
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(BRAND_MODEL_PATH, map_location="cpu"))
    model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------------------
# Replacement for LLM-based detection
# This matches your old signature EXACTLY:
#   get_brands_from_image(front_bottles, image) → list[str]
#
# No other code in your pipeline needs to change.
# -------------------------------------------------------------
def get_brands_from_image(front_bottles, image_bgr):
    """
    front_bottles: list of dicts from YOLO, each containing x1, y1, x2, y2
    image_bgr: the full BGR image (numpy array)
    returns: list of brand strings (same format as old LLM function)
    """

    brand_results = []

    for bottle in front_bottles:
        x1, y1 = int(bottle["x1"]), int(bottle["y1"])
        x2, y2 = int(bottle["x2"]), int(bottle["y2"])

        # Crop the bottle front (same as your old code structure)
        crop = image_bgr[y1:y2, x1:x2]
        
        if BRAND_DETECTION_VERSION == "GPT":
            formatted = llm_detect_brand(crop)
        else:
            # Convert BGR → RGB → tensor
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            x = preprocess(pil_img).unsqueeze(0)

            # Predict
            with torch.no_grad():
                logits = model(x)
                probs = tnnf.softmax(logits, dim=1)
                pred_idx = probs.argmax().item()
            
            formatted = CLASS_NAMES[pred_idx]

        brand_results.append(formatted)

    return brand_results


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
        print(f"Detected {len(front_bottles)} bottles but got {len(brands_list)} brand predictions.")
        min_len = min(len(front_bottles), len(brands_list))
        front_bottles = front_bottles[:min_len]
        brands_list = brands_list[:min_len]
    
    return {tuple(bbox.values()): brand for bbox, brand in zip(front_bottles, brands_list)}
