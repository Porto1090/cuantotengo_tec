import cv2
import torch
from PIL import Image
import torch.nn.functional as tnnf
from torchvision import transforms, models

# -------------------------------------------------------------
# Load model ONCE at import time (fast and efficient)
# -------------------------------------------------------------
CLASS_NAMES = ["Dos_Equis_Lager", "Manzanita_Sol", "Modelo_Especial", "Negra_Modelo", "New_Mix_Jimador_Paloma_Lata", "Pepsi_Black", "Pepsi_Light", "Pepsi_Regular"]      # ← your classes here
MODEL_PATH = "./models/brand_model_3class.pt"                           # ← your trained model file

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
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

        # Convert BGR → RGB → tensor
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        x = preprocess(pil_img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            logits = model(x)
            probs = tnnf.softmax(logits, dim=1)
            pred_idx = probs.argmax().item()

        LABEL_TO_BRAND_FLAVOR = {
            "Dos_Equis_Lager": ("Dos Equis", "Lager"),
            "Manzanita_Sol": ("Manzanita Sol", "Original"),
            "Modelo_Especial": ("Modelo", "Especial"),
            "Negra_Modelo": ("Modelo", "Negra"),
            "New_Mix_Jimador_Paloma_Lata": ("New Mix", "Jimador Paloma Lata"),
            "Pepsi_Black": ("Pepsi", "Black"),
            "Pepsi_Light": ("Pepsi", "Light"),
            "Pepsi_Regular": ("Pepsi", "Regular"),
        }

        raw_label = CLASS_NAMES[pred_idx]   # e.g., "Diet_Coke"

        brand_name, flavor_name = LABEL_TO_BRAND_FLAVOR[raw_label]

        # Format EXACTLY like GPT used to output
        formatted = f"can - {brand_name} - {flavor_name}"

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
        print(f"⚠️ Detected {len(front_bottles)} bottles but got {len(brands_list)} brand predictions.")
        min_len = min(len(front_bottles), len(brands_list))
        front_bottles = front_bottles[:min_len]
        brands_list = brands_list[:min_len]
    
    return {tuple(bbox.values()): brand for bbox, brand in zip(front_bottles, brands_list)}
