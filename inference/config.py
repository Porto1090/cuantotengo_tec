"""
Configuration parameters for inference pipeline
""" 

SKU_CATALOG = {
    "Coca-Cola": ["Orange Cream", "Original", "Zero Sugar"],
    "Coke": ["Diet"],
    "Sprite": ["Original", "Zero"],
    "Canada Dry": ["Original"],
    "Dr Pepper": ["Original"],
    "Chobani": [
        "Strawberries & Cream",
        "Mixed Berry Vanilla",
        "Strawberry Banana",
        "Mixed Berry"
    ],
    "Gatorade Gatorlyte": [
        "Lemon-Lime",
        "Strawberry Kiwi",
        "Orange",
        "Cherry Lime",
        "Glacier Freeze"
    ],
    "Olipop Prebiotic Soda": ["Cream Soda", "Cherry Soda"],
    "Poppi Prebiotic Soda": [
        "Raspberry Rose",
        "Strawberry Lemon",
        "Orange Cream",
        "Lemon Lime",
        "Orange",
        "Grape"
    ],
    "Hood": ["Fat Free Milk"],
    "Seltzer Water": ["Lime"],
    "Modelo": ["Especial", "Negra"],
    "Pepsi": ["Black", "Original", "Light"],
    "Manzanita Sol": ["Original"],
    "Monster": [
        "Pipeline Punch",
        "Mango Loco",
        "Ultra Vice Guava",
        "Electric Blue"
    ],
    "So Good So You Sparkling Energy Drink": ["Strawberry Mango"],
    "Vitaminwater": ["Lemonade Squeezed"],
    "Fanta": ["Orange"],
    "Good & Gather": [
        "Vegetable Oil",
        "Distilled White Vinegar"
    ],
    "Vita Coco": [
        "Extra Coconut"
    ],
    "Recess": [
        "Lime Citrus",
        "Tropical Bliss",
        "Grapefruit Tangerine"
    ],
    "Polar Seltzer": [
        "Mandarin",
        "Raspberry Lime",
        "Lime",
        "Black Cherry",
        "Original"],
    "Poland Spring": ["Original"],
    "Quilmes": ["Classic"]
}

# TODO: Better version control than manually updating here
MODEL_VERSIONS = {
    "cap_model_version": 0.1,
    "front_model_version": 0.1,
    "lane_model_version": 0.1
}

# --- Column Detection ---
DISTANCE_THRESHOLD_TO_LINE = 200  # Max allowed perpendicular distance from column line NOTE: not used currently
DISTANCE_THRESHOLD_TO_FRONT_CAP = 900  # Max allowed direct distance from front cap
VANISHING_BOX_SIZE = 20  # Tight region for vanishing point estimation
OUTLIER_THRESHOLD = 150  # Threshold to determine misaligned columns NOTE: not used currently

# --- Cap Detection ---
CAP_MODEL_PATH = "models/bottle_can_cap_yolo_weights.pt"
CAP_DETECTION_CONFIDENCE = 0.3
SOFT_NMS_IOU_THRESHOLD = 0.5
SOFT_NMS_SIGMA = 0.5
SOFT_NMS_SCORE_THRESHOLD = 0.3

# --- Front Bottle Detection ---
FRONT_BOTTLE_MODEL_PATH = "models/bottlefront_weights.pt"
FRONT_DETECTION_CONFIDENCE = 0.5
FRONT_CAP_OVERLAP_THRESHOLD = 0.9  # 90% of cap inside front box

# --- MEXICO VERSION ---
MX_BRAND_MODEL_PATH = "models/brand_model_3class.pt"
MX_CLASS_NAMES = ["dosequis_lager", "manzanitasol_original", "modelo_especial", "modelo_negra", "newmix_jimadorpaloma", "pepsi_black", "pepsi_light", "pepsi_regular"]
MX_CLASS_NAMES_DICT = {
    "dosequis_lager": "Dos Equis Lager",
    "manzanitasol_original": "Manzanita Sol Original",
    "modelo_especial": "Modelo Especial",
    "modelo_negra": "Modelo Negra",
    "newmix_jimadorpaloma": "New Mix Jimador Paloma Lata",
    "pepsi_black": "Pepsi Black",
    "pepsi_light": "Pepsi Light",
    "pepsi_regular": "Pepsi Regular",
}

# --- LABORATORY VERSION ---
LAB_BRAND_MODEL_PATH = "models/brand_model_3class_lab.pt"
LAB_CLASS_NAMES = ["canadadry_gingerale", "cocacola_dietcoke", "seltzer_lime"]
LAB_CLASS_NAMES_DICT = {
    "canadadry_gingerale": "Canada Dry Ginger Ale",
    "cocacola_dietcoke": "Coca-Cola Diet Coke",
    "seltzer_lime": "Seltzer Water Lime"
}