"""
Configuration parameters for inference pipeline
""" 

SKU_CATALOG = {
    "Jumex": ["Manzana", "Durazno", "Mango"],
    "Boing": ["Mango", "Guayaba", "Fresa", "Manzana"],
    "Mirinda": ["Original"],
    "Manzanita Sol": ["Original"],
    "Pepsi": ["Original"],
    "Squirt": ["Original"],
    "Mirinda": ["Original"],
    "7 Up": ["Original"]
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