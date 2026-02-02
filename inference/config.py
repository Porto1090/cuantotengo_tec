"""
Configuration parameters for inference pipeline
""" 

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
MX_CLASS_NAMES = ["Dos_Equis_Lager", "Manzanita_Sol", "Modelo_Especial", "Negra_Modelo", "New_Mix_Jimador_Paloma_Lata", "Pepsi_Black", "Pepsi_Light", "Pepsi_Regular"]
MX_CLASS_NAMES_DICT = {
    "Dos_Equis_Lager": ("Dos Equis", "Lager"),
    "Manzanita_Sol": ("Manzanita Sol", "Original"),
    "Modelo_Especial": ("Modelo", "Especial"),
    "Negra_Modelo": ("Modelo", "Negra"),
    "New_Mix_Jimador_Paloma_Lata": ("New Mix", "Jimador Paloma Lata"),
    "Pepsi_Black": ("Pepsi", "Black"),
    "Pepsi_Light": ("Pepsi", "Light"),
    "Pepsi_Regular": ("Pepsi", "Regular"),
}

# --- LABORATORY VERSION ---
LAB_BRAND_MODEL_PATH = "models/brand_model_3class_lab.pt"
LAB_CLASS_NAMES = ["Canada_Dry", "Diet_Coke", "Seltzer_Lime"]
LAB_CLASS_NAMES_DICT = {
    "Diet_Coke": ("Coca-Cola", "Diet Coke"),
    "Canada_Dry": ("Canada Dry", "Ginger Ale"),
    "Seltzer_Lime": ("Seltzer Water", "Lime")
}