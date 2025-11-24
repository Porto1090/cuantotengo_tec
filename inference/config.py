# TODO: Better version control than manually updating here
MODEL_VERSIONS = {
    "cap_model_version": 0.1,
    "front_model_version": 0.1,
    "lane_model_version": 0.1
}

# --- Column Detection ---
DISTANCE_THRESHOLD_TO_LINE = 200  # Max allowed perpendicular distance from column line
DISTANCE_THRESHOLD_TO_FRONT_CAP = 900  # Max allowed direct distance from front cap
VANISHING_BOX_SIZE = 20  # Tight region for vanishing point estimation
OUTLIER_THRESHOLD = 150  # Threshold to determine misaligned columns

TOPS_MODEL_PATH = "bottle_can_cap_yolo_weights.pt"

# --- Product Dictionary Placeholder ---
# Standardized Product List: {brand_name: [flavors]}
standard_drinks = {
    # "Coca-Cola": ["Original", "Cherry", "Vanilla", "Zero Sugar"],
    # "Coca-Cola": ["Original", "Zero Sugar"],
    # "Coke": ["Diet", "Diet Caffeine free"],
    # "Pepsi": ["Original", "Wild Cherry", "Mango", "Zero Sugar"],
    # Gatorade Gatorlyte": ["Glacier Freeze", "Fruit Punch", "Lemon-Lime", "Cool Blue", "Orange", "Cherry Lime", "Strawberry Kiwi"],
    # "Red Bull": ["Energy Drink", "Sugarfree", "Juneberry", "Watermelon", "The Red Edition Watermelon Supgar Free", "Strawberry Apricot", "Curuba Elderflower","The Spring Edition Grapefruit & Blossom Suger Free"],
    # "Monster": ["Original", "Ultra Sunrise", "Mango Loco", "Zero Ultra"],
    # "Dr Pepper": ["Original", "Cherry", "Vanilla Float", "Cream Soda"],
    # "Sprite": ["Original", "Tropical Mix", "Cherry", "Lemonade","Zero"],
    # "Mountain Dew": ["Original", "Baja Blast", "Code Red", "Major Melon"],
    # "Arizona": ["Half & Half", "Green Tea", "Arnold Palmer", "Mucho Mango", "RX Energy"],
    # "Fanta": ["Orange", "Grape", "Strawberry", "Pineapple"],
    # "Powerade": ["Mountain Berry Blast", "Fruit Punch", "Lemon-Lime", "Grape"],
    # "Vitaminwater": ["XXX Acai-Blueberry-Pomegranate", "Power-C Dragonfruit", "Revive Fruit Punch", "Focus Kiwi-Strawberry"],
    # "Snapple": ["Peach Tea", "Lemon Tea", "Mango Madness", "Kiwi Strawberry"],
    # "Nestea": ["Lemon", "Raspberry", "Peach", "Sweet Tea"],
    # "Tropicana": ["Orange Juice", "Pineapple Mango", "Peach Passion", "Berry Blend"],
    # "Minute Maid": ["Lemonade", "Fruit Punch", "Mango Passion", "Berry Punch"],
    # "Ocean Spray": ["Cranberry", "Cran-Grape", "Cran-Pomegranate", "Cran-Apple"],
    # "Lipton": ["Iced Tea Lemon", "Peach", "Green Tea Citrus", "Raspberry"],
    # "Starbucks": ["Coffee Frappuccino", "Mocha Frappuccino", "Vanilla Frappuccino", "Caramel Frappuccino"],
    # "Bang": ["Rainbow Unicorn", "Sour Heads", "Black Cherry Vanilla", "Purple Haze"],
    # "Chobani 20G Protein": ["Strawberries & Cream (20G Protein)", "Mixed Berry & Vanilla (20G Protein)", "Greek Yogurt Strawberry Banana", "Greek Yogurt Mixed Berry"],
    # "Olipop Prebiotic Soda":["Cream Soda", "Cherry Cola"],
    # "Poppi Prebiotic Soda":["Raspberry Rose","Strawberry Lemon", "Orange Cream", "Lemon Lime", "Orange", "Grape"],
    # "Pure Life": ["Water"],
    # "Schweppes": ["Ginger Ale"],
    # "V8": ["Original"],
    # "Olipop": ["Lemon Lime"],
    # "Spindrift": ["Rapberry Lime"],

    # Argentina brands
    # "Levite": ["Pomelo", "Manzana", "Pomelo Rosado", "Pera", "Naranja"], # Realmente es limón?
    # "Benedictino": ["Sin Gas", "Con Gas"],
    # "Powerade": ["Uva", "Mountain Blast", "Frutas Tropicales", "Manzana"],
    # "Smart Water": ["Sin Gas", "Con Gas"],
    # "Coca Cola": ["Zero", "Light"],
    # "Sprite": ["Zero", "Original"],
    # "Villavicencio": ["Agua Natural"],
    # "Villa del Sur": ["Agua Natural"],
    # "Aqua Rius": ["Uva", "Manzana", "Pomelo"],
    # "H2OH": ["Limoneto", "Manzanilla"],
    # "Paso de los Toros": ["Tónica", "Pomelo"],
    # "Mirinda": ["Sabor Naranja", "Manzana"],
    # "Pepsi": ["Pepsi Twist", "Black","Original"],
    # "7UP":["Lima Limón", "sugar-free lemon-lime flavor"],
    # "Gatorade":["Frutas Tropicales", "Limón", "Uva", "Green Mango", "Cool Blue"],
    # "Nestlé Pureza Vital": ["water"],
    # "Bonafont": ["water"],
    # "Quilmes": ["Clásica"],
    # "Andes Origen": ["Negra", "IPA Andina", "Rubia", "Roja"],
    # "Stella Artois": ["Premium Lager"],
    # "Corona": ["Cerveza"]

    # Mexico brands
    # "Coca Cola": ["Zero", "Light", "Original"],
    # "Sprite": ["Zero", "Original"],
    # "Fanta": ["Fresa", "Naranja", "Piña", "Toronja"],
    # "Ciel": ["Agua Natural", "Agua Mineral", "Levité Durazno", "Levité Manzana"],
    # "Pepsi": ["Original", "Black", "Light"],
    # "Manzanita Sol": ["Original"],
    # "Mirinda": ["Naranja", "Fresa"],
    # "Sidral Mundet": ["Original", "Light"],
    # "Jumex": ["Mango", "Durazno", "Manzana", "Piña", "Multifrutas"],
    # "Boing": ["Guayaba", "Mango", "Fresa", "Uva", "Manzana"],
    # "Bonafont": ["Agua Natural", "Agua Ligera Limón", "Agua Ligera Toronja"],
    # "Epura": ["Agua Natural", "Agua Mineral"],
    # "Topo Chico": ["Agua Mineral", "Twist Limón", "Twist Toronja"],
    # "Electrolit": ["Fresa-Kiwi", "Uva", "Coco", "Manzana", "Naranja"],
    # "Powerade": ["Azul", "Rojo", "Morado", "Naranja"],
    # "Gatorade": ["Naranja", "Limón", "Cool Blue", "Mora Azul"],
    # "Del Valle": ["Naranja", "Durazno", "Manzana", "Mango", "Piña"],
    # "Vallefrut": ["Naranja", "Manzana", "Durazno"],
    # "Santa Clara": ["Leche Entera", "Light", "Sabor Chocolate", "Sabor Fresa"]

    # ctl field experiments
    # "Coca-Cola": ["Original", "Zero Sugar","Diet Coke", "Diet Coke Caffeine Free"],
    # "Canada Dry": ["Ginger Ale"],
    # "Seltzer Water": ["Lime", "Original", "Lemon", "Raspberry Lime"],
    # "Pepsi": ["Original", "Diet"],
    # "Sprite": ["Lemon-Lime"]
    "Coca-Cola": ["Diet Coke"],          # For class: Diet_Coke
    "Canada Dry": ["Ginger Ale"],        # For class: Canada_Dry
    "Seltzer Water": ["Lime"]            # For class: Seltzer_Lime
}

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

# --- Brand Detection ---
GPT_SYSTEM_PROMPT = (
    "You are an expert in identifying beverage brands. Analyze the provided bottle image "
    "and return its EXACT brand and flavor details in the STRICT format: 'BRAND - FLAVOR'."
)
GPT_USER_PROMPT_TEMPLATE = (
    "Identify the beverage brand and flavor in this image and answer as 'BRAND - FLAVOR'. "
    "Here is a dictionary with the brands (dictionary keys) and their corresponding flavors (dictionary values) you should consider:\n{standard_drinks}\n"
    "Please, restrict your predictions exclusively to brands and flavors in this dictionary. All products in the image are in the dictionary, so make your best guess from the list, even if unsure."
)
GPT_MODEL = "gpt-4o"
GPT_TEMPERATURE = 0.0
GPT_MAX_TOKENS = 100

# --- Misc ---
MODEL_VERSION = "0"