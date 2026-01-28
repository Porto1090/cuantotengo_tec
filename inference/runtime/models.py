from ultralytics import YOLO

def load_models(device, CAP_MODEL_PATH, FRONT_BOTTLE_MODEL_PATH):
    cap_model = YOLO(CAP_MODEL_PATH)
    front_model = YOLO(FRONT_BOTTLE_MODEL_PATH)
    cap_model.to(device)
    front_model.to(device)
    
    return cap_model, front_model
