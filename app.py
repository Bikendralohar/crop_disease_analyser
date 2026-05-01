import os
import io
import json
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import transforms
from src.model import build_model

app = Flask(__name__)

# -- Configuration --
MODEL_PATH = os.path.join("outputs", "checkpoints", "best_model.pth")
DB_PATH = os.path.join("src", "treatment_db.json")
CLASS_DIR = "data/PlantDisease"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Load Class Names --
if os.path.exists(CLASS_DIR):
    CLASS_NAMES = sorted(os.listdir(CLASS_DIR))
else:
    CLASS_NAMES = [] 

# -- Load Treatment Database --
treatment_db = {}
if os.path.exists(DB_PATH):
    with open(DB_PATH, 'r') as f:
        treatment_db = json.load(f)

# -- HELPER: Key Normalizer --
def get_db_info(folder_name):
    """
    Tries to find a match in the JSON even if underscores differ.
    Example: Folder 'Apple__scab' -> Matches DB Key 'Apple___Apple_scab'
    """
    # 1. Try exact match
    if folder_name in treatment_db:
        return treatment_db[folder_name]
    
    # 2. Try fuzzy matching (ignore underscores and case)
    clean_folder = folder_name.replace("_", "").lower()
    for key in treatment_db:
        clean_key = key.replace("_", "").lower()
        if clean_folder == clean_key:
            return treatment_db[key]
            
    # 3. Fallback for common partial matches (e.g. "Apple__scab" in "Apple___Apple_scab")
    for key in treatment_db:
        if folder_name.split("__")[-1] in key:
            return treatment_db[key]

    return None

def format_label(name):
    return name.replace("___", " — ").replace("__", " — ").replace("_", " ").title()

# -- Model & Transforms --
model = None
if os.path.exists(MODEL_PATH) and CLASS_NAMES:
    model = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
        
        folder_name = CLASS_NAMES[idx.item()]
        
        # USE THE NEW KEY NORMALIZER
        info = get_db_info(folder_name)
        
        if not info:
            desc = "Biological analysis complete. Data mapping mismatch (Check JSON keys)."
            treat = "Check if JSON key matches folder name exactly."
        else:
            desc = info["description"]
            treat = info["treatment"]
        
        return jsonify({
            "disease": format_label(folder_name),
            "confidence": f"{conf.item()*100:.1f}%",
            "description": desc,
            "treatment": treat
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_guides")
def get_guides():
    guides = {}
    for name in CLASS_NAMES:
        info = get_db_info(name)
        if info:
            guides[name] = info
        else:
            guides[name] = {
                "description": f"No JSON data found for key: {name}",
                "treatment": "Check src/treatment_db.json"
            }
    return jsonify(guides)

if __name__ == "__main__":
    app.run(debug=True, port=5000)