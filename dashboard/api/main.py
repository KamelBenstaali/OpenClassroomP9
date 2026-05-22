import json
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import cv2
import torch
import base64

# Import de Segment Anything (MedSAM est basé dessus)
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Attention: le module segment_anything n'est pas installé.")

app = FastAPI(title="MedSAM Segmentation API", version="1.0")

# CORS middleware pour permettre à Streamlit (UI) de communiquer avec l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chemin relatif vers le modèle MedSAM
import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDSAM_CHECKPOINT_PATH = os.path.join(BASE_DIR, "medsam_vit_b.pth")

# Téléchargement automatique du modèle si non présent (ex: sur Hugging Face Spaces)
if not os.path.exists(MEDSAM_CHECKPOINT_PATH):
    print("Modèle introuvable localement. Téléchargement depuis Hugging Face en cours... (Cela peut prendre un moment)")
    url = "https://huggingface.co/SansuiHan/medical_models/resolve/main/medsam_vit_b.pth"
    urllib.request.urlretrieve(url, MEDSAM_CHECKPOINT_PATH)
    print("Téléchargement terminé !")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dossier temporaire pour servir les images au canvas Streamlit
TEMP_DIR = os.path.join(BASE_DIR, "temp_images")
os.makedirs(TEMP_DIR, exist_ok=True)

from fastapi.responses import FileResponse
@app.get("/images/{filename}")
async def get_image(filename: str):
    filepath = os.path.join(TEMP_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="Image not found")

print(f"Chargement du modèle MedSAM depuis {MEDSAM_CHECKPOINT_PATH} sur {device}...")
try:
    # 1. Instanciation de l'architecture vide
    medsam_model = sam_model_registry["vit_b"](checkpoint=None)
    
    # 2. Chargement manuel des poids avec map_location pour éviter l'erreur CUDA sur Mac (CPU)
    state_dict = torch.load(MEDSAM_CHECKPOINT_PATH, map_location=device)
    medsam_model.load_state_dict(state_dict)
    
    # 3. Placement sur le bon device
    medsam_model = medsam_model.to(device)
    predictor = SamPredictor(medsam_model)
    print("✅ Modèle MedSAM chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    predictor = None

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode une image numpy en base64 pour la réponse JSON."""
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Could not encode image")
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de Segmentation MedSAM !"}

@app.post("/predict/")
async def predict_segmentation(
    file: UploadFile = File(...),
    prompt: str = Form(None)
):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Le modèle MedSAM n'est pas chargé.")
        
    try:
        # 1. Lire et décoder l'image envoyée par Streamlit
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Préparer le modèle
        predictor.set_image(img_rgb)
        
        # 3. Analyser le prompt envoyé par l'UI (si existant)
        box = None
        point_coords = None
        point_labels = None
        
        if prompt:
            prompt_dict = json.loads(prompt)
            if "box" in prompt_dict:
                box = np.array(prompt_dict["box"])[None, :]
            if "points" in prompt_dict and "labels" in prompt_dict:
                point_coords = np.array(prompt_dict["points"])
                point_labels = np.array(prompt_dict["labels"])
                
        # Fallback de sécurité : boîte par défaut si aucun prompt
        if box is None and point_coords is None:
            H, W, _ = img_rgb.shape
            margin = int(W * 0.05)
            box = np.array([margin, margin, W - margin, H - margin])[None, :]
            
        # 4. Prédiction du masque avec les bons arguments
        masks, scores, logits = predictor.predict(
            box=box,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        
        # Le masque est un tableau booléen, on le convertit en image (0 ou 255)
        mask = masks[0]
        mask_img = (mask * 255).astype(np.uint8)
        
        # 4. Renvoyer le masque encodé en Base64
        mask_base64 = encode_image_to_base64(mask_img)
        
        return JSONResponse({
            "status": "success",
            "score": float(scores[0]),
            "mask_base64": mask_base64
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
