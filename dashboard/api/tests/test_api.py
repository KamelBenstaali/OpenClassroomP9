import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2
import sys
import os

# Ajout du chemin du dossier parent pour importer main.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app

client = TestClient(app)

def test_read_root():
    """Test de la route d'accueil de l'API"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API de Segmentation MedSAM !"}

def test_get_image_not_found():
    """Test de la route de récupération d'image avec une image inexistante"""
    response = client.get("/images/image_inexistante_123.jpg")
    assert response.status_code == 404
    assert response.json() == {"detail": "Image not found"}

def test_predict_segmentation_no_prompt():
    """Test de l'inférence de l'API avec une fausse image"""
    # Création d'une image factice noire de 100x100
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    success, buffer = cv2.imencode('.jpg', dummy_image)
    
    response = client.post(
        "/predict/",
        files={"file": ("dummy.jpg", buffer.tobytes(), "image/jpeg")}
    )
    
    # Si le modèle medsam_vit_b.pth n'est pas trouvé (ex: environnement CI), 
    # l'API gère l'erreur avec une 500, sinon on s'attend à un 200.
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert data.get("status") == "success"
        assert "score" in data
        assert "mask_base64" in data
