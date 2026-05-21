import pytest
from PIL import Image
import numpy as np
import sys
import os

# Ajout du chemin du dossier parent pour importer app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import apply_transformations

def test_apply_transformations():
    """Test de la fonction de transformation d'image utilisée dans l'EDA"""
    # Création d'une image PIL factice de 100x100 pixels RGB
    dummy_array = np.zeros((100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_array)
    
    # Appel de la fonction
    blurred, equalized, rotated, flipped = apply_transformations(image)
    
    # Vérification du floutage
    assert isinstance(blurred, np.ndarray)
    assert blurred.shape == (100, 100, 3)
    
    # Vérification de l'égalisation
    assert isinstance(equalized, np.ndarray)
    assert equalized.shape == (100, 100, 3)
    
    # Vérification de la rotation
    assert isinstance(rotated, np.ndarray)
    assert rotated.shape == (100, 100, 3)
    
    # Vérification du miroir
    assert isinstance(flipped, np.ndarray)
    assert flipped.shape == (100, 100, 3)
