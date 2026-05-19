import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Plus besoin de patch, on injecte l'image directement dans l'état du Canvas en Base64.
import base64
import io

st.set_page_config(
    page_title="ISIC 2018 - Segmentation de Lésions",
    page_icon="🩺",
    layout="wide"
)

# --- Fonctions utiles ---
def apply_transformations(image):
    """Applique le floutage et l'égalisation d'histogramme pour l'EDA"""
    # Convertir PIL Image en numpy array (RGB)
    img_array = np.array(image)
    
    # Floutage (Blur)
    blurred = cv2.GaussianBlur(img_array, (15, 15), 0)
    
    # Egalisation d'histogramme (sur l'espace LAB pour préserver les couleurs)
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    l_eq = cv2.equalizeHist(l)
    img_eq = cv2.merge((l_eq, a, b))
    equalized = cv2.cvtColor(img_eq, cv2.COLOR_LAB2RGB)
    
    return blurred, equalized

# --- Navigation Horizontale ---
st.markdown("<h2 style='text-align: center;'>🩺 Projet P9 - Dashboard de Segmentation</h2>", unsafe_allow_html=True)
page = st.radio(
    "Navigation :", 
    ["Analyse Exploratoire (EDA)", "Moteur de Prédiction"], 
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("---")

# ==========================================
# PAGE 1 : ANALYSE EXPLORATOIRE (EDA)
# ==========================================
if page == "Analyse Exploratoire (EDA)":
    st.title("📊 Analyse Exploratoire des Données (ISIC 2018)")
    st.markdown("""
    Cette section présente une exploration du jeu de données **ISIC 2018**, utilisé pour l'entraînement 
    de notre modèle de segmentation de lésions cutanées.
    """)
    
    st.header("1. Répartition des données")
    
    # Métriques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Images d'entraînement", value="2 594")
    with col2:
        st.metric(label="Images de validation", value="100")
    with col3:
        st.metric(label="Images de test", value="1 000")
        
    st.markdown("### Comptage et distribution")
    # Simulation de données pour le graphique interactif (WCAG compliance)
    labels = ['Entraînement', 'Validation', 'Test']
    sizes = [2594, 100, 1000]
    # Palette Okabe-Ito, excellente pour l'accessibilité (Daltonisme)
    colors = ['#0072B2', '#E69F00', '#009E73'] 
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, sizes, color=colors)
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Répartition du dataset ISIC 2018 (Critère WCAG respecté)")
    
    # Ajouter la valeur sur chaque barre pour plus de lisibilité
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 50, int(yval), ha='center', va='bottom', fontweight='bold')
        
    st.pyplot(fig, clear_figure=True)
    
    st.header("2. Transformations d'images (Data Augmentation / Pre-processing)")
    st.markdown("Les spécifications demandent de présenter des exemples de transformations. Testez avec une image :")
    
    uploaded_eda = st.file_uploader("Chargez une image pour visualiser les transformations", type=["jpg", "png", "jpeg"], key="eda")
    if uploaded_eda is not None:
        image = Image.open(uploaded_eda)
        blurred, equalized = apply_transformations(image)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Image Originale", use_container_width=True)
        with col2:
            st.image(blurred, caption="Floutage (Gaussian Blur)", use_container_width=True)
        with col3:
            st.image(equalized, caption="Égalisation d'histogramme", use_container_width=True)

# ==========================================
# PAGE 2 : MOTEUR DE PREDICTION
# ==========================================
elif page == "Moteur de Prédiction":
    st.title("🧠 Moteur de Prédiction (Segmentation)")
    st.markdown("""
    Utilisez le modèle d'Intelligence Artificielle MedSAM pour segmenter automatiquement la lésion sur une image dermatologique.
    """)
    
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0
    if "box_xmin" not in st.session_state: st.session_state.box_xmin = 0
    if "box_ymin" not in st.session_state: st.session_state.box_ymin = 0
    if "box_xmax" not in st.session_state: st.session_state.box_xmax = 0
    if "box_ymax" not in st.session_state: st.session_state.box_ymax = 0
    if "manual_boxes" not in st.session_state: st.session_state.manual_boxes = []

    st.header("1. Sélection de la donnée en entrée")
    input_method = st.radio("Comment souhaitez-vous fournir l'image ?", 
                            ["Télécharger une image (Upload)", "Choisir un exemple du dataset"])
    
    image_to_predict = None
    
    if input_method == "Télécharger une image (Upload)":
        uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"], key="pred")
        if uploaded_file is not None:
            image_to_predict = Image.open(uploaded_file)
            st.success("Image chargée avec succès.")
    else:
        # Sélection d'une image d'exemple parmi celles extraites dans 'data_example'
        import os
        example_dir = os.path.join(os.path.dirname(__file__), "data_example")
        if os.path.exists(example_dir):
            example_images = [f for f in os.listdir(example_dir) if f.endswith('.jpg')]
            example_images.sort()
        else:
            example_images = []
            
        if not example_images:
            st.error("⚠️ Aucune image trouvée dans le dossier 'data_example'.")
        else:
            selected_example = st.selectbox("Choisissez une image de test :", example_images)
            image_path = os.path.join(example_dir, selected_example)
            
            # Chargement de la vraie image
            image_to_predict = Image.open(image_path)
            st.success(f"Image {selected_example} prête.")
            
    st.header("2. Configuration du Modèle (MedSAM)")
    st.info("💡 Le modèle utilisé est **MedSAM**, un 'Foundation Model' qui requiert un guide (prompt) pour segmenter la lésion.")
    
    # On ne garde que le traçage par rectangle (Boîte englobante)
    prompt_type = "Boîte englobante (Rectangle)"
    prompt_data = {}
    
    if image_to_predict is None:
        st.warning("Veuillez sélectionner ou uploader une image à l'étape 1 avant de configurer le prompt.")
    else:
        # Redimensionnement pour l'affichage du Canvas (largeur fixe à 600px)
        display_width = 600
        ratio = display_width / image_to_predict.width
        display_height = int(image_to_predict.height * ratio)
        disp_image = image_to_predict.resize((display_width, display_height))
        
        # Conversion de l'image en Base64 (nécessaire pour le fond du canvas)
        buffered = io.BytesIO()
        disp_image.convert('RGB').save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        img_data_uri = f"data:image/jpeg;base64,{img_str}"

        # --- MISE EN PAGE HORIZONTALE (IMAGE + ACCESSIBILITÉ) ---
        col_canvas, col_options = st.columns([2, 1], gap="large")

        with col_options:
            st.markdown("### ♿ Accessibilité (WCAG)")
            st.info("Utilisez ces champs si vous ne pouvez pas utiliser la souris.")
            
            # Grille 2x2 pour les coordonnées
            c1, c2 = st.columns(2)
            man_xmin = c1.number_input("X Min", min_value=0, max_value=image_to_predict.width, key="box_xmin")
            man_ymin = c2.number_input("Y Min", min_value=0, max_value=image_to_predict.height, key="box_ymin")
            
            c3, c4 = st.columns(2)
            man_xmax = c3.number_input("X Max", min_value=0, max_value=image_to_predict.width, key="box_xmax")
            man_ymax = c4.number_input("Y Max", min_value=0, max_value=image_to_predict.height, key="box_ymax")
            
            use_manual_box = (man_xmax > man_xmin) and (man_ymax > man_ymin)

            use_manual_box = (man_xmax > man_xmin) and (man_ymax > man_ymin)

            # --- FONCTIONS DE RAPPEL ---
            def confirm_box():
                if use_manual_box:
                    # On ajoute la boîte actuelle à la liste permanente
                    new_box = {
                        "type": "rect",
                        "left": st.session_state.box_xmin * ratio,
                        "top": st.session_state.box_ymin * ratio,
                        "width": (st.session_state.box_xmax - st.session_state.box_xmin) * ratio,
                        "height": (st.session_state.box_ymax - st.session_state.box_ymin) * ratio,
                        "fill": "rgba(0,0,0,0)",
                        "stroke": "#00FF00",
                        "strokeWidth": 3,
                        "selectable": True
                    }
                    st.session_state.manual_boxes.append(new_box)
                    # Reset des champs (les clés de widgets)
                    st.session_state.box_xmin = 0
                    st.session_state.box_ymin = 0
                    st.session_state.box_xmax = 0
                    st.session_state.box_ymax = 0
                    st.session_state.canvas_key += 1

            def reset_all():
                st.session_state.box_xmin = 0
                st.session_state.box_ymin = 0
                st.session_state.box_xmax = 0
                st.session_state.box_ymax = 0
                st.session_state.manual_boxes = []
                st.session_state.canvas_key += 1

            # --- BOUTONS ---
            st.button("➕ Ajouter cette boîte", on_click=confirm_box, use_container_width=True, type="primary", disabled=not use_manual_box)
            st.button("🗑️ Tout réinitialiser", on_click=reset_all, use_container_width=True)
            
            edit_mode = st.checkbox("Mode Modification / Suppression 🖱️", key="edit_box")

        with col_canvas:
            st.markdown("**Tracé de la boîte englobante :**")
            
            # Injection de la boîte manuelle dans le canvas
            draw_state = {
                "version": "4.4.0",
                "objects": [
                    {
                        "type": "image",
                        "originX": "left", "originY": "top",
                        "left": 0, "top": 0,
                        "width": display_width, "height": display_height,
                        "src": img_data_uri,
                        "selectable": False, "evented": False
                    }
                ],
                "background": "transparent"
            }

            # On ajoute d'abord toutes les boîtes déjà confirmées
            draw_state["objects"].extend(st.session_state.manual_boxes)

            # On ajoute la boîte en cours de saisie (pour prévisualisation)
            if use_manual_box:
                draw_state["objects"].append({
                    "type": "rect",
                    "left": man_xmin * ratio,
                    "top": man_ymin * ratio,
                    "width": (man_xmax - man_xmin) * ratio,
                    "height": (man_ymax - man_ymin) * ratio,
                    "fill": "rgba(0,0,0,0)",
                    "stroke": "#00FF00",
                    "strokeWidth": 3,
                    "selectable": True,
                    "opacity": 0.6  # Légère transparence pour indiquer que c'est une prévisualisation
                })

            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=3,
                stroke_color="#00FF00",
                background_image=None,
                background_color="",
                height=display_height,
                width=display_width,
                drawing_mode="transform" if edit_mode else "rect",
                initial_drawing=draw_state,
                key=f"canvas_box_{st.session_state.canvas_key}"
            )

        # --- ANALYSE DU RÉSULTAT ---
        # On filtre uniquement les vrais rectangles dessinés par l'utilisateur (on ignore l'image de fond)
        rects = [obj for obj in canvas_result.json_data["objects"] if obj["type"] == "rect"] if canvas_result.json_data else []
        
        if len(rects) > 0:
            last_obj = rects[-1]
            left = last_obj["left"]
            top = last_obj["top"]
            width = last_obj["width"] * last_obj.get("scaleX", 1)
            height = last_obj["height"] * last_obj.get("scaleY", 1)
            
            xmin = int(left / ratio)
            ymin = int(top / ratio)
            xmax = int((left + width) / ratio)
            ymax = int((top + height) / ratio)
            
            prompt_data = {'box': [xmin, ymin, xmax, ymax]}
            
            is_manual = (xmin == man_xmin and ymin == man_ymin and xmax == man_xmax and ymax == man_ymax) if use_manual_box else False
            st.success(f"Boîte {'manuelle (WCAG)' if is_manual else 'dessinée (Souris)'} enregistrée : X:{xmin}, Y:{ymin}, W:{int(width/ratio)}, H:{int(height/ratio)}")
        else:
            st.info("Utilisez la souris pour dessiner à gauche ou les champs à droite pour l'accessibilité.")
            
    st.header("3. Résultat de la prédiction")
    if image_to_predict is not None:
        if st.button("Lancer la segmentation 🚀", type="primary"):
            with st.spinner('Analyse par le modèle MedSAM en cours... (Cela peut prendre quelques secondes)'):
                import requests
                import json
                import io
                import base64
                
                # Convertir l'image PIL en bytes pour l'envoi HTTP
                img_byte_arr = io.BytesIO()
                image_to_predict.convert('RGB').save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Préparer la requête (multipart/form-data)
                files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
                data = {'prompt': json.dumps(prompt_data)} if prompt_data else {}
                
                try:
                    # Envoi à l'API locale
                    response = requests.post("http://127.0.0.1:8000/predict/", files=files, data=data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        mask_base64 = result['mask_base64']
                        score = result.get('score', 0.0)
                        
                        # Décoder le masque binarisé
                        mask_data = base64.b64decode(mask_base64)
                        mask_image = Image.open(io.BytesIO(mask_data))
                        
                        st.success(f'Segmentation terminée avec succès ! (Confiance MedSAM : {score:.2f})')
                        
                        # Afficher côte à côte
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image_to_predict, caption="Image Originale", use_container_width=True)
                        with col2:
                            st.image(mask_image, caption="Masque Prédit (MedSAM)", use_container_width=True)
                    else:
                        st.error(f"Erreur API ({response.status_code}) : {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Impossible de se connecter à l'API. Vérifiez que le serveur uvicorn est bien lancé.")
    elif input_method == "Choisir un exemple du dataset":
         if st.button("Lancer la segmentation 🚀", type="primary"):
             st.warning("Veuillez implémenter le chargement des images d'exemple.")
