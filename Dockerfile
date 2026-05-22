FROM python:3.10-slim

WORKDIR /app

# Copier d'abord les requirements pour exploiter le cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du dépôt (qui correspond au contenu de Src)
COPY . /app

# Se placer dans le dossier de l'API
WORKDIR /app/dashboard/api

# Hugging Face Spaces utilise par défaut le port 7860
EXPOSE 7860

# Commande de lancement de l'API FastAPI (Uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
