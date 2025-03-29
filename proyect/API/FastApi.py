# api/endpoints.py
from fastapi import APIRouter
from pydantic import BaseModel
import torch
import joblib  # Usado para cargar el vectorizador (si lo guardaste en .pkl)
import numpy as np
import os
import sys
sys.path.append("..")
from ia_tools.RNN import RNNNet
# Determinar la ruta base de tu proyecto (esto puede variar según tu estructura)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Cargar el modelo entrenado
model = torch.load(os.path.join(BASE_DIR, '..', 'ia_tools', 'models', 'rnn_model.pth'))  # Ruta a tu archivo .pth
model.eval()  # Establecer el modelo en modo de evaluación

# Cargar el vectorizador
vectorizer = joblib.load("vectorizer.pkl")  # Si tienes el vectorizador guardado como .pkl

# Crear el router de FastAPI
router = APIRouter()

class Email(BaseModel):
    text: str  # El cuerpo del correo

@router.post("/predict/")
async def predict_email(email: Email):
    # Preprocesar el texto del correo
    X = vectorizer.transform([email.text]).toarray()  # Vectorizar el texto del correo
    X_tensor = torch.tensor(X, dtype=torch.float32)  # Convertir a tensor

    # Hacer la predicción
    with torch.no_grad():
        output = model(X_tensor)
        prediction = torch.round(torch.sigmoid(output)).item()  # Convertir el output a 0 o 1

    # Retornar la predicción (1 = phishing, 0 = no phishing)
    return {"prediction": prediction}
