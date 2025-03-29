# main.py
from fastapi import FastAPI
import uvicorn
from API.FastApi import router  # Importar los endpoints desde el archivo api/endpoints.py

# Crear la aplicación FastAPI
app = FastAPI()

# Incluir las rutas de predicción de phishing
app.include_router(router)

uvicorn.run(app, host="127.0.0.1", port=8000)
