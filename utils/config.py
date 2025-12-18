import os
from pathlib import Path
from dotenv import load_dotenv
import json
import streamlit as st
from ultralytics import YOLO
from google import genai

# Cargar variables de entorno
load_dotenv()

# Configuración de rutas
DIRECTORIO_BASE = Path(__file__).resolve().parent.parent
RUTA_MODELO = DIRECTORIO_BASE / "models" / "best.pt"
JSON_CATEGORIAS = DIRECTORIO_BASE / "data" / "categories.json"
CSV_REGISTROS = DIRECTORIO_BASE / "data" / "records_scm.csv"

# Configuración de Gemini
cliente = None
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        cliente = genai.Client(api_key=api_key)
    else:
        pass
except Exception as e:
    st.error(f"Error al inicializar Gemini: {e}")

# Cargar categorías
try:
    with open(JSON_CATEGORIAS, "r", encoding="utf-8") as f:
        categorias = json.load(f)
    nombres = categorias.get("names", [])
except FileNotFoundError:
    st.error(f"Error: No se encontró el archivo de categorías en {JSON_CATEGORIAS}.")
    nombres = []
    categorias = {}

# Cargar modelo YOLO
modelo = None