import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import json
from utils.helpers import append_record, ensure_records_file
from pathlib import Path
import plotly.express as px

# ===============================
# CONFIGURATION
# ===============================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best-classify.pt"
CATEGORIES_JSON = BASE_DIR / "data" / "categories.json"
RECORDS_CSV = BASE_DIR / "data" / "records.csv"

ensure_records_file(RECORDS_CSV)

@st.cache_resource
def load_model(path):
    return YOLO(str(path))

model = load_model(MODEL_PATH)

st.set_page_config(page_title='Clasificador de Desechos', layout='wide')
st.title('Clasificación de Desechos del Hogar')

# ===============================
# LOAD CATEGORY INFORMATION
# ===============================
with open(CATEGORIES_JSON, "r", encoding="utf-8") as f:
    categories = json.load(f)

names = categories.get("names", [])

# ===============================
# IMAGE UPLOAD SECTION
# ===============================
st.header("Clasificación mediante carga de imagen")

col1, col2 = st.columns([1,1])

with col1:
    uploaded = st.file_uploader(
        "Sube una imagen de un desecho",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Imagen cargada", use_column_width=True)

        # inferencia
        results = model(np.array(img))[0]
        cls_id = int(results.probs.top1)
        conf = float(results.probs.top1conf)

        class_name = model.names[cls_id]

        st.subheader(f"Predicción: {class_name} ({conf*100:.2f}%)")

        # guardar registro
        append_record(
            RECORDS_CSV,
            "upload",
            getattr(uploaded, "name", "uploaded_image"),
            class_name,
            conf
        )


with col2:
    st.header("Clasificación mediante webcam")
    camera_image = st.camera_input("Toma una fotografía")

    if camera_image is not None:
        img = Image.open(camera_image).convert("RGB")
        st.image(img, caption="Imagen capturada", use_column_width=True)

        results = model(np.array(img))[0]
        cls_id = int(results.probs.top1)
        conf = float(results.probs.top1conf)
        class_name = model.names[cls_id]

        st.subheader(f"Predicción: {class_name} ({conf*100:.2f}%)")

        append_record(
            RECORDS_CSV,
            "webcam",
            "webcam_capture",
            class_name,
            conf
        )


# ===============================
# RECORDS AND STATISTICS
# ===============================
st.markdown("---")
st.header("Historial y estadísticas de clasificación")

df = pd.read_csv(RECORDS_CSV)

st.subheader("Registros recientes")
st.dataframe(df.tail(50), use_container_width=True)

import altair as alt

st.subheader("Cantidad total por categoría")

if not df.empty:
    counts = df["class"].value_counts().reset_index()
    counts.columns = ["class", "count"]

    # Altair chart
    chart = (
        alt.Chart(counts)
        .mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
            opacity=0.85
        )
        .encode(
            x=alt.X("class:N", title="Categoría", sort="-y"),
            y=alt.Y("count:Q", title="Cantidad"),
            color=alt.Color(
                "class:N",
                scale=alt.Scale(
                    scheme="set2"   # paleta elegante y suave
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip("class:N", title="Categoría"),
                alt.Tooltip("count:Q", title="Cantidad")
            ]
        )
        .properties(
            width="container",
            height=400,
            title=""
        )
        .configure_title(
            fontSize=20,
            anchor="start"
        )
        .configure_axis(
            labelFontSize=13,
            titleFontSize=14
        )
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Aún no hay registros suficientes para generar la gráfica.")


# ===============================
# CATEGORY INFORMATION
# ===============================
st.markdown("---")
st.header("Información sobre las categorías de desechos")

for nm in names:
    info = categories["info"].get(nm, {})

    with st.expander(f"Información sobre {nm}"):
        st.write("Descripción:", info.get("description", ""))
        st.write("Cómo desecharlo:", info.get("handling", ""))
        st.write("¿Es reciclable?:", info.get("recyclable", ""))

st.markdown("---")
st.write("Archivo de registros:", str(RECORDS_CSV))
