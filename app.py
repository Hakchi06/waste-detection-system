import streamlit as st
from PIL import Image
import folium
from streamlit_folium import st_folium
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
from utils.config import CSV_REGISTROS
from utils.helpers import asegurar_archivo_registros, calcular_impacto_ambiental, obtener_centros_reciclaje_panama
from utils.detection import ejecutar_deteccion_analisis_gemini
from utils.dashboard import mostrar_dashboard

# Asegurar que el archivo de registros exista
asegurar_archivo_registros(CSV_REGISTROS)

# Configuración de página mejorada
st.set_page_config(
    page_title='Sistema de Gestión de Residuos',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Estilos CSS personalizados para mejor apariencia
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .alert-card {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Función para obtener ubicación actual
def obtener_ubicacion_actual():
    try:
        response = requests.get('http://ip-api.com/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"{data.get('lat', 8.98)}, {data.get('lon', -79.52)}"
    except:
        pass
    return "8.98, -79.52"


# Función para mostrar mapa interactivo
def mostrar_mapa_residuos(df_filtrado):
    if df_filtrado.empty:
        st.info("No hay datos para mostrar en el mapa")
        return

    # Crear mapa centrado en Panamá
    mapa = folium.Map(location=[8.98, -79.52], zoom_start=10)

    # Agregar marcadores para cada registro
    for _, row in df_filtrado.iterrows():
        try:
            lat, lon = map(float, row['coordenadas'].split(','))
            popup_text = f"""
            <b>Sector:</b> {row['sector']}<br>
            <b>Tipo:</b> {row['class']}<br>
            <b>Peso:</b> {row.get('peso_total_foto_kg', 'N/A')} kg<br>
            <b>Fecha:</b> {row['timestamp'][:10]}
            """

            # Color según tipo de residuo
            color_map = {
                'PLASTIC': 'blue',
                'METAL': 'gray',
                'PAPER': 'green',
                'GLASS': 'lightblue',
                'BIODEGRADABLE': 'orange',
                'CARDBOARD': 'brown'
            }
            color = color_map.get(row['class'], 'red')

            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                popup=popup_text,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(mapa)
        except:
            continue

    st_folium(mapa, width=700, height=500)

# Navegación principal
st.sidebar.title("Gestion de Residuos")
pagina = st.sidebar.radio(
    "Selecciona una sección:",
    ["Registro de Residuos", "Dashboard Analítico", "Centro Educativo"]
)

if pagina == "Registro de Residuos":
    # Header mejorado
    st.markdown("""
    <div class="main-header">
        <h1>Sistema de Gestión de Residuos</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("Registro de Residuos")

    # Controles mejorados
    col_inputs, col_upload = st.columns([1, 1])

    with col_inputs:
        st.subheader("Información del Sitio")

        # Ubicación automática
        if st.button("Obtener Ubicación Actual"):
            ubicacion_actual = obtener_ubicacion_actual()
            st.session_state.ubicacion_actual = ubicacion_actual
            st.success(f"Ubicación obtenida: {ubicacion_actual}")

        entrada_sector = st.text_input(
            "Sector / Corregimiento:",
            key='sector_in',
            value='Ciudad de Panamá',
            help="Ingresa el sector o corregimiento donde se encuentra el residuo"
        )

        entrada_coordenadas = st.text_input(
            "Coordenadas GPS (Lat, Long):",
            key='gps_in',
            value=st.session_state.get('ubicacion_actual', '8.98, -79.52'),
            help="Formato: latitud, longitud (ej: 8.98, -79.52)"
        )

        # Tipo de reporte
        tipo_reporte = st.selectbox(
            "Tipo de Reporte:",
            ["Limpieza Urbana", "Punto Crítico", "Recolección Programada", "Otro"],
            help="Selecciona el tipo de situación que estás reportando"
        )

    with col_upload:
        st.subheader("Captura de Imagen")

        # Opciones de captura
        metodo_captura = st.radio(
            "Método de captura:",
            ["Subir archivo", "Usar cámara"],
            horizontal=True
        )

        if metodo_captura == "Subir archivo":
            uploaded = st.file_uploader(
                "Selecciona una imagen del residuo",
                type=["jpg", "jpeg", "png"],
                help="Sube una foto clara del área con residuos"
            )
            img = Image.open(uploaded).convert("RGB") if uploaded else None
        else:
            camera_image = st.camera_input("Captura con la cámara del dispositivo")
            img = Image.open(camera_image).convert("RGB") if camera_image else None

        # Barra lateral para configuración de detección
        st.sidebar.header("Configuración de Detección")
        umbral_confianza = st.sidebar.slider(
            "Umbral de Confianza Mínima:",
            0.0, 1.0, 0.5, 0.05,
            key='conf_threshold',
            help="Solo se registrarán residuos detectados con certeza superior a este valor"
        )
        modelo_ia = st.sidebar.selectbox(
            "Modelo de IA preferido:",
            ["YOLOv8 + Gemini", "Solo YOLOv8"],
            key="modelo_ia",
            help="Elige si usar análisis avanzado con Gemini o solo detección con YOLO"
        )

        # Procesar imagen si está disponible
        if img is not None:
            imagen_mostrar = img
            procesada = False

            if st.button("Analizar Residuos", type="primary", use_container_width=True):
                with st.spinner("Analizando imagen con IA..."):
                    try:
                        fuente = "upload" if metodo_captura == "Subir archivo" else "webcam"
                        nombre_archivo = getattr(uploaded, "name", "captura_camara") if metodo_captura == "Subir archivo" else "captura_webcam"

                        usar_gemini = modelo_ia == "YOLOv8 + Gemini"
                        resultado = ejecutar_deteccion_analisis_gemini(
                            img, fuente, nombre_archivo,
                            entrada_sector, entrada_coordenadas, umbral_confianza, usar_gemini
                        )

                        if resultado:
                            total_items = resultado.get('total_items', 0)
                            st.success(f"✅ Análisis completado exitosamente! Se detectaron {total_items} ítems.")

                            # Mostrar imagen procesada si está disponible
                            if 'imagen_procesada' in resultado:
                                imagen_mostrar = resultado['imagen_procesada']
                                procesada = True

                            # Mostrar resultados
                            st.subheader("Resultados del Análisis")
                            col_res1, col_res2, col_res3 = st.columns(3)

                            with col_res1:
                                peso_estimado = resultado.get('peso_total', 0)
                                st.metric("Peso Estimado Total", f"{peso_estimado:.1f} kg")
                            with col_res2:
                                impacto_co2 = calcular_impacto_ambiental(pd.DataFrame([resultado]))
                                st.metric("CO₂ Ahorrado", f"{impacto_co2:.1f} kg")
                            with col_res3:
                                st.metric("Confianza Mínima", f"{umbral_confianza*100:.0f}%")

                        else:
                            st.error("❌ Error: El modelo de detección YOLO no se pudo cargar. Verifica que el archivo 'models/best.pt' exista y sea válido.")

                    except Exception as e:
                        st.error(f"❌ Error durante el análisis: {str(e)}")


elif pagina == "Dashboard Analítico":
    mostrar_dashboard()

elif pagina == "Centro Educativo":
    st.markdown("""
    <div class="main-header">
        <h1>Centro Educativo Ambiental</h1>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Guía de Reciclaje", "Centros de Reciclaje", "Horarios de Recolección", "Impacto Ambiental"])

    with tab1:
        st.header("Guía de Reciclaje en Panamá")

        categorias_reciclaje = {
            "Residuos Orgánicos": {
                "descripcion": "Restos de comida, cáscaras de frutas, vegetales, etc.",
                "como_reciclar": "Deposítalos en contenedores de compostaje o residuos orgánicos",
                "beneficio": "Se convierten en abono natural para plantas"
            },
            "Cartón y Papel": {
                "descripcion": "Cajas, periódicos, revistas, papel de oficina",
                "como_reciclar": "Aplasta las cajas y deposítalas en contenedores azules",
                "beneficio": "Reduce la tala de árboles"
            },
            "Plásticos": {
                "descripcion": "Botellas PET, envases plásticos, bolsas",
                "como_reciclar": "Enjuaga y deposita en contenedores amarillos",
                "beneficio": "Reduce contaminación marina"
            },
            "Metales": {
                "descripcion": "Latas de aluminio, latas de conserva",
                "como_reciclar": "Aplasta y deposita en contenedores grises",
                "beneficio": "Ahorra energía en producción"
            },
            "Vidrio": {
                "descripcion": "Botellas y frascos de vidrio",
                "como_reciclar": "Enjuaga y deposita en contenedores verdes",
                "beneficio": "El vidrio se puede reciclar infinitamente"
            }
        }

        for categoria, info in categorias_reciclaje.items():
            with st.expander(categoria):
                st.write(f"**Descripción:** {info['descripcion']}")
                st.write(f"**Cómo reciclar:** {info['como_reciclar']}")
                st.write(f"**Beneficio ambiental:** {info['beneficio']}")

    with tab2:
        st.header("Centros de Reciclaje en Panamá")

        centros = obtener_centros_reciclaje_panama()

        for centro in centros:
            with st.expander(f"Centro {centro['nombre']}"):
                st.write(f"**Dirección:** {centro['direccion']}")
                st.write(f"**Horario:** {centro['horario']}")
                st.write(f"**Teléfono:** {centro['telefono']}")
                st.write(f"**Materiales aceptados:** {', '.join(centro['materiales'])}")

                # Botón para ver en mapa
                if st.button(f"Ver ubicación de {centro['nombre']}", key=f"map_{centro['nombre']}"):
                    st.components.v1.html(f"""
                    <iframe src="https://www.google.com/maps/embed/v1/place?key=YOUR_API_KEY&q={centro['direccion'].replace(' ', '+')}" width="100%" height="300" frameborder="0" style="border:0" allowfullscreen></iframe>
                    """)

    with tab3:
        st.header("Horarios de Recolección por Sector")

        horarios_panama = {
            "Ciudad de Panamá": {
                "Lunes": "Residuos orgánicos y reciclables",
                "Martes": "Residuos sólidos mixtos",
                "Miércoles": "Residuos orgánicos y reciclables",
                "Jueves": "Residuos sólidos mixtos",
                "Viernes": "Residuos orgánicos y reciclables",
                "Sábado": "Residuos sólidos mixtos",
                "Domingo": "Sin recolección"
            },
            "San Miguelito": {
                "Lunes": "Reciclables",
                "Martes": "Residuos sólidos",
                "Miércoles": "Reciclables",
                "Jueves": "Residuos sólidos",
                "Viernes": "Reciclables",
                "Sábado": "Residuos sólidos",
                "Domingo": "Sin recolección"
            }
        }

        sector_seleccionado = st.selectbox("Selecciona tu sector:", list(horarios_panama.keys()))

        st.subheader(f"Horarios para {sector_seleccionado}")
        horario_df = pd.DataFrame(list(horarios_panama[sector_seleccionado].items()), columns=['Día', 'Tipo de Recolección'])
        st.table(horario_df)

    with tab4:
        st.header("Calculadora de Impacto Ambiental")

        st.write("Descubre cuánto impacto ambiental generas y cómo puedes reducirlo:")

        col_calc1, col_calc2 = st.columns(2)

        with col_calc1:
            residuos_semanales = st.slider("Kg de residuos generados por semana:", 1, 50, 10)
            porcentaje_reciclable = st.slider("% de residuos reciclables:", 0, 100, 30)

        with col_calc2:
            # Cálculos de impacto
            co2_anual = residuos_semanales * 52 * 0.5  # kg CO2 por kg de residuos
            arboles_salvados = (residuos_semanales * porcentaje_reciclable / 100) * 52 / 10  # árboles salvados por año
            agua_ahorrada = residuos_semanales * 52 * 100  # litros de agua

            st.metric("CO₂ evitado al reciclar (kg/año)", f"{co2_anual:.1f}")
            st.metric("Árboles salvados por reciclaje", f"{arboles_salvados:.1f}")
            st.metric("Agua ahorrada (litros/año)", f"{agua_ahorrada:,}")