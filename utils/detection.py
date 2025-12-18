import streamlit as st
import numpy as np
import pandas as pd
import re
from utils.config import cliente, categorias, CSV_REGISTROS
from utils.helpers import agregar_registro
from ultralytics import YOLO
from pathlib import Path
import os

def extraer_peso_estimado(respuesta_texto):
    # Extrae el peso total estimado en kg de la respuesta de Gemini
    patrones = [
        r"Peso Total Estimado[:\s]*([\d\.]+)\s*kg",
        r"peso total[:\s]*([\d\.]+)\s*kg",
        r"([\d\.]+)\s*kg"
    ]
    for patron in patrones:
        match = re.search(patron, respuesta_texto, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return 0.0

def obtener_resumen_datos(df, datos_categorias, conteo_actual):
    # Genera un resumen de los datos y del conteo actual para el prompt de Gemini
    resumen_csv = "Historial Total de Desechos (Top 5):\n"
    if not df.empty:
        top_clases = df["class"].value_counts().head(5)
        resumen_csv += top_clases.to_string()
    else:
        resumen_csv += "Aún no hay registros históricos."
        
    info_categorias = "Información de las categorías de desecho (JSON):\n"
    for nombre in datos_categorias.get("names", []):
        info = datos_categorias["info"].get(nombre, {})
        info_categorias += f"- **{nombre}**: Reciclable: {info.get('recyclable', 'N/A')}.\n"
    
    resumen_actual = "Conteo de la FOTO ACTUAL:\n"
    resumen_actual += conteo_actual.to_string()

    return f"{resumen_csv}\n\n{info_categorias}\n\n{resumen_actual}"

# Global para cachear el modelo
modelo_cache = None

def ejecutar_deteccion_analisis_gemini(imagen, tipo_fuente, nombre_archivo, sector, coordenadas, umbral_confianza, usar_gemini=True):
    # Ejecuta YOLO, guarda los registros con GPS y llama a Gemini
    
    global modelo_cache
    if modelo_cache is None:
        RUTA_MODELO = Path(os.getcwd()) / "models" / "best.pt"
        try:
            modelo_cache = YOLO(str(RUTA_MODELO))
        except Exception as e:
            st.error(f"Error al cargar el modelo YOLO: {e}")
            return
    
    modelo = modelo_cache
    resultados = modelo(np.array(imagen), conf=umbral_confianza)[0]
    
    conteo_actual = {nombre: 0 for nombre in modelo.names.values()}
    registros_para_csv = []
    
    for caja in resultados.boxes:
        id_clase = int(caja.cls)
        confianza = float(caja.conf)
        nombre_clase = modelo.names.get(id_clase, f"Clase ID {id_clase}")
        
        conteo_actual[nombre_clase] += 1
        
        registros_para_csv.append({
            'source': tipo_fuente, 'file_name': nombre_archivo, 'sector': sector, 
            'coordenadas': coordenadas, 'class': nombre_clase, 'confidence': confianza
        })

    total_detectado = sum(conteo_actual.values())
    st.subheader(f"Detección completada: {total_detectado} ítems encontrados (Conf > {umbral_confianza*100:.0f}%)")

    peso_estimado_total = 0.0

    imagen_salida = resultados.plot()
    st.image(imagen_salida, caption=f"Imagen con {total_detectado} desechos detectados", width='stretch')

    df_conteo = pd.Series(conteo_actual).rename_axis('class').to_frame('count').sort_values('count', ascending=False)
    st.markdown("### Reporte de Cuantificación por Foto")
    st.dataframe(df_conteo, width='stretch')

    st.markdown("---")
    if cliente and total_detectado > 0 and usar_gemini:
        st.subheader("Análisis Avanzado")
        df_historial = pd.read_csv(CSV_REGISTROS) 
        resumen_datos = obtener_resumen_datos(df_historial, categorias, df_conteo['count'])
        
        tarea = (
            f"Analiza la composición de desechos encontrados en esta foto (Conteo de la FOTO ACTUAL en el sector '{sector}'). "
            f"Responde en formato Markdown:\n"
            f"1. **Peso Total Estimado (kg)**: Estima el peso total aproximado de *todos* los desechos detectados en esta foto basándote en conocimientos generales de pesos típicos de desechos. Muestra la estimación en kilogramos (kg). (Ej: 'Peso Total Estimado: 15.3 kg').\n"
            f"2. **Prioridad de Reciclaje Inmediato**: Indica las 2-3 categorías más valiosas para el reciclaje detectadas en esta foto y sugiere la acción más inmediata para el municipio (ej: coordinar camión específico, notificar centro de acopio).\n"
            f"3. **Riesgo Ambiental Clave**: Indica si la composición (Orgánico vs. Plástico, etc.) representa un problema de salud pública/contaminación del agua más urgente y por qué. "
        )
        prompt_completo = f"CONTEXTO DE DATOS:\n{resumen_datos}\n\nTAREA:\n{tarea}"
        
        try:
            with st.spinner('Generando análisis avanzado para toma de decisiones...'):
                respuesta = cliente.models.generate_content(
                    model='gemini-2.5-flash', contents=prompt_completo
                )
            peso_estimado_total = extraer_peso_estimado(respuesta.text)
            st.success(respuesta.text)
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                st.warning("**Servidores de Gemini sobrecargados**\n\nLos servidores de Google están temporalmente saturados. El análisis avanzado estará disponible en unos minutos. Los datos de detección se guardaron correctamente.")
            elif "400" in error_msg or "INVALID_ARGUMENT" in error_msg:
                st.error("❌ **Error de configuración**\n\nRevisa tu clave de API de Gemini. Puede estar expirada o ser inválida.")
            elif "403" in error_msg or "PERMISSION_DENIED" in error_msg:
                st.error("🚫 **Acceso denegado**\n\nVerifica que tu clave de API tenga permisos para usar Gemini.")
            else:
                st.warning(f"⚠️ **Error en análisis avanzado**\n\n{error_msg}\n\nLos datos básicos de detección se guardaron correctamente.")
            peso_estimado_total = 0.0
    else:
        pass

    # Si no se obtuvo estimación de peso, calcular una aproximación simple
    if peso_estimado_total == 0.0 and total_detectado > 0:
        peso_estimado_total = total_detectado * 0.1  # Estimación de 100g por ítem promedio
        st.info(f"Estimación simple de peso: {peso_estimado_total:.1f} kg (basado en {total_detectado} ítems a 100g cada uno)")

    # Agregar registros con el peso estimado
    try:
        for registro in registros_para_csv:
             agregar_registro(CSV_REGISTROS, registro['source'], registro['file_name'], 
                           registro['sector'], registro['coordenadas'], 
                           registro['class'], registro['confidence'], peso_estimado_total)
    except Exception as e:
        st.warning(f"Error al guardar registros en CSV: {e}. Los datos de detección se procesaron correctamente.")

    # Retornar resultados para mostrar métricas
    return {
        'total_items': total_detectado,
        'peso_total': peso_estimado_total,
        'desglose': df_conteo.to_dict('records'),
        'imagen_procesada': imagen_salida
    }