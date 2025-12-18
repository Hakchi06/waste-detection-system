import streamlit as st
import numpy as np
import pandas as pd
import re
from src.config.settings import cliente, categorias, CSV_REGISTROS
from src.data.manager import DataManager
from ultralytics import YOLO
from pathlib import Path
import os

class WasteDetector:
    def __init__(self):
        self.model_cache = None
        self.data_manager = DataManager(CSV_REGISTROS)

    def extract_estimated_weight(self, response_text):
        # Extrae el peso total estimado en kg de la respuesta de Gemini
        patterns = [
            r"Peso Total Estimado[:\s]*([\d\.]+)\s*kg",
            r"peso total[:\s]*([\d\.]+)\s*kg",
            r"([\d\.]+)\s*kg"
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return 0.0

    def get_data_summary(self, df, category_data, current_count):
        # Genera un resumen de los datos y del conteo actual para el prompt de Gemini
        csv_summary = "Historial Total de Desechos (Top 5):\n"
        if not df.empty:
            top_classes = df["class"].value_counts().head(5)
            csv_summary += top_classes.to_string()
        else:
            csv_summary += "Aún no hay registros históricos."

        category_info = "Información de las categorías de desecho (JSON):\n"
        for name in category_data.get("names", []):
            info = category_data["info"].get(name, {})
            category_info += f"- **{name}**: Reciclable: {info.get('recyclable', 'N/A')}.\n"

        current_summary = "Conteo de la FOTO ACTUAL:\n"
        current_summary += current_count.to_string()

        return f"{csv_summary}\n\n{category_info}\n\n{current_summary}"

    def load_model(self):
        if self.model_cache is None:
            model_path = Path(os.getcwd()) / "models" / "best.pt"
            try:
                self.model_cache = YOLO(str(model_path))
            except Exception as e:
                st.error(f"Error al cargar el modelo YOLO: {e}")
                return None
        return self.model_cache

    def detect_and_analyze(self, image, source_type, file_name, sector, coordinates, confidence_threshold, use_gemini=True):
        # Ejecuta YOLO, guarda los registros con GPS y llama a Gemini

        model = self.load_model()
        if not model:
            return None

        results = model(np.array(image), conf=confidence_threshold)[0]

        current_count = {name: 0 for name in model.names.values()}
        records_for_csv = []

        for box in results.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            class_name = model.names.get(class_id, f"Clase ID {class_id}")

            current_count[class_name] += 1

            records_for_csv.append({
                'source': source_type, 'file_name': file_name, 'sector': sector,
                'coordenadas': coordinates, 'class': class_name, 'confidence': confidence
            })

        total_detected = sum(current_count.values())
        st.subheader(f"Detección completada: {total_detected} ítems encontrados (Conf > {confidence_threshold*100:.0f}%)")

        estimated_total_weight = 0.0

        processed_image = results.plot()
        st.image(processed_image, caption=f"Imagen con {total_detected} desechos detectados", width='stretch')

        count_df = pd.Series(current_count).rename_axis('class').to_frame('count').sort_values('count', ascending=False)
        st.markdown("### Reporte de Cuantificación por Foto")
        st.dataframe(count_df, width='stretch')

        st.markdown("---")
        if cliente and total_detected > 0 and use_gemini:
            st.subheader("Análisis Avanzado")
            historical_df = pd.read_csv(CSV_REGISTROS)
            data_summary = self.get_data_summary(historical_df, categorias, count_df['count'])

            task = (
                f"Analiza la composición de desechos encontrados en esta foto (Conteo de la FOTO ACTUAL en el sector '{sector}'). "
                f"Responde en formato Markdown:\n"
                f"1. **Peso Total Estimado (kg)**: Estima el peso total aproximado de *todos* los desechos detectados en esta foto basándote en conocimientos generales de pesos típicos de desechos. Muestra la estimación en kilogramos (kg). (Ej: 'Peso Total Estimado: 15.3 kg').\n"
                f"2. **Prioridad de Reciclaje Inmediato**: Indica las 2-3 categorías más valiosas para el reciclaje detectadas en esta foto y sugiere la acción más inmediata para el municipio (ej: coordinar camión específico, notificar centro de acopio).\n"
                f"3. **Riesgo Ambiental Clave**: Indica si la composición (Orgánico vs. Plástico, etc.) representa un problema de salud pública/contaminación del agua más urgente y por qué. "
            )
            full_prompt = f"CONTEXTO DE DATOS:\n{data_summary}\n\nTAREA:\n{task}"

            try:
                with st.spinner('Generando análisis avanzado para toma de decisiones...'):
                    response = cliente.models.generate_content(
                        model='gemini-2.5-flash', contents=full_prompt
                    )
                estimated_total_weight = self.extract_estimated_weight(response.text)
                st.success(response.text)
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
                estimated_total_weight = 0.0
        else:
            pass

        # Guardar registros
        for record in records_for_csv:
            self.data_manager.add_record(
                record['source'], record['file_name'], record['sector'],
                record['coordenadas'], record['class'], record['confidence'], estimated_total_weight / len(records_for_csv) if records_for_csv else 0
            )

        return {
            'total_items': total_detected,
            'peso_total': estimated_total_weight,
            'conteo': current_count
        }