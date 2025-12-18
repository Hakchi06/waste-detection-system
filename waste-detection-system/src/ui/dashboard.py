import streamlit as st
import pandas as pd
import altair as alt
import datetime
from src.config.settings import categorias, CSV_REGISTROS
from src.data.manager import DataManager
import folium
from streamlit_folium import st_folium

data_manager = DataManager(CSV_REGISTROS)

def mostrar_mapa_residuos(df_filtrado, mostrar_peso=True):
    if df_filtrado.empty:
        st.info("No hay datos para mostrar en el mapa")
        return

    # Crear mapa centrado en Panamá
    mapa = folium.Map(location=[8.98, -79.52], zoom_start=10)

    # Agregar marcadores circulares para cada residuo, coloreados por tipo
    puntos_agregados = 0
    for _, row in df_filtrado.iterrows():
        try:
            lat, lon = map(float, [s.strip() for s in row['coordenadas'].split(',')])
            peso_text = f"<b>Peso:</b> {row.get('peso_total_foto_kg', 'N/A')} kg<br>" if mostrar_peso else ""
            popup_text = f"""
            <b>Sector:</b> {row['sector']}<br>
            <b>Tipo:</b> {row['class']}<br>
            {peso_text}
            <b>Fecha:</b> {row['timestamp'].strftime('%Y-%m-%d')}
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
            puntos_agregados += 1
        except Exception as e:
            continue

    st.write(f"Puntos agregados al mapa: {puntos_agregados}")
    st_folium(mapa, width=700, height=500)

def mostrar_dashboard():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;'>
        <h1>Dashboard Analítico de Residuos</h1>
    </div>
    """, unsafe_allow_html=True)

    df = pd.read_csv(CSV_REGISTROS)

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date

        df_filtrado = df.copy()

        # Métricas principales mejoradas
        if not df_filtrado.empty:
            # Cálculos de métricas
            total_general = df_filtrado["class"].value_counts().sum()
            reciclables = ['PLASTIC', 'METAL', 'PAPER', 'GLASS', 'CARDBOARD']
            total_reciclable = df_filtrado[df_filtrado["class"].isin(reciclables)]["class"].value_counts().sum()
            porcentaje_reciclable = (total_reciclable / total_general) * 100 if total_general > 0 else 0
            avg_confidence = df_filtrado['confidence'].mean() * 100
            total_peso = df_filtrado['peso_total_foto_kg'].sum() if 'peso_total_foto_kg' in df_filtrado.columns else 0
            impacto_co2 = data_manager.calculate_environmental_impact(df_filtrado)

            # Alertas inteligentes
            porc_residuales = df_filtrado[~df_filtrado["class"].isin(reciclables)]["class"].value_counts().sum() / total_general * 100 if total_general > 0 else 0

            if porc_residuales > 40:
                st.error(f"ALERTA: Residuales ({porc_residuales:.1f}%) exceden el 40%. Riesgo Sanitario alto.")
            elif porcentaje_reciclable > 60:
                st.success(f"Excelente: {porcentaje_reciclable:.1f}% de reciclables detectados!")

            # Métricas en tarjetas mejoradas
            st.markdown("### Indicadores Clave")

            col_metrica1, col_metrica2, col_metrica3, col_metrica4 = st.columns(4)

            with col_metrica1:
                st.metric(
                    label="Total de Ítems",
                    value=f"{total_general:,}",
                    delta="Registrados"
                )

            with col_metrica2:
                st.metric(
                    label="% Reciclable",
                    value=f"{porcentaje_reciclable:.1f}%",
                    delta="Eficiencia"
                )

            with col_metrica3:
                st.metric(
                    label="Peso Total Estimado",
                    value=f"{total_peso:.1f} kg",
                    delta="Acumulado"
                )

            with col_metrica4:
                st.metric(
                    label="CO₂ Ahorrado",
                    value=f"{impacto_co2:.1f} kg",
                    delta="Anual"
                )

            # Gráficos mejorados
            st.markdown("### Análisis Visual")

            col_graf1, col_graf2 = st.columns(2)

            with col_graf1:
                # Gráfico de distribución por tipo
                chart_data = df_filtrado["class"].value_counts().reset_index()
                chart_data.columns = ['Tipo', 'Cantidad']

                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('Tipo', sort='-y'),
                    y='Cantidad',
                    color=alt.condition(
                        alt.datum.Tipo.isin(reciclables),
                        alt.value('#10b981'),
                        alt.value('#ef4444')
                    )
                ).properties(
                    title="Distribución por Tipo de Residuo",
                    width=300,
                    height=250
                )
                st.altair_chart(chart, use_container_width=True)

            with col_graf2:
                # Gráfico de reciclables vs no reciclables
                reciclable_data = pd.DataFrame({
                    'Categoría': ['Reciclables', 'No Reciclables'],
                    'Cantidad': [total_reciclable, total_general - total_reciclable]
                })

                pie_chart = alt.Chart(reciclable_data).mark_arc(innerRadius=50).encode(
                    theta='Cantidad',
                    color=alt.Color('Categoría', scale=alt.Scale(domain=['Reciclables', 'No Reciclables'], range=['#10b981', '#ef4444'])),
                    tooltip=['Categoría', 'Cantidad']
                ).properties(
                    title="Reciclables vs No Reciclables",
                    width=300,
                    height=250
                )
                st.altair_chart(pie_chart, use_container_width=True)

            # Mapa interactivo
            st.markdown("### Mapa de Ubicaciones")

            col_map_filt1, col_map_filt2 = st.columns(2)

            with col_map_filt1:
                vista_mapa = st.radio(
                    "Vista del mapa:",
                    ["Todos los residuos", "Solo reciclables", "Solo no reciclables"],
                    key="vista_mapa"
                )

            with col_map_filt2:
                mostrar_peso = st.checkbox("Mostrar peso en popups", value=True, key="mostrar_peso")

            # Aplicar filtro de vista
            df_mapa = df_filtrado.copy()
            if vista_mapa == "Solo reciclables":
                df_mapa = df_mapa[df_mapa["class"].isin(reciclables)]
            elif vista_mapa == "Solo no reciclables":
                df_mapa = df_mapa[~df_mapa["class"].isin(reciclables)]

            mostrar_mapa_residuos(df_mapa, mostrar_peso)

        else:
            st.warning("⚠️ No hay datos disponibles.")

    else:
        st.info("ℹ️ No hay datos registrados aún. Comienza registrando residuos para ver el análisis.")