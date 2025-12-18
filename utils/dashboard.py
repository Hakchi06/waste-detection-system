import streamlit as st
import pandas as pd
import altair as alt
import datetime
from utils.config import categorias, CSV_REGISTROS
from utils.helpers import obtener_categoria_valor_reciclaje, generar_resumen_reporte, calcular_impacto_ambiental
import folium
from streamlit_folium import st_folium

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
            impacto_co2 = calcular_impacto_ambiental(df_filtrado)

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
                    help="Número total de residuos detectados"
                )

            with col_metrica2:
                st.metric(
                    label="% Reciclable",
                    value=f"{porcentaje_reciclable:.1f}%",
                    delta=f"{total_reciclable} items",
                    help="Porcentaje de materiales reciclables"
                )

            with col_metrica3:
                st.metric(
                    label="Peso Total Estimado",
                    value=f"{total_peso:.1f} kg",
                    help="Peso total estimado de residuos"
                )

            with col_metrica4:
                st.metric(
                    label="CO₂ Ahorrado",
                    value=f"{impacto_co2:.1f} kg",
                    help="Dióxido de carbono ahorrado por reciclaje"
                )

            # Visualizaciones mejoradas
            st.markdown("---")
            st.markdown("### Análisis Visual")

            # Primera fila de gráficos
            col_chart1, col_chart2 = st.columns([1, 1])

            with col_chart1:
                st.markdown("#### Distribución por Tipo")
                counts = df_filtrado["class"].value_counts().reset_index()
                counts.columns = ["Tipo", "Cantidad"]

                chart_composition = (
                    alt.Chart(counts)
                    .mark_arc(innerRadius=50, outerRadius=120)
                    .encode(
                        theta=alt.Theta("Cantidad:Q"),
                        color=alt.Color("Tipo:N",
                                      scale=alt.Scale(scheme="set2"),
                                      legend=alt.Legend(title="Tipo de Residuo")),
                        tooltip=[alt.Tooltip("Tipo:N", title="Tipo"),
                               alt.Tooltip("Cantidad:Q", title="Cantidad")]
                    )
                    .properties(height=300, title="")
                )
                st.altair_chart(chart_composition, use_container_width=True)

            with col_chart2:
                st.markdown("#### Valor Reciclaje")
                df_filtrado['categoria_valor'] = df_filtrado['class'].apply(obtener_categoria_valor_reciclaje)
                value_counts = df_filtrado['categoria_valor'].value_counts().reset_index()
                value_counts.columns = ['Categoría', 'Cantidad']

                chart_value = (
                    alt.Chart(value_counts)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, opacity=0.85)
                    .encode(
                        x=alt.X("Categoría:N", title="Categoría de Valor"),
                        y=alt.Y("Cantidad:Q", title="Cantidad"),
                        color=alt.Color("Categoría:N",
                                      scale=alt.Scale(domain=['Alto Valor Reciclable', 'Bajo Valor Reciclable', 'Residuales/Orgánico'],
                                                    range=['#22c55e', '#eab308', '#ef4444']),
                                      legend=None),
                        tooltip=[alt.Tooltip("Categoría:N", title="Categoría"),
                               alt.Tooltip("Cantidad:Q", title="Cantidad")]
                    )
                    .properties(height=300, title="")
                )
                st.altair_chart(chart_value, use_container_width=True)

            # Segunda fila de gráficos
            col_trend, col_time = st.columns([1, 1])

            with col_trend:
                st.markdown("#### Tendencia Temporal")
                df_tendencia = df_filtrado.groupby('date')['id'].count().reset_index()
                df_tendencia.columns = ['Fecha', 'Cantidad']

                chart_trend = (
                    alt.Chart(df_tendencia)
                    .mark_line(point=True, strokeWidth=3)
                    .encode(
                        x=alt.X("Fecha:T", title="Fecha"),
                        y=alt.Y("Cantidad:Q", title="Cantidad de Residuos"),
                        tooltip=[alt.Tooltip("Fecha:T", title="Fecha"),
                               alt.Tooltip("Cantidad:Q", title="Cantidad")]
                    )
                    .properties(height=250, title="")
                )
                st.altair_chart(chart_trend, use_container_width=True)

            with col_time:
                st.markdown("#### Distribución por Hora")
                df_filtrado['hora'] = df_filtrado['timestamp'].dt.hour
                hourly_counts = df_filtrado.groupby('hora')['id'].count().reset_index()
                hourly_counts.columns = ['Hora', 'Cantidad']

                chart_hourly = (
                    alt.Chart(hourly_counts)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X("Hora:O", title="Hora del Día"),
                        y=alt.Y("Cantidad:Q", title="Registros"),
                        color=alt.Color("Cantidad:Q", scale=alt.Scale(scheme="blues")),
                        tooltip=[alt.Tooltip("Hora:O", title="Hora"),
                               alt.Tooltip("Cantidad:Q", title="Cantidad")]
                    )
                    .properties(height=250, title="")
                )
                st.altair_chart(chart_hourly, use_container_width=True)

            # Análisis por sector si hay múltiples sectores
            if len(df_filtrado['sector'].unique()) > 1:
                st.markdown("---")
                st.markdown("### Análisis por Sector")

                sector_comparison = df_filtrado.groupby('sector')['id'].count().reset_index()
                sector_comparison.columns = ['Sector', 'Total_Residuos']

                chart_sector = (
                    alt.Chart(sector_comparison)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("Sector:N", title="Sector", sort="-y"),
                        y=alt.Y("Total_Residuos:Q", title="Total de Residuos"),
                        color=alt.Color("Sector:N", scale=alt.Scale(scheme="category10")),
                        tooltip=[alt.Tooltip("Sector:N", title="Sector"),
                               alt.Tooltip("Total_Residuos:Q", title="Total")]
                    )
                    .properties(height=300, title="Comparación por Sector")
                )
                st.altair_chart(chart_sector, use_container_width=True)

            # Tabla de datos detallados
            st.markdown("---")
            st.markdown("### Datos Detallados")

            # Preparar datos para mostrar
            df_mostrar = df_filtrado.copy()
            df_mostrar['timestamp'] = df_mostrar['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            df_mostrar = df_mostrar[['timestamp', 'sector', 'class', 'confidence', 'peso_total_foto_kg', 'coordenadas']]
            df_mostrar.columns = ['Fecha/Hora', 'Sector', 'Tipo', 'Confianza', 'Peso (kg)', 'Coordenadas']

            st.dataframe(
                df_mostrar.tail(50),
                use_container_width=True,
                column_config={
                    "Confianza": st.column_config.NumberColumn(format="%.1f%%"),
                    "Peso (kg)": st.column_config.NumberColumn(format="%.2f kg")
                }
            )

            # Mapa interactivo
            st.markdown("---")
            st.markdown("### Mapa Interactivo")
            
            # Filtros específicos para el mapa
            col_map_filt1, col_map_filt2 = st.columns(2)
            
            with col_map_filt1:
                vista_mapa = st.selectbox(
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