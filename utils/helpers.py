import os
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd
import uuid

def asegurar_archivo_registros(ruta_archivo):
    # Asegura que el archivo CSV de registros exista con los encabezados correctos
    if not os.path.exists(ruta_archivo):
        encabezado = ['id', 'timestamp', 'source', 'file_name', 'sector', 'coordenadas', 'class', 'confidence', 'peso_total_foto_kg']
        df = pd.DataFrame(columns=encabezado)
        df.to_csv(ruta_archivo, index=False)

def agregar_registro(ruta_archivo, fuente, nombre_archivo, sector, coordenadas, nombre_clase, confianza, peso_total_foto_kg):
    # Añade un nuevo registro de detección al archivo CSV
    nuevo_registro = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'source': fuente,
        'file_name': nombre_archivo,
        'sector': sector,
        'coordenadas': coordenadas,
        'class': nombre_clase,
        'confidence': confianza,
        'peso_total_foto_kg': peso_total_foto_kg
    }
    df_nuevo = pd.DataFrame([nuevo_registro])
    df_nuevo.to_csv(ruta_archivo, mode='a', header=False, index=False)

def obtener_categoria_valor_reciclaje(nombre_clase):
    # Clasifica el desecho como Alto Valor, Bajo Valor o Residual
    alto_valor = ['PLASTIC', 'METAL', 'GLASS']
    bajo_valor = ['CARDBOARD', 'PAPER'] 
    
    if nombre_clase in alto_valor:
        return 'Alto Valor Reciclable'
    elif nombre_clase in bajo_valor:
        return 'Bajo Valor Reciclable'
    else:
        return 'Residuales/Orgánico'

def calcular_impacto_ambiental(df):
    # Calcula el impacto ambiental basado en los residuos reciclables
    if df.empty:
        return 0.0

    total_co2_ahorrado = 0.0

    # Calcular impacto por tipo de residuo
    for _, row in df.iterrows():
        if 'class' in row and 'peso_total_foto_kg' in row:
            peso = float(row.get('peso_total_foto_kg', 0))
            tipo = row['class']

            # Factores específicos por tipo de material
            factores = {
                'PLASTIC': 0.8,
                'METAL': 0.6,
                'PAPER': 0.3,
                'GLASS': 0.4,
                'CARDBOARD': 0.3,
                'BIODEGRADABLE': 0.1
            }

            factor = factores.get(tipo, 0.2)
            total_co2_ahorrado += peso * factor

    return total_co2_ahorrado

def obtener_centros_reciclaje_panama():
    # Retorna información de centros de reciclaje en Panamá
    centros = [
        {
            "nombre": "Centro de Reciclaje Ciudad de Panamá",
            "direccion": "Calle 50, Ciudad de Panamá",
            "horario": "Lunes a Viernes: 8:00 AM - 5:00 PM",
            "telefono": "+507 123-4567",
            "materiales": ["Plástico", "Papel", "Cartón", "Metal", "Vidrio"]
        },
        {
            "nombre": "EcoCentro Panamá Oeste",
            "direccion": "Arraiján, Panamá Oeste",
            "horario": "Lunes a Sábado: 7:00 AM - 4:00 PM",
            "telefono": "+507 234-5678",
            "materiales": ["Plástico", "Metal", "Vidrio", "Electrónicos"]
        },
        {
            "nombre": "Recicla Panamá - San Miguelito",
            "direccion": "San Miguelito, Calle Principal",
            "horario": "Lunes a Viernes: 9:00 AM - 6:00 PM",
            "telefono": "+507 345-6789",
            "materiales": ["Papel", "Cartón", "Plástico", "Orgánicos"]
        },
        {
            "nombre": "Centro Verde Panamá",
            "direccion": "Corregimiento de Ancón",
            "horario": "Martes a Domingo: 10:00 AM - 3:00 PM",
            "telefono": "+507 456-7890",
            "materiales": ["Orgánicos", "Compostaje", "Jardinería"]
        }
    ]
    return centros

def generar_resumen_reporte(df_filtrado, categorias):
    # Genera un resumen ejecutivo del conjunto de datos filtrado
    if df_filtrado.empty:
        return "El informe no puede generarse: no hay datos para el rango y sector seleccionado."

    total_elementos = df_filtrado.shape[0]
    total_fotos = df_filtrado['file_name'].nunique()

    conteos = df_filtrado["class"].value_counts()

    # Cálculos de Reciclaje
    reciclables = [nombre for nombre, info in categorias["info"].items() if info.get("recyclable", False) == True]
    total_reciclable = conteos[conteos.index.isin(reciclables)].sum()
    porcentaje_reciclable = (total_reciclable / total_elementos) * 100 if total_elementos > 0 else 0

    # Top 3 de Desechos
    top_3 = conteos.head(3).to_string()

    # Cálculo de Peso Estimado Total
    peso_total_kg = df_filtrado.drop_duplicates(subset=['file_name'])['peso_total_foto_kg'].sum()

    # Conteo de Puntos Críticos
    puntos_criticos = df_filtrado.groupby(['file_name', 'sector', 'coordenadas']).size().reset_index(name='Total_Desechos').sort_values('Total_Desechos', ascending=False).head(3)
    puntos_criticos_str = puntos_criticos.to_string(index=False)

    reporte = f"""
    ### INFORME DE GESTIÓN MUNICIPAL EJECUTIVO

    **Periodo de Análisis:** {df_filtrado['date'].min()} a {df_filtrado['date'].max()}
    **Sector(es) Analizado(s):** {', '.join(df_filtrado['sector'].unique())}

    ---

    #### 1. Métrica de Impacto General
    - Total de Desechos Contados: **{total_elementos} ítems**
    - Número de Puntos de Limpieza/Fotos Registradas: **{total_fotos}**
    - Peso Total Estimado: **{peso_total_kg:.2f} kg**

    #### 2. Prioridad de Gestión
    - Categorías Dominantes (Top 3):
        {top_3}

    - Puntos Críticos (Top 3 por mayor acumulación):
        {puntos_criticos_str}

    #### 3. Eficiencia de Reciclaje
    - Porcentaje de Material Reciclable: **{porcentaje_reciclable:.1f}%**
    """
    return reporte
