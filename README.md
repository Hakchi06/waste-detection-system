# Sistema de Gestión de Residuos - Panamá

Aplicación web avanzada desarrollada con **Python**, **Streamlit**, **YOLO** y **Gemini AI** para la **detección automática, cuantificación y análisis de residuos sólidos** en Panamá, con registro histórico, mapas interactivos y estadísticas ambientales.

---

## 🌟 Descripción del Proyecto

Este proyecto tiene como propósito apoyar las iniciativas de gestión de residuos en Panamá, brindando a ciudadanos, municipios y organizaciones una herramienta integral para:

- 📸 **Detección automática** de residuos mediante IA (YOLO + Gemini)
- 🗺️ **Geolocalización** y mapas interactivos de puntos críticos
- 📊 **Análisis ambiental** con cálculo de impacto CO₂
- 🎯 **Reportes ciudadanos** para puntos de acumulación
- 📈 **Dashboard analítico** con métricas en tiempo real
- 📚 **Centro educativo** sobre reciclaje y gestión de residuos
- ⚙️ **Configuración avanzada** y exportación de datos

Está diseñado específicamente para la realidad panameña, considerando los desafíos únicos de gestión de residuos en el país.

---

## ✨ Características Principales

### 🔍 Detección y Análisis
- Clasificación automática mediante modelo YOLOv8 entrenado
- Estimación de peso usando Gemini AI
- Análisis de impacto ambiental (CO₂ ahorrado)
- Sistema de confianza configurable

### 📍 Geolocalización
- Captura automática de coordenadas GPS
- Mapas interactivos con Folium
- Visualización de puntos críticos por sector
- Integración con servicios de geolocalización

### 📊 Dashboard Analítico
- Gráficos interactivos con Altair
- Filtros avanzados por fecha, sector y tipo
- Alertas inteligentes de riesgo sanitario
- Tendencias temporales y análisis por hora
- Comparaciones entre sectores

### 📚 Centro Educativo
- Guía completa de reciclaje en Panamá
- Directorio de centros de reciclaje
- Horarios de recolección por sector
- Calculadora de impacto ambiental personal

### ⚙️ Gestión Avanzada
- Configuración de parámetros de detección
- Exportación de datos (CSV, JSON)
- Sistema de notificaciones configurables
- Reportes ejecutivos automáticos

---

## 🏗️ Arquitectura del Proyecto

El proyecto sigue una arquitectura modular organizada en capas:

### 📁 Estructura Modular
- **`src/`**: Código fuente principal
  - **`main.py`**: Punto de entrada de la aplicación Streamlit
  - **`config/`**: Configuraciones y constantes del sistema
  - **`data/`**: Gestión de datos, registros y cálculos ambientales
  - **`detection/`**: Lógica de detección IA (YOLO + Gemini)
  - **`ui/`**: Interfaces de usuario y dashboards

### 🎯 Principios de Diseño
- **Separación de responsabilidades**: Cada módulo tiene una función específica
- **Clases orientadas a objetos**: Uso de clases para encapsular lógica relacionada
- **Configuración centralizada**: Todas las constantes en `settings.py`
- **Gestión de dependencias**: Imports claros y organizados

### 🔄 Flujo de Datos
1. **UI** (`main.py`) → Recibe entrada del usuario
2. **Detection** (`detector.py`) → Procesa imagen con IA
3. **Data** (`manager.py`) → Almacena y calcula métricas
4. **UI** (`dashboard.py`) → Muestra resultados y análisis

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.10+**
- **Streamlit** - Framework web
- **Ultralytics YOLOv8** - Detección de objetos
- **Google Gemini AI** - Estimación de peso
- **Pandas** - Manipulación de datos
- **Altair** - Visualizaciones interactivas
- **Folium** - Mapas interactivos
- **Requests** - APIs externas
- **Pillow** - Procesamiento de imágenes

---

## 📁 Estructura del Proyecto

```
waste-detection-system/
│
├── app.py                      # Aplicación principal con navegación
├── README.md                   # Documentación del proyecto
├── requirements.txt            # Dependencias Python
├── .env                       # Variables de entorno (API keys)
│
├── data/
│   ├── categories.json        # Definición de categorías de residuos
│   ├── records.csv           # Base de datos de registros
│   └── records_scm.csv       # Backup de registros
│
├── models/
│   ├── best-classify.pt      # Modelo YOLO de clasificación
│   └── best.pt              # Modelo YOLO de detección
│
├── src/                      # Código fuente modular
│   ├── __init__.py
│   ├── main.py               # Punto de entrada principal
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py       # Configuración y constantes
│   ├── data/
│   │   ├── __init__.py
│   │   └── manager.py        # Gestión de datos y registros
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py       # Lógica de detección IA
│   └── ui/
│       ├── __init__.py
│       └── dashboard.py      # Interfaces de usuario
│
├── utils/                    # Código legacy (a migrar)
│   └── ...
│
└── scripts/
    └── export_report.py      # Scripts de exportación
```

---

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/Onig/waste-detection-system.git
cd waste-detection-system
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno
Crear archivo `.env` en la raíz del proyecto:
```env
GEMINI_API_KEY=tu_clave_api_aqui
```

### 4. Ejecutar la aplicación
```bash
streamlit run app.py
```

---

## 📖 Uso de la Aplicación

### 1. 📸 Registro de Residuos
- Selecciona el sector y coordenadas
- Sube una imagen o usa la cámara
- El sistema detectará automáticamente los residuos
- Revisa los resultados y métricas ambientales

### 2. 📊 Dashboard Analítico
- Filtra datos por sector, fecha y tipo de residuo
- Visualiza tendencias y distribuciones
- Revisa alertas de riesgo sanitario
- Genera reportes ejecutivos

### 3. 🗺️ Mapa Interactivo
- Visualiza la distribución geográfica de residuos
- Identifica puntos críticos de acumulación
- Filtra por tipo de residuo y sector

### 4. 📚 Centro Educativo
- Aprende sobre reciclaje en Panamá
- Encuentra centros de reciclaje cercanos
- Consulta horarios de recolección
- Calcula tu impacto ambiental personal

### 5. ⚙️ Configuración
- Ajusta parámetros de detección
- Exporta datos para análisis externos
- Configura notificaciones y alertas

---

## 🎯 Impacto Ambiental

El sistema calcula automáticamente el impacto ambiental de las actividades de reciclaje:

- **CO₂ Ahorrado**: Estimación de emisiones evitadas
- **Árboles Salvados**: Equivalente en conservación forestal
- **Agua Ahorrada**: Litros de agua preservados
- **Eficiencia de Reciclaje**: Porcentaje de materiales reciclables

---

## 🌍 Contexto Panameño

Este proyecto está diseñado considerando:
- La realidad urbana de Panamá
- Los desafíos de gestión de residuos sólidos
- La necesidad de participación ciudadana
- La integración con autoridades municipales
- La educación ambiental de la población

---

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Para contribuir:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/Hakchi06/waste-detection-system.git
cd waste-classifier
````

### 2. Crear entorno virtual (opcional)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run run.py
```

---

## Archivo de registro: **records.csv**

Cada clasificación se guarda con los siguientes campos:

| Campo      | Descripción                           |
| ---------- | ------------------------------------- |
| timestamp  | Fecha y hora                          |
| source     | Origen de la imagen (upload / webcam) |
| filename   | Nombre o referencia del archivo       |
| class      | Categoría clasificada                 |
| confidence | Nivel de confianza del modelo         |

---

## Archivo **categories.json**

Contiene la información de cada categoría disponible:

```json
{
  "names": ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"],
  "info": {
    "BIODEGRADABLE": {
      "description": "Residuos orgánicos que pueden descomponerse naturalmente.",
      "handling": "Recolectar por separado para compostaje.",
      "recyclable": false
    },
    "CARDBOARD": {
      "description": "Cajas y empaques rígidos de cartón.",
      "handling": "Aplanar y mantener seco.",
      "recyclable": true
    }
  }
}
```
