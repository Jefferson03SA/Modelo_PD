
from pathlib import Path

# --- Directorios ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# --- Archivos ---
DATASET_FILE = DATA_DIR / "datos_abiertos_vigilancia_dengue_2000_2023.csv"
MODEL_FILE = MODELS_DIR / "dengue_model.joblib"
ENCODER_FILE = MODELS_DIR / "label_encoders.joblib"

# --- Configuración del Modelo ---
# Departamentos de interés según el documento
TARGET_DEPARTMENTS = ["PIURA", "ICA", "LA LIBERTAD", "LIMA", "LORETO"]

# Columnas a utilizar del dataset original
RELEVANT_COLUMNS = [
    "departamento",
    "provincia",
    "distrito",
    "ano",
    "semana",
    "tipo_dx",
    "sexo",
    "edad",
    "tipo_edad"
]

# Columnas categóricas que necesitan codificación
CATEGORICAL_FEATURES = ["departamento", "provincia", "distrito", "sexo", "tipo_edad"]

# Lista final de características para el modelo (en orden)
MODEL_FEATURES = [
    'edad',
    'sexo',
    'tipo_edad',
    'casos_lag1',
    'casos_diff',
    'departamento',
    'provincia',
    'distrito',
    'ano',
    'semana'
]


# Variable objetivo que crearemos
TARGET_VARIABLE = "brote"

# Parámetros del modelo Random Forest
MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1  # Usar todos los procesadores disponibles
}

# Configuración para la división de datos
TEST_SIZE = 0.2
SHUFFLE_DATA = False # Importante para mantener la secuencia temporal
