
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from src import config

def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    print(f"Cargando datos desde: {file_path}")
    df = pd.read_csv(
        file_path,
        usecols=config.RELEVANT_COLUMNS,
        encoding='latin1',  # Común en datasets de fuentes gubernamentales
        sep=',' # Especificar el separador correcto
    )
    return df

def clean_data(df):
    """Limpia y preprocesa los datos."""
    print("Limpiando datos...")
    # Convertir nombres de columnas a minúsculas y sin espacios
    df.columns = df.columns.str.lower().str.strip()

    # Manejo de valores nulos (ej. eliminar filas con nulos en columnas clave)
    df.dropna(subset=['departamento', 'semana', 'ano'], inplace=True)

    # Estandarizar valores de texto (mayúsculas para consistencia)
    for col in ['departamento', 'provincia', 'distrito']:
        df[col] = df[col].str.upper().str.strip()

    # Asegurar tipos de datos correctos
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df.dropna(subset=['edad'], inplace=True)
    df['edad'] = df['edad'].astype(int)

    return df

def create_target_variable(df):
    """Crea la variable objetivo 'brote'."""
    print("Creando variable objetivo 'brote'...")
    # 1. Crear la columna 'casos' (1 para confirmado, 0 para otros)
    df['casos'] = (df['tipo_dx'] == 'C').astype(int)

    # 2. Agregar casos por semana y departamento
    weekly_cases = df.groupby(['ano', 'semana', 'departamento'])['casos'].sum().reset_index()

    # 3. Definir umbral de brote usando una media móvil para capturar tendencias
    # Un "brote" ocurre si los casos de la semana superan la media de las últimas 4 semanas
    rolling_avg = weekly_cases.groupby('departamento')['casos'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )
    weekly_cases[config.TARGET_VARIABLE] = (weekly_cases['casos'] > rolling_avg * 1.2).astype(int) # 20% sobre la media móvil

    # Unir la variable objetivo al dataframe principal
    df = pd.merge(df, weekly_cases[['ano', 'semana', 'departamento', config.TARGET_VARIABLE]],
                  on=['ano', 'semana', 'departamento'], how='left')
    # Rellenar posibles nulos y asegurar que la columna es de tipo entero
    df[config.TARGET_VARIABLE] = df[config.TARGET_VARIABLE].fillna(0).astype(int)

    return df

def feature_engineering(df):
    """Genera nuevas características para el modelo."""
    print("Realizando ingeniería de características...")
    # Ordenar por tiempo para crear lags correctos
    df.sort_values(by=['ano', 'semana', 'departamento'], inplace=True)

    # Crear lag de casos (casos de la semana anterior)
    df['casos_lag1'] = df.groupby('departamento')['casos'].shift(1).fillna(0)

    # Crear diferencia de casos
    df['casos_diff'] = df['casos'].diff().fillna(0)

    return df

def encode_categorical_features(df):
    """Codifica las variables categóricas usando Label Encoding."""
    print("Codificando variables categóricas...")
    encoders = {}
    for feature in config.CATEGORICAL_FEATURES:
        if feature in df.columns:
            encoder = LabelEncoder()
            df[feature] = encoder.fit_transform(df[feature].astype(str))
            encoders[feature] = encoder
        else:
            print(f"Advertencia: La columna categórica '{feature}' no se encontró en el DataFrame.")

    # Guardar los encoders para usarlos en la predicción
    dump(encoders, config.ENCODER_FILE)
    print(f"Encoders guardados en: {config.ENCODER_FILE}")

    return df

def preprocess_data(df):
    """Ejecuta todo el pipeline de preprocesamiento."""
    df = clean_data(df)
    df = create_target_variable(df)
    df = feature_engineering(df)
    df = encode_categorical_features(df)
    return df
