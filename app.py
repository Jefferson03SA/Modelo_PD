
import streamlit as st
import pandas as pd
import plotly.express as px
from joblib import load
from src import config
from src.data_processing import load_data, preprocess_data

def load_artifacts():
    """Carga el modelo y los encoders guardados."""
    try:
        model = load(config.MODEL_FILE)
        encoders = load(config.ENCODER_FILE)
        print("Modelo y encoders cargados exitosamente.")
        return model, encoders
    except FileNotFoundError:
        st.error(
            "Error: No se encontraron los archivos del modelo. "
            "Por favor, ejecute el script de entrenamiento 'train.py' primero."
        )
        return None, None

def main():
    """Función principal de la aplicación Streamlit."""
    st.set_page_config(page_title="Predictor de Brotes de Dengue", layout="wide")
    
    st.title("Dashboard de Predicción de Brotes de Dengue")
    st.write(
        "Esta herramienta utiliza un modelo de Machine Learning para predecir la "
        "probabilidad de brotes de dengue en diferentes regiones del Perú."
    )

    model, encoders = load_artifacts()
    if model is None:
        return

    # Cargar y procesar datos (usamos una versión en caché para velocidad)
    @st.cache_data
    def get_processed_data():
        df_raw = load_data(config.DATASET_FILE)
        # OJO: No aplicamos el encoding aquí, solo limpieza y feature engineering
        # El encoding se hace sobre la marcha para la predicción
        df_clean = preprocess_data(df_raw)
        return df_clean

    df = get_processed_data()

    # --- Panel Lateral de Filtros ---
    st.sidebar.header("Filtros de Visualización")
    
    # RF-04: Filtrado de Datos por Región
    # Usamos los nombres originales antes del encoding para el selector
    dept_encoder = encoders['departamento']
    department_options = list(dept_encoder.classes_)
    
    selected_department = st.sidebar.selectbox(
        "Seleccione un Departamento",
        options=department_options,
        index=department_options.index("LIMA") # Valor por defecto
    )

    # Filtrar datos para el departamento seleccionado
    df_filtered = df[df['departamento'] == dept_encoder.transform([selected_department])[0]]

    st.header(f"Análisis para el Departamento de: {selected_department}")

    if df_filtered.empty:
        st.warning("No hay datos disponibles para el departamento seleccionado.")
        return

    # --- Visualización de Resultados (RF-08) ---
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Evolución de Casos de Dengue Confirmados")
        # Agrupar por semana para visualización
        df_plot = df_filtered.groupby(['ano', 'semana'])['casos'].sum().reset_index()
        df_plot['fecha'] = pd.to_datetime(df_plot['ano'].astype(str) + df_plot['semana'].astype(str) + '1', format='%Y%W%w')
        
        fig = px.line(
            df_plot, x='fecha', y='casos',
            title="Casos Confirmados por Semana",
            labels={'fecha': 'Fecha', 'casos': 'Número de Casos Confirmados'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Predicción para la última semana ---
    with col2:
        st.subheader("Predicción de Brote")
        
        # Obtener los datos de la última semana disponible
        last_week_data = df_filtered.sort_values(by=['ano', 'semana']).iloc[-1:]
        
        if not last_week_data.empty:
            # Usar la lista de features del config para asegurar el orden
            features = config.MODEL_FEATURES
            
            # Asegurarse de que todas las features existan y reordenar
            X_predict = last_week_data[features]

            # Realizar predicción
            prediction = model.predict(X_predict)[0]
            proba = model.predict_proba(X_predict)[0]

            if prediction == 1:
                st.error(f"ALTO RIESGO DE BROTE")
                st.metric(label="Confianza de la Predicción", value=f"{proba[1]:.2%}")
            else:
                st.success(f"RIESGO NORMAL")
                st.metric(label="Confianza de la Predicción", value=f"{proba[0]:.2%}")
            
            st.write("Predicción para la última semana registrada:")
            st.write(f"Año: {last_week_data['ano'].values[0]}, Semana: {last_week_data['semana'].values[0]}")

if __name__ == "__main__":
    main()
