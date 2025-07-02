
import pandas as pd
from src import config
from src.data_processing import load_data, preprocess_data
from src.model_training import (
    get_features_and_target,
    split_data,
    train_model,
    evaluate_model,
    save_model
)

def main():
    """
    Orquesta el pipeline completo de entrenamiento del modelo.
    """
    # 1. Cargar y procesar datos
    df_raw = load_data(config.DATASET_FILE)
    df_processed = preprocess_data(df_raw)

    # Filtrar por departamentos de interés si es necesario
    # En este caso, entrenaremos con todos para tener un modelo más robusto,
    # pero el filtrado se puede aplicar aquí si se desea.
    # df_filtered = df_processed[df_processed['departamento'].isin(config.TARGET_DEPARTMENTS)]
    
    # 2. Definir características y objetivo
    X, y = get_features_and_target(df_processed)
    
    # 3. Dividir datos
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 4. Entrenar modelo
    model = train_model(X_train, y_train)
    
    # 5. Evaluar modelo
    evaluate_model(model, X_test, y_test)
    
    # 6. Guardar modelo
    save_model(model)

if __name__ == "__main__":
    main()
