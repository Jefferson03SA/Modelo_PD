

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump
from src import config

def get_features_and_target(df):
    """Define y separa las características (X) y la variable objetivo (y)."""
    # Usar la lista de features centralizada desde el config
    features = config.MODEL_FEATURES
    
    # Asegurarse de que todas las features existan en el df
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features]
    y = df[config.TARGET_VARIABLE]
    
    print("Características seleccionadas para el modelo (desde config):")
    print(available_features)
    
    return X, y

def split_data(X, y):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    print(f"Dividiendo los datos: {1-config.TEST_SIZE:.0%} entrenamiento, {config.TEST_SIZE:.0%} prueba.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        shuffle=config.SHUFFLE_DATA,
        random_state=config.MODEL_PARAMS['random_state'] if config.SHUFFLE_DATA else None
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Entrena el modelo Random Forest."""
    print("Entrenando el modelo Random Forest...")
    model = RandomForestClassifier(
        n_estimators=config.MODEL_PARAMS['n_estimators'],
        random_state=config.MODEL_PARAMS['random_state'],
        n_jobs=config.MODEL_PARAMS['n_jobs']
    )
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")
    return model

def evaluate_model(model, X_test, y_test):
    """Evalúa el rendimiento del modelo."""
    print("Evaluando el modelo...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"Precisión (Accuracy): {accuracy:.4f}")
    print("\nMatriz de Confusión:")
    print(conf_matrix)
    print("\nReporte de Clasificación:")
    print(class_report)
    
    # Verificar si se cumple el requisito RNF-03
    if accuracy >= 0.85:
        print("\n[✓] Requisito de Fiabilidad (RNF-03) cumplido: Precisión >= 85%.")
    else:
        print("\n[!] Requisito de Fiabilidad (RNF-03) NO cumplido: Precisión < 85%.")

def save_model(model):
    """Guarda el modelo entrenado en un archivo."""
    dump(model, config.MODEL_FILE)
    print(f"\nModelo guardado exitosamente en: {config.MODEL_FILE}")

