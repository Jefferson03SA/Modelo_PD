# Modelo Predictivo de Brotes de Dengue

Este proyecto implementa un modelo de Machine Learning para predecir la ocurrencia de brotes de dengue en Perú y visualiza los resultados a través de un dashboard interactivo construido con Streamlit.

## Características

- **Pipeline Modular:** El código está estructurado para ser legible y mantenible, separando el procesamiento de datos, el entrenamiento y la aplicación.
- **Modelo Random Forest:** Utiliza un clasificador de Bosque Aleatorio para predecir el riesgo de brote.
- **Dashboard Interactivo:** Una interfaz de usuario simple creada con Streamlit para visualizar la evolución de casos y las predicciones del modelo por departamento.
- **Entorno Reproducible:** Gestionado con `pyenv` y un archivo `requirements.txt` para facilitar la instalación.


## Configuración y Ejecución

Sigue estos pasos para poner en marcha el proyecto en un entorno Linux.

### Prerrequisitos

- **Python 3.10** (gestionado preferiblemente con `pyenv`).
- **Git**.

### 1. Clonar el Repositorio

```bash
git clone https://github.com/Jefferson03SA/Modelo_PD.git
cd Modelo_PD
```

### 2. Configurar el Entorno Virtual

Este proyecto usa `pyenv` para gestionar la versión de Python.

```bash
# Establecer la versión local de Python (pyenv la instalará si es necesario)
pyenv local 3.10.13

# Crear y activar el entorno virtual
python -m venv venv
source venv/bin/activate  # Para bash/zsh
# O `source venv/bin/activate.fish` para fish shell
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Colocar el Conjunto de Datos

**IMPORTANTE:** El conjunto de datos no está incluido en este repositorio debido a su tamaño.

1.  **Descarga el dataset:** 
    ```bash
    https://drive.google.com/file/d/1k9hflOIzhvW1GPITWpsTNOstc3tcSJKk/view?usp=drive_link
    ```

2.  **Crea la carpeta `data`:**
    ```bash
    mkdir -p data
    ```
3.  **Mueve tu archivo CSV** a la carpeta `data` y asegúrate de que se llame `datos_abiertos_vigilancia_dengue_2000_2023.csv`.

### 5. Entrenar el Modelo

Antes de poder ejecutar la aplicación, necesitas entrenar el modelo. Esto generará el archivo `dengue_model.joblib` en la carpeta `models/`.

```bash
python train.py
```
La salida debería mostrar el rendimiento del modelo y confirmar que se ha guardado.

### 6. Ejecutar la Aplicación Streamlit

Una vez que el modelo esté entrenado, lanza el dashboard interactivo.

```bash
streamlit run app.py
```

Abre tu navegador y ve a la URL local que te indique la terminal (normalmente `http://localhost:8501`).

---
## Estructura del Proyecto

```
.
├── app.py              # Aplicación principal de Streamlit
├── data/               # (Ignorado por Git) Contiene el dataset CSV
├── models/             # (Ignorado por Git) Almacena el modelo entrenado
├── src/                # Código fuente modular
│   ├── config.py
│   ├── data_processing.py
│   └── model_training.py
├── train.py            # Script para orquestar el entrenamiento
├── .gitignore
├── README.md
└── requirements.txt
```
