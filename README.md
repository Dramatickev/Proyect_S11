# Beta Bank — Predicción de abandono de clientes

Proyecto de machine learning orientado a identificar clientes con riesgo de abandonar Beta Bank. El análisis combina preparación reproducible de datos, comparación de modelos, validación cruzada y una aplicación interactiva en Streamlit.

## Resultado principal

El mejor desempeño se obtuvo con un **Random Forest ajustado mediante validación cruzada**:

| Métrica | Resultado en prueba |
|---|---:|
| F1 Score | **0.6219** |
| ROC AUC | **0.8610** |

El modelo final integra el preprocesamiento y la clasificación en un único `Pipeline`, por lo que puede recibir registros con la estructura original del dataset y utilizarse directamente desde la aplicación.

## Problema de negocio

Retener clientes suele ser menos costoso que adquirir nuevos. El objetivo es anticipar qué clientes presentan mayor riesgo de abandono para apoyar la priorización de acciones de retención.

La variable objetivo es `Exited`:

- `0`: el cliente permanece.
- `1`: el cliente abandona el banco.

El criterio mínimo del proyecto era alcanzar un F1 Score de `0.59`. El modelo final obtuvo `0.6219` sobre un conjunto de prueba que no se utilizó durante el ajuste.

## Datos

El dataset contiene **10,000 clientes** y presenta un desbalance relevante en la variable objetivo:

- Aproximadamente 79.63% permanece.
- Aproximadamente 20.37% abandona.

Características utilizadas:

- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`

Los identificadores `RowNumber`, `CustomerId` y `Surname` se excluyeron del entrenamiento. La variable `Geography` solamente contiene clientes de Francia, Alemania y España; por ello, la aplicación limita las predicciones a esos tres países.

## Metodología

### 1. División de datos

- 80% para entrenamiento.
- 20% para prueba.
- División estratificada para conservar la proporción de clases.

### 2. Preprocesamiento

El preprocesamiento se ajusta exclusivamente con los datos de entrenamiento para evitar fuga de información:

- Imputación de variables numéricas mediante la mediana.
- Imputación de variables categóricas mediante la moda.
- Estandarización de variables numéricas.
- One-Hot Encoding con manejo de categorías desconocidas.

### 3. Desbalance y selección del modelo

Se compararon los siguientes enfoques:

- Regresión logística base.
- Regresión logística con `class_weight='balanced'`.
- Regresión logística con SMOTE aplicado únicamente al entrenamiento.
- Ajuste del threshold mediante predicciones out-of-fold.
- Random Forest base.
- Random Forest con búsqueda aleatoria de hiperparámetros.

La selección de hiperparámetros se realizó mediante `RandomizedSearchCV` con cinco particiones. El conjunto de prueba permaneció intacto hasta la evaluación final.

## Comparación de modelos

| Modelo | F1 Score | ROC AUC |
|---|---:|---:|
| Random Forest ajustado | **0.6219** | **0.8610** |
| Random Forest base | 0.6103 | 0.8558 |
| Regresión logística con SMOTE | 0.5058 | 0.7754 |
| Logística balanceada + threshold | 0.5038 | 0.7773 |
| Regresión logística balanceada | 0.5000 | 0.7773 |
| Regresión logística base | 0.2873 | 0.7749 |

## Aplicación interactiva

El dashboard de Streamlit incluye dos secciones:

### Predicción individual

- Captura de características originales del cliente.
- Probabilidad estimada de abandono.
- Clasificación con threshold de `0.50`.
- Vista de los datos enviados al pipeline.

### Rendimiento del modelo

- Métricas del modelo final.
- Comparación interactiva de todos los modelos.
- Tabla de resultados generada desde el notebook.
- Resumen de metodología y limitaciones.

La aplicación carga `churn_pipeline.joblib`, que contiene el preprocesamiento y el Random Forest final, y `model_metrics.csv`, que conserva los resultados calculados por el notebook.

## Estructura del proyecto

```text
Proyect_S11/
├── app/
│   └── app.py
├── data/
│   └── churn.csv
├── models/
│   ├── churn_pipeline.joblib
│   └── model_metrics.csv
├── notebooks/
│   └── beta_bank_churn_analysis.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Ejecución local

Desde la raíz del proyecto, crea y activa un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instala las dependencias:

```powershell
python -m pip install -r requirements.txt
```

Inicia la aplicación:

```powershell
streamlit run app/app.py
```

Streamlit mostrará la dirección local, normalmente `http://localhost:8501`.

## Reproducción del análisis

1. Confirma que `data/churn.csv` esté disponible.
2. Abre `notebooks/beta_bank_churn_analysis.ipynb`.
3. Selecciona el kernel del entorno `.venv`.
4. Reinicia el kernel y ejecuta todas las celdas.

La última sección del notebook vuelve a generar:

- `models/churn_pipeline.joblib`
- `models/model_metrics.csv`

## Tecnologías

- Python
- Pandas y NumPy
- Scikit-learn
- Imbalanced-learn
- Plotly
- Streamlit
- Joblib
- Jupyter Notebook
- Git y GitHub

## Limitaciones

- El modelo solamente representa clientes de Francia, Alemania y España.
- Los datos no incluyen información temporal para evaluar cambios de comportamiento a lo largo del tiempo.
- La probabilidad estimada no sustituye el criterio comercial ni una estrategia de retención.
- Antes de utilizar el modelo en producción sería necesario validarlo con datos recientes, revisar su calibración y considerar el costo de falsos positivos y falsos negativos.

## Autor

Proyecto desarrollado por **Kevin Hernandez** como parte de su formación en Ingeniería en Ciencia de Datos.
