# 🏦 Churn Prediction - Beta Bank

## 📌 Descripción del proyecto

Los clientes de Beta Bank están abandonando el banco de forma progresiva. Dado que retener clientes es más rentable que adquirir nuevos, el objetivo de este proyecto es desarrollar un modelo de **clasificación binaria** que permita predecir si un cliente abandonará el banco.

El modelo se entrena utilizando datos históricos de clientes, incluyendo información demográfica, financiera y de comportamiento.

---

## 🎯 Objetivo

Construir un modelo de machine learning que:

- Prediga si un cliente abandonará el banco (`Exited`)
- Alcance un **F1-score ≥ 0.59**
- Evalúe también la métrica **AUC-ROC** para medir la capacidad de discriminación del modelo

---

## 📊 Descripción de los datos

El dataset contiene información de 10,000 clientes con las siguientes variables:

- **CreditScore**: puntaje crediticio
- **Geography**: país de residencia
- **Gender**: género
- **Age**: edad
- **Tenure**: años como cliente
- **Balance**: saldo de la cuenta
- **NumOfProducts**: productos contratados
- **HasCrCard**: posee tarjeta de crédito
- **IsActiveMember**: actividad del cliente
- **EstimatedSalary**: salario estimado
- **Exited**: variable objetivo (1 = abandonó, 0 = permanece)

---

## ⚙️ Preprocesamiento de datos

Se realizaron los siguientes pasos:

- Eliminación de variables no informativas:
    - `RowNumber`, `CustomerId`, `Surname`
- Codificación de variables categóricas mediante **One-Hot Encoding**
- Manejo de valores nulos en `Tenure` usando la mediana
- Estandarización de variables numéricas para mejorar el rendimiento de los modelos

---

## 🔀 División de datos

El dataset fue dividido en:

- **Entrenamiento**: 60%
- **Validación**: 20%
- **Prueba**: 20%

Esto permitió entrenar modelos, ajustar hiperparámetros y evaluar el desempeño final en datos no vistos.

---

## ⚖️ Análisis de desbalance

Se detectó un desbalance en la variable objetivo:

- ~80% clientes que permanecen
- ~20% clientes que abandonan

Este desbalance puede afectar el rendimiento del modelo, especialmente en la detección de la clase minoritaria.

---

## 🤖 Modelos evaluados

Se entrenaron y compararon los siguientes modelos:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**

Inicialmente se entrenaron sin técnicas de balanceo para establecer una línea base.

---

## ⚖️ Técnicas de balanceo utilizadas

Para mejorar el rendimiento del modelo se aplicaron tres enfoques:

### 1. Ajuste de pesos de clase

Uso de `class_weight='balanced'` para penalizar errores en la clase minoritaria.

### 2. Upsampling

Aumento de la clase minoritaria mediante duplicación de muestras.

### 3. Downsampling

Reducción de la clase mayoritaria para equilibrar el dataset.

---

## 🏆 Selección del modelo final

El mejor desempeño se obtuvo con:

- **Modelo**: Random Forest
- **Técnica**: Downsampling
- **Número de estimadores**: 80

Este modelo logró el mejor equilibrio entre precisión y recall.

---

## 📈 Resultados

### 🔹 Conjunto de prueba (TEST)

- **F1-score**: 0.60
- **AUC-ROC**: 0.85

---

## 📊 Interpretación de métricas

- **F1-score (0.60)**: indica un buen equilibrio entre precisión y recall en la detección de clientes que abandonan.
- **AUC-ROC (0.85)**: el modelo tiene una alta capacidad para distinguir entre clientes que abandonan y los que permanecen.

---

## 🧠 Conclusiones

El modelo desarrollado permite identificar clientes con alta probabilidad de abandonar el banco, cumpliendo con el objetivo mínimo de desempeño.

El uso de técnicas de balanceo fue clave para mejorar el rendimiento, especialmente en la detección de la clase minoritaria.

Este modelo puede ser utilizado como herramienta de apoyo para diseñar estrategias de retención de clientes, permitiendo actuar de forma preventiva.

---

## 🚀 Posibles mejoras

- Optimización de hiperparámetros (GridSearch / RandomSearch)
- Ajuste del umbral de clasificación
- Uso de técnicas más avanzadas de balanceo (ej. SMOTE)
- Feature engineering

---

## 🛠️ Tecnologías utilizadas

- Python
- Pandas
- Scikit-learn
- Matplotlib

---

## 📌 Autor

Proyecto desarrollado por **Kevin Hernandez** como parte de su formación en ciencia de datos.
