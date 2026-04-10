# 🏦 Predicción de Abandono de Clientes (Churn Prediction)

## 📌 Descripción del Proyecto

Este proyecto tiene como objetivo predecir si un cliente abandonará el banco (churn) utilizando técnicas de Machine Learning. A partir de datos históricos sobre el comportamiento de los clientes, se construyó un modelo capaz de identificar a aquellos con mayor probabilidad de abandono.

El enfoque principal fue maximizar el **F1 Score**, logrando un equilibrio entre precisión y recall, además de evaluar el desempeño mediante la métrica **ROC AUC**.

---

## 🎯 Objetivo

Desarrollar un modelo de clasificación que permita:

* Identificar clientes con alto riesgo de abandono
* Optimizar estrategias de retención
* Mejorar la toma de decisiones del negocio

---

## 📊 Dataset

El conjunto de datos incluye información relevante de los clientes, como:

* Datos demográficos (Edad, Género, País)
* Información financiera (Balance, Salario estimado, Credit Score)
* Relación con el banco (Antigüedad, Número de productos, Actividad)

---

## ⚙️ Metodología

### 1. Preparación de datos

* Limpieza de datos
* Codificación de variables categóricas
* Separación en conjuntos de entrenamiento y prueba

### 2. Análisis del desequilibrio de clases

Se identificó que la clase de clientes que abandonan el banco es minoritaria, lo cual impacta el desempeño del modelo.

### 3. Modelado inicial

Se entrenaron modelos base sin tratar el desequilibrio:

* Regresión Logística
* Random Forest

### 4. Mejora del modelo

Se aplicaron diferentes estrategias:

* Ajuste del **umbral de decisión (threshold tuning)**
* Comparación de múltiples modelos
* Evaluación con métricas enfocadas en clasificación desbalanceada

### 5. Selección del modelo final

El modelo de **Random Forest** presentó el mejor desempeño.

---

## 📈 Resultados

| Métrica   | Valor |
| --------- | ----- |
| F1 Score  | 0.61  |
| ROC AUC   | 0.85  |
| Precision | 0.59  |
| Recall    | 0.63  |

✔ Se superó el requisito mínimo de F1 Score (0.59)

---

## 🔍 Insights

* La **edad** es el factor más importante en la predicción de churn.
* Variables financieras como:

  * Balance
  * Salario estimado
  * Credit Score
    tienen un alto impacto en el abandono.
* Las variables demográficas tienen menor relevancia.

---

## 🧠 Conclusiones

El modelo desarrollado permite identificar de manera efectiva a los clientes con mayor probabilidad de abandonar el banco. Esto proporciona una herramienta valiosa para implementar estrategias de retención más eficientes y focalizadas.

El uso de métricas como F1 Score y ROC AUC permitió evaluar correctamente el desempeño en un contexto de clases desbalanceadas.

---

## 🚀 Posibles mejoras

* Optimización de hiperparámetros (Grid Search / Random Search)
* Validación cruzada
* Implementación en producción (API o aplicación web)
* Integración con dashboards (Power BI)

---

## 🛠️ Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## 👨‍💻 Autor

Proyecto desarrollado por Kevin Hernandez

---

## ⭐ Nota

Este proyecto forma parte de un ejercicio de Ciencia de Datos enfocado en problemas reales de negocio en el sector bancario.
