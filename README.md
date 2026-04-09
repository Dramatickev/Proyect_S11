# 🏦 Predicción de abandono de clientes (Churn Prediction)

## 📌 Descripción del proyecto

El objetivo de este proyecto es predecir si un cliente abandonará el banco en un futuro cercano. Para ello, se desarrollaron y evaluaron distintos modelos de Machine Learning, buscando maximizar el F1-score debido al desbalance de clases presente en los datos.

---

## 📊 Dataset

El conjunto de datos contiene información de 10,000 clientes, incluyendo variables como:

* Edad
* Score de crédito
* Balance
* Número de productos
* Actividad del cliente
* Salario estimado
* País y género

La variable objetivo es:

* `Exited`: indica si el cliente abandonó el banco (1) o no (0)

---

## ⚙️ Proceso

### 1. Preparación de datos

* Eliminación de columnas no relevantes
* Manejo de valores nulos (imputación en `Tenure`)
* Codificación de variables categóricas (One-Hot Encoding)

---

### 2. Análisis exploratorio

* Identificación de desbalance de clases (~80/20)
* Análisis de variables relevantes

---

### 3. Modelado

Se probaron distintos enfoques:

#### 🔹 Logistic Regression (baseline)

* F1: ~0.28
* Bajo desempeño debido al desbalance

#### 🔹 Logistic Regression + class_weight

* F1: ~0.50
* Mejora significativa

#### 🔹 SMOTE

* No mejoró el modelo
* Disminuyó el rendimiento en test

#### 🔹 Random Forest

* Mejor desempeño general

---

### 4. Optimización

Se ajustaron hiperparámetros del modelo Random Forest:

* `n_estimators = 300`
* `max_depth = 10`
* `class_weight = 'balanced'`

---

## 🏆 Resultados finales

* **Modelo:** Random Forest optimizado
* **F1-score:** 0.625 ✅
* **AUC-ROC:** 0.864 🚀

El modelo cumple con el objetivo de alcanzar un F1-score mayor a 0.59.

---

## 📈 Conclusiones

* Logistic Regression no fue suficiente para capturar la complejidad del problema
* Técnicas como SMOTE no siempre mejoran el rendimiento
* Random Forest permitió obtener mejores resultados al capturar relaciones no lineales

---

## 🛠️ Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Jupyter Notebook

---

## 🚀 Próximos pasos

* Despliegue del modelo en una aplicación con Streamlit
* Visualización de insights en Power BI
* Mejora del modelo con técnicas más avanzadas

---

## 👤 Autor

Proyecto realizado por [Tu Nombre]
