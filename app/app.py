from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from joblib import load


st.set_page_config(
    page_title='Beta Bank — Churn Predictor',
    page_icon='🏦',
    layout='wide'
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / 'models' / 'churn_pipeline.joblib'
METRICS_PATH = PROJECT_ROOT / 'models' / 'model_metrics.csv'


@st.cache_resource
def cargar_modelo(path):
    return load(path)


@st.cache_data
def cargar_metricas(path):
    return pd.read_csv(path)


if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error(
        'No se encontraron los archivos del modelo o las métricas. '
        'Ejecuta primero el notebook completo.'
    )
    st.stop()


modelo = cargar_modelo(MODEL_PATH)
metricas = cargar_metricas(METRICS_PATH)


st.title('Beta Bank — Predicción de abandono')

st.write(
    'Aplicación interactiva para estimar el riesgo de abandono '
    'de clientes y consultar el rendimiento de los modelos evaluados.'
)


with st.sidebar:
    st.header('Información del modelo')
    st.success('Pipeline cargado correctamente')
    st.write('**Modelo final:** Random Forest')
    st.write('**Threshold de clasificación:** 0.50')
    st.caption(f'Archivo: {MODEL_PATH.name}')


tab_prediccion, tab_rendimiento = st.tabs([
    'Predicción individual',
    'Rendimiento del modelo'
])


with tab_prediccion:
    st.subheader('Evaluar un cliente')

    st.caption(
        'El modelo fue entrenado exclusivamente con clientes '
        'de Francia, Alemania y España.'
    )

    with st.form('formulario_cliente'):
        columna_1, columna_2, columna_3 = st.columns(3)

        with columna_1:
            credit_score = st.number_input(
                'Puntuación crediticia',
                min_value=300,
                max_value=850,
                value=650
            )

            pais_visible = st.selectbox(
                'País',
                options=['Francia', 'Alemania', 'España']
            )

            genero_visible = st.selectbox(
                'Género',
                options=['Mujer', 'Hombre']
            )

            age = st.number_input(
                'Edad',
                min_value=18,
                max_value=100,
                value=40
            )

        with columna_2:
            tenure = st.number_input(
                'Antigüedad como cliente',
                min_value=0,
                max_value=10,
                value=5
            )

            balance = st.number_input(
                'Saldo bancario',
                min_value=0.0,
                value=75000.0,
                step=1000.0
            )

            num_products = st.number_input(
                'Número de productos',
                min_value=1,
                max_value=4,
                value=1
            )

        with columna_3:
            has_credit_card = st.checkbox(
                'Tiene tarjeta de crédito',
                value=True
            )

            is_active_member = st.checkbox(
                'Es miembro activo',
                value=True
            )

            estimated_salary = st.number_input(
                'Salario estimado',
                min_value=0.0,
                value=100000.0,
                step=1000.0
            )

        evaluar = st.form_submit_button(
            'Calcular riesgo de abandono',
            width='stretch'
        )

    if evaluar:
        mapa_paises = {
            'Francia': 'France',
            'Alemania': 'Germany',
            'España': 'Spain'
        }

        mapa_genero = {
            'Mujer': 'Female',
            'Hombre': 'Male'
        }

        cliente = pd.DataFrame([{
            'CreditScore': credit_score,
            'Geography': mapa_paises[pais_visible],
            'Gender': mapa_genero[genero_visible],
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': int(has_credit_card),
            'IsActiveMember': int(is_active_member),
            'EstimatedSalary': estimated_salary
        }])

        probabilidad = modelo.predict_proba(cliente)[0, 1]
        prediccion = int(probabilidad >= 0.50)

        st.divider()
        st.subheader('Resultado de la evaluación')

        columna_metrica, columna_resultado = st.columns([1, 2])

        with columna_metrica:
            st.metric(
                'Probabilidad estimada de abandono',
                f'{probabilidad:.1%}'
            )

            st.progress(
                float(probabilidad),
                text='Nivel estimado por el modelo'
            )

        with columna_resultado:
            if prediccion == 1:
                st.error(
                    'El modelo clasifica a este cliente con riesgo '
                    'de abandono utilizando un threshold de 0.50.'
                )
            else:
                st.success(
                    'El modelo clasifica a este cliente como probable '
                    'permanencia utilizando un threshold de 0.50.'
                )

            st.caption(
                'Esta estimación apoya la priorización de clientes, '
                'pero no sustituye una decisión comercial.'
            )

        with st.expander('Ver información enviada al modelo'):
            st.dataframe(
                cliente,
                width='stretch',
                hide_index=True
            )


with tab_rendimiento:
    st.subheader('Rendimiento del modelo final')

    modelo_final = metricas.loc[
        metricas['Modelo'] == 'Random Forest ajustado'
    ].iloc[0]

    metrica_1, metrica_2, metrica_3 = st.columns(3)

    metrica_1.metric(
        'F1 Score',
        f"{modelo_final['F1 Score']:.4f}"
    )

    metrica_2.metric(
        'ROC AUC',
        f"{modelo_final['ROC AUC']:.4f}"
    )

    metrica_3.metric(
        'Modelos comparados',
        len(metricas)
    )

    resultados_largos = metricas.melt(
        id_vars='Modelo',
        value_vars=['F1 Score', 'ROC AUC'],
        var_name='Métrica',
        value_name='Puntuación'
    )

    figura = px.bar(
        resultados_largos,
        y='Modelo',
        x='Puntuación',
        color='Métrica',
        orientation='h',
        barmode='group',
        text='Puntuación',
        category_orders={
            'Modelo': metricas['Modelo'].tolist()
        },
        color_discrete_map={
            'F1 Score': '#2563EB',
            'ROC AUC': '#D97706'
        },
        title='Comparación de desempeño por modelo'
    )

    figura.update_traces(
        texttemplate='%{text:.3f}',
        textposition='outside',
        cliponaxis=False
    )

    figura.update_layout(
        template='plotly_white',
        height=500,
        xaxis_title='Puntuación',
        yaxis_title='',
        legend_title='',
        margin=dict(l=220, r=50, t=80, b=50)
    )

    figura.update_xaxes(
        range=[0, 1],
        gridcolor='#E5E7EB'
    )

    figura.update_yaxes(
        autorange='reversed'
    )

    st.plotly_chart(
        figura,
        width='stretch'
    )

    st.subheader('Resultados completos')

    st.dataframe(
        metricas.style.format({
            'F1 Score': '{:.4f}',
            'ROC AUC': '{:.4f}'
        }),
        width='stretch',
        hide_index=True
    )

    st.info(
        'El modelo final se seleccionó mediante validación cruzada. '
        'La evaluación reportada se realizó sobre un conjunto de '
        'prueba que no fue utilizado durante el ajuste.'
    )

    with st.expander('Metodología y limitaciones'):
        st.markdown(
            """
            - División estratificada: 80% entrenamiento y 20% prueba.
            - Imputación, codificación y escalado dentro del pipeline.
            - Ajuste mediante validación cruzada de cinco particiones.
            - El modelo solamente representa Francia, Alemania y España.
            - Los resultados apoyan decisiones de retención, pero no
              sustituyen el criterio comercial ni una validación con
              información más reciente.
            """
        )