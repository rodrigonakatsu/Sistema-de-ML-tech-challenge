# -*- coding: utf-8 -*-
"""
Aplicacao Streamlit para Predicao de Obesidade
Sistema preditivo para auxiliar equipe medica
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Configuracao da pagina
st.set_page_config(
    page_title="Predicao de Obesidade",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Carregar artefatos do modelo
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('modelo_obesidade.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, label_encoders, target_encoder, feature_names
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None, None, None, None

model, scaler, label_encoders, target_encoder, feature_names = load_model_artifacts()

# Header
st.markdown('<h1 class="main-header">🏥 Sistema de Predicao de Obesidade</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema preditivo para auxiliar a equipe medica no diagnostico de obesidade</p>', unsafe_allow_html=True)

# Sidebar - Informacoes do modelo
with st.sidebar:
    st.header("ℹ️ Informacoes do Modelo")
    st.info("""
    **Modelo:** Random Forest Classifier

    **Acuracia:** 99.29%

    **Objetivo:** Prever o nivel de obesidade com base em caracteristicas fisicas e habitos de vida

    **Classes:**
    - Peso Insuficiente
    - Peso Normal
    - Sobrepeso Nivel I
    - Sobrepeso Nivel II
    - Obesidade Tipo I
    - Obesidade Tipo II
    - Obesidade Tipo III
    """)

    st.header("📊 Sobre os Dados")
    st.write("""
    Dataset com 2111 registros contendo informacoes sobre:
    - Dados demograficos
    - Medidas fisicas
    - Habitos alimentares
    - Estilo de vida
    """)

# Inicializar valores padrao no session_state se nao existirem
if 'gender' not in st.session_state:
    st.session_state.gender = 'Male'
if 'age' not in st.session_state:
    st.session_state.age = 25
if 'height' not in st.session_state:
    st.session_state.height = 1.70
if 'weight' not in st.session_state:
    st.session_state.weight = 70.0
if 'family_history' not in st.session_state:
    st.session_state.family_history = 'no'
if 'favc' not in st.session_state:
    st.session_state.favc = 'no'
if 'fcvc' not in st.session_state:
    st.session_state.fcvc = 2.0
if 'ncp' not in st.session_state:
    st.session_state.ncp = 3.0
if 'caec' not in st.session_state:
    st.session_state.caec = 'no'
if 'smoke' not in st.session_state:
    st.session_state.smoke = 'no'
if 'ch2o' not in st.session_state:
    st.session_state.ch2o = 2.0
if 'scc' not in st.session_state:
    st.session_state.scc = 'no'
if 'faf' not in st.session_state:
    st.session_state.faf = 1.0
if 'tue' not in st.session_state:
    st.session_state.tue = 3.0
if 'calc' not in st.session_state:
    st.session_state.calc = 'no'
if 'mtrans' not in st.session_state:
    st.session_state.mtrans = 'Public_Transportation'

# Tabs principais
tab1, tab2, tab3 = st.tabs(["🔮 Fazer Predicao", "📈 Interpretar Resultado", "🧪 Teste Rapido"])

with tab1:
    st.header("Insira os Dados do Paciente")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Dados Basicos")
        gender = st.selectbox("Genero", ["Female", "Male"],
                             index=0 if st.session_state.get('gender', 'Male') == "Female" else 1)
        age = st.slider("Idade", 14, 100, int(st.session_state.get('age', 25)))
        height = st.number_input("Altura (metros)", 0.50, 2.50, float(st.session_state.get('height', 1.70)), 0.01)
        weight = st.number_input("Peso (kg)", 20.0, 500.0, float(st.session_state.get('weight', 70.0)), 0.5)

    with col2:
        st.subheader("Historico e Habitos Alimentares")
        family_history = st.selectbox("Historico familiar de obesidade?", ["yes", "no"],
                                     index=0 if st.session_state.get('family_history', 'no') == "yes" else 1)
        favc = st.selectbox("Consome alimentos caloricos com frequencia?", ["yes", "no"],
                           index=0 if st.session_state.get('favc', 'no') == "yes" else 1)
        fcvc = st.slider("Frequencia de consumo de vegetais (0-3)", 0.0, 3.0, float(st.session_state.get('fcvc', 2.0)), 0.5)
        ncp = st.slider("Numero de refeicoes principais por dia", 1.0, 4.0, float(st.session_state.get('ncp', 3.0)), 0.5)
        caec_options = ["no", "Sometimes", "Frequently", "Always"]
        caec = st.selectbox("Come entre as refeicoes?", caec_options,
                           index=caec_options.index(st.session_state.get('caec', 'no')))

    with col3:
        st.subheader("Estilo de Vida")
        smoke = st.selectbox("Fuma?", ["no", "yes"],
                            index=0 if st.session_state.get('smoke', 'no') == "no" else 1)
        ch2o = st.slider("Consumo diario de agua (litros)", 0.0, 3.0, float(st.session_state.get('ch2o', 2.0)), 0.5)
        scc = st.selectbox("Monitora calorias?", ["no", "yes"],
                          index=0 if st.session_state.get('scc', 'no') == "no" else 1)
        faf = st.slider("Frequencia de atividade fisica (0-3)", 0.0, 3.0, float(st.session_state.get('faf', 1.0)), 0.5)
        tue = st.slider("Tempo com dispositivos tecnologicos (horas/dia)", 0.0, 10.0, float(st.session_state.get('tue', 3.0)), 0.5,
                       help="🟢 0-2h: Saudavel | 🟡 3-5h: Moderado | 🔴 5h+: Alto risco")
        calc_options = ["no", "Sometimes", "Frequently", "Always"]
        calc = st.selectbox("Frequencia de consumo de alcool", calc_options,
                           index=calc_options.index(st.session_state.get('calc', 'no')))
        mtrans_options = ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
        mtrans = st.selectbox("Meio de transporte", mtrans_options,
                             index=mtrans_options.index(st.session_state.get('mtrans', 'Public_Transportation')))

    # Botao de predicao
    if st.button("🔮 Realizar Predicao", type="primary", use_container_width=True):
        if model is not None:
            try:
                # Criar dataframe com os dados de entrada
                input_data = pd.DataFrame({
                    'Gender': [gender],
                    'Age': [age],
                    'Height': [height],
                    'Weight': [weight],
                    'family_history': [family_history],
                    'FAVC': [favc],
                    'FCVC': [fcvc],
                    'NCP': [ncp],
                    'CAEC': [caec],
                    'SMOKE': [smoke],
                    'CH2O': [ch2o],
                    'SCC': [scc],
                    'FAF': [faf],
                    'TUE': [tue],
                    'CALC': [calc],
                    'MTRANS': [mtrans]
                })

                # Feature engineering (mesma logica do treinamento)
                input_data['BMI'] = input_data['Weight'] / (input_data['Height'] ** 2)
                input_data['Weight_Age_Ratio'] = input_data['Weight'] / input_data['Age']
                input_data['Age_Category'] = pd.cut(input_data['Age'],
                                                    bins=[0, 25, 35, 50, 100],
                                                    labels=['Young', 'Adult', 'Middle_Age', 'Senior'])
                input_data['Healthy_Score'] = (
                    input_data['FCVC'] +
                    input_data['CH2O'] +
                    input_data['FAF'] -
                    (input_data['TUE'] / 2)
                )
                input_data['Risk_Score'] = (
                    input_data['FAVC'].map({'yes': 1, 'no': 0}) +
                    input_data['family_history'].map({'yes': 1, 'no': 0}) +
                    (input_data['NCP'] > 3).astype(int) +
                    input_data['SMOKE'].map({'yes': 1, 'no': 0})
                )

                # Encoding
                for col in ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'Age_Category']:
                    if col in label_encoders:
                        # Verificar se o valor existe no encoder
                        le = label_encoders[col]
                        if input_data[col].values[0] in le.classes_:
                            input_data[col] = le.transform(input_data[col])
                        else:
                            # Se nao existir, usar o primeiro valor
                            input_data[col] = 0

                # Normalizar
                input_scaled = scaler.transform(input_data[feature_names])

                # Converter de volta para DataFrame com nomes das features
                input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

                # Predicao
                prediction = model.predict(input_scaled_df)
                prediction_proba = model.predict_proba(input_scaled_df)

                # Decodificar predicao
                predicted_class = target_encoder.inverse_transform(prediction)[0]

                # Exibir resultado
                st.markdown("---")
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"## 🎯 Resultado da Predicao")

                    # Mapear classe para cor
                    color_map = {
                        'Insufficient_Weight': '#3498db',
                        'Normal_Weight': '#2ecc71',
                        'Overweight_Level_I': '#f39c12',
                        'Overweight_Level_II': '#e67e22',
                        'Obesity_Type_I': '#e74c3c',
                        'Obesity_Type_II': '#c0392b',
                        'Obesity_Type_III': '#8e44ad'
                    }

                    color = color_map.get(predicted_class, '#95a5a6')

                    st.markdown(f"""
                    <div style='background-color: {color}; padding: 2rem; border-radius: 10px; color: white; text-align: center;'>
                        <h1 style='margin: 0; font-size: 2.5rem;'>{predicted_class.replace('_', ' ')}</h1>
                    </div>
                    """, unsafe_allow_html=True)

                    # IMC calculado
                    bmi = weight / (height ** 2)
                    st.metric("IMC Calculado", f"{bmi:.2f}")

                with col2:
                    st.markdown("### Confianca do Modelo")
                    max_proba = prediction_proba[0].max()
                    st.metric("Confianca", f"{max_proba*100:.1f}%")

                    # Grafico de gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=max_proba*100,
                        title={'text': "Confianca"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': color},
                               'steps': [
                                   {'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 75], 'color': "gray"}],
                               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
                    fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # Grafico de probabilidades
                st.markdown("### 📊 Probabilidades por Classe")
                proba_df = pd.DataFrame({
                    'Classe': target_encoder.classes_,
                    'Probabilidade': prediction_proba[0] * 100
                }).sort_values('Probabilidade', ascending=True)

                fig = px.bar(proba_df, x='Probabilidade', y='Classe', orientation='h',
                            color='Probabilidade', color_continuous_scale='RdYlGn_r',
                            labels={'Probabilidade': 'Probabilidade (%)'})
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erro ao realizar predicao: {e}")
                st.exception(e)
        else:
            st.error("Modelo nao carregado. Verifique os arquivos.")

with tab2:
    st.header("📈 Como Interpretar o Resultado")

    st.markdown("""
    ### Classificacao dos Niveis de Obesidade

    O modelo classifica os pacientes em 7 categorias:

    1. **Peso Insuficiente (Insufficient Weight)**
       - IMC < 18.5
       - Recomendacao: Avaliar causas e considerar acompanhamento nutricional

    2. **Peso Normal (Normal Weight)**
       - IMC 18.5 - 24.9
       - Recomendacao: Manter habitos saudaveis

    3. **Sobrepeso Nivel I (Overweight Level I)**
       - IMC 25.0 - 27.4
       - Recomendacao: Iniciar mudancas no estilo de vida

    4. **Sobrepeso Nivel II (Overweight Level II)**
       - IMC 27.5 - 29.9
       - Recomendacao: Intervencao com dieta e exercicios

    5. **Obesidade Tipo I (Obesity Type I)**
       - IMC 30.0 - 34.9
       - Recomendacao: Acompanhamento medico regular

    6. **Obesidade Tipo II (Obesity Type II)**
       - IMC 35.0 - 39.9
       - Recomendacao: Intervencao medica intensiva

    7. **Obesidade Tipo III (Obesity Type III)**
       - IMC >= 40.0
       - Recomendacao: Intervencao medica urgente, considerar cirurgia bariatrica

    ### Fatores Mais Importantes

    O modelo considera principalmente:
    - **IMC (Indice de Massa Corporal)**: Fator mais importante
    - **Peso e Altura**: Medidas fisicas fundamentais
    - **Habitos Alimentares**: Consumo de calorias e vegetais
    - **Atividade Fisica**: Frequencia de exercicios
    - **Historico Familiar**: Predisposicao genetica
    """)

with tab3:
    st.header("🧪 Teste Rapido")
    st.info("Clique em um dos casos abaixo para carregar automaticamente os dados e volte para a aba 'Fazer Predicao'")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Caso 1: Peso Normal", use_container_width=True, type="primary"):
            st.session_state.gender = 'Female'
            st.session_state.age = 25
            st.session_state.height = 1.65
            st.session_state.weight = 60.0
            st.session_state.family_history = 'no'
            st.session_state.favc = 'no'
            st.session_state.fcvc = 3.0
            st.session_state.ncp = 3.0
            st.session_state.caec = 'Sometimes'
            st.session_state.smoke = 'no'
            st.session_state.ch2o = 2.5
            st.session_state.scc = 'yes'
            st.session_state.faf = 2.0
            st.session_state.tue = 1.5
            st.session_state.calc = 'no'
            st.session_state.mtrans = 'Walking'
            st.success("✅ Caso 1 carregado! Volte para a aba 'Fazer Predicao'")
            st.rerun()

    with col2:
        if st.button("Caso 2: Obesidade", use_container_width=True, type="primary"):
            st.session_state.gender = 'Male'
            st.session_state.age = 35
            st.session_state.height = 1.75
            st.session_state.weight = 110.0
            st.session_state.family_history = 'yes'
            st.session_state.favc = 'yes'
            st.session_state.fcvc = 1.0
            st.session_state.ncp = 4.0
            st.session_state.caec = 'Always'
            st.session_state.smoke = 'no'
            st.session_state.ch2o = 1.0
            st.session_state.scc = 'no'
            st.session_state.faf = 0.0
            st.session_state.tue = 6.0
            st.session_state.calc = 'Frequently'
            st.session_state.mtrans = 'Automobile'
            st.success("✅ Caso 2 carregado! Volte para a aba 'Fazer Predicao'")
            st.rerun()

    with col3:
        if st.button("Caso 3: Sobrepeso", use_container_width=True, type="primary"):
            st.session_state.gender = 'Male'
            st.session_state.age = 27
            st.session_state.height = 1.80
            st.session_state.weight = 87.0
            st.session_state.family_history = 'no'
            st.session_state.favc = 'no'
            st.session_state.fcvc = 2.0
            st.session_state.ncp = 3.0
            st.session_state.caec = 'Sometimes'
            st.session_state.smoke = 'no'
            st.session_state.ch2o = 2.0
            st.session_state.scc = 'no'
            st.session_state.faf = 2.0
            st.session_state.tue = 3.0
            st.session_state.calc = 'Frequently'
            st.session_state.mtrans = 'Public_Transportation'
            st.success("✅ Caso 3 carregado! Volte para a aba 'Fazer Predicao'")
            st.rerun()

    st.markdown("---")
    st.markdown("""
    ### 📋 Informações dos Casos de Teste:

    **Caso 1: Peso Normal**
    - Mulher, 25 anos, 1.65m, 60kg
    - Hábitos saudáveis: exercício regular, boa alimentação
    - Resultado esperado: Normal Weight com alta confiança

    **Caso 2: Obesidade**
    - Homem, 35 anos, 1.75m, 110kg
    - Hábitos ruins: sedentário, má alimentação, histórico familiar
    - Resultado esperado: Obesity Type I ou superior

    **Caso 3: Sobrepeso**
    - Homem, 27 anos, 1.80m, 87kg (IMC: 26.85)
    - Hábitos moderados: exercício regular, sem comida calórica frequente
    - Resultado esperado: Overweight Level I
    """)

# Footer fixo
st.markdown("""
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #666;
        text-align: center;
        padding: 10px 0;
        border-top: 2px solid #1f77b4;
        z-index: 999;
        font-size: 0.85rem;
    }
    .footer p {
        margin: 2px 0;
    }
    .main-content {
        margin-bottom: 80px;
    }
</style>
<div class="footer">
    <p><strong>Sistema de Predicao de Obesidade</strong> | Desenvolvido para auxiliar profissionais de saude</p>
    <p>Modelo: Random Forest | Acuracia: 99.29% | Dataset: 2111 registros</p>
</div>
""", unsafe_allow_html=True)
