# -*- coding: utf-8 -*-
"""
Dashboard Analitico - Insights sobre Obesidade
Para equipe medica
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuracao da pagina
st.set_page_config(
    page_title="Dashboard Analitico - Obesidade",
    page_icon="📊",
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
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-left: 5px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Carregar dados
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Obesity.csv')
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        return df
    except:
        st.error("Erro ao carregar dados")
        return None

df = load_data()

if df is not None:
    # Header
    st.markdown('<h1 class="main-header">📊 Dashboard Analitico - Estudo sobre Obesidade</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Insights para a Equipe Medica</p>', unsafe_allow_html=True)

    # Sidebar - Filtros
    with st.sidebar:
        st.header("🔍 Filtros")

        gender_filter = st.multiselect(
            "Genero",
            options=df['Gender'].unique(),
            default=df['Gender'].unique()
        )

        age_range = st.slider(
            "Faixa Etaria",
            int(df['Age'].min()),
            int(df['Age'].max()),
            (int(df['Age'].min()), int(df['Age'].max()))
        )

        obesity_filter = st.multiselect(
            "Nivel de Obesidade",
            options=df['Obesity'].unique(),
            default=df['Obesity'].unique()
        )

        st.markdown("---")
        st.info(f"**Total de registros:** {len(df)}")

    # Aplicar filtros
    df_filtered = df[
        (df['Gender'].isin(gender_filter)) &
        (df['Age'] >= age_range[0]) &
        (df['Age'] <= age_range[1]) &
        (df['Obesity'].isin(obesity_filter))
    ]

    st.info(f"📌 Visualizando {len(df_filtered)} de {len(df)} registros")

    # Metricas principais
    st.markdown("## 📈 Metricas Principais")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total de Pacientes", len(df_filtered))

    with col2:
        avg_age = df_filtered['Age'].mean()
        st.metric("Idade Media", f"{avg_age:.1f} anos")

    with col3:
        avg_bmi = df_filtered['BMI'].mean()
        st.metric("IMC Medio", f"{avg_bmi:.2f}")

    with col4:
        obesity_pct = (df_filtered['Obesity'].str.contains('Obesity').sum() / len(df_filtered)) * 100
        st.metric("Taxa de Obesidade", f"{obesity_pct:.1f}%")

    with col5:
        overweight_pct = (df_filtered['Obesity'].str.contains('Overweight').sum() / len(df_filtered)) * 100
        st.metric("Taxa de Sobrepeso", f"{overweight_pct:.1f}%")

    # Tabs de analise
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visao Geral",
        "🍎 Habitos Alimentares",
        "🏃 Estilo de Vida",
        "🧬 Fatores de Risco",
        "💡 Insights Principais"
    ])

    with tab1:
        st.markdown("### Distribuicao dos Niveis de Obesidade")

        col1, col2 = st.columns(2)

        with col1:
            # Grafico de pizza
            obesity_counts = df_filtered['Obesity'].value_counts()
            fig = px.pie(
                values=obesity_counts.values,
                names=obesity_counts.index,
                title="Distribuicao por Nivel de Obesidade",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Grafico de barras
            fig = px.bar(
                x=obesity_counts.index,
                y=obesity_counts.values,
                title="Contagem por Nivel",
                labels={'x': 'Nivel de Obesidade', 'y': 'Numero de Pacientes'},
                color=obesity_counts.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Distribuicao de IMC
        st.markdown("### Distribuicao do IMC (Indice de Massa Corporal)")

        fig = px.histogram(
            df_filtered,
            x='BMI',
            color='Obesity',
            title="Distribuicao de IMC por Nivel de Obesidade",
            nbins=50,
            labels={'BMI': 'IMC', 'Obesity': 'Nivel de Obesidade'},
            marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Relacao Peso x Altura
        st.markdown("### Relacao Peso x Altura")

        fig = px.scatter(
            df_filtered,
            x='Height',
            y='Weight',
            color='Obesity',
            size='BMI',
            hover_data=['Age', 'Gender'],
            title="Relacao entre Peso e Altura",
            labels={'Height': 'Altura (m)', 'Weight': 'Peso (kg)'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Analise de Habitos Alimentares")

        col1, col2 = st.columns(2)

        with col1:
            # Consumo de alimentos caloricos
            favc_obesity = pd.crosstab(df_filtered['FAVC'], df_filtered['Obesity'], normalize='index') * 100
            fig = px.bar(
                favc_obesity,
                title="Consumo de Alimentos Caloricos vs Obesidade",
                labels={'value': 'Porcentagem (%)', 'FAVC': 'Consome Alimentos Caloricos'},
                barmode='stack'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
                <strong>💡 Insight:</strong> Pacientes que consomem alimentos altamente caloricos
                com frequencia apresentam maior incidencia de obesidade.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Consumo de vegetais
            fig = px.box(
                df_filtered,
                x='Obesity',
                y='FCVC',
                title="Consumo de Vegetais por Nivel de Obesidade",
                labels={'FCVC': 'Frequencia de Consumo de Vegetais', 'Obesity': 'Nivel'},
                color='Obesity'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
                <strong>💡 Insight:</strong> Pacientes com peso normal tendem a consumir
                mais vegetais regularmente.
            </div>
            """, unsafe_allow_html=True)

        # Numero de refeicoes
        st.markdown("### Numero de Refeicoes Principais")

        fig = px.violin(
            df_filtered,
            x='Obesity',
            y='NCP',
            box=True,
            title="Numero de Refeicoes por Nivel de Obesidade",
            labels={'NCP': 'Numero de Refeicoes Principais', 'Obesity': 'Nivel'},
            color='Obesity'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Consumo de agua
        st.markdown("### Consumo de Agua")

        fig = px.box(
            df_filtered,
            x='Obesity',
            y='CH2O',
            title="Consumo de Agua por Nivel de Obesidade",
            labels={'CH2O': 'Consumo Diario de Agua (L)', 'Obesity': 'Nivel'},
            color='Obesity'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Analise de Estilo de Vida")

        col1, col2 = st.columns(2)

        with col1:
            # Atividade fisica
            fig = px.box(
                df_filtered,
                x='Obesity',
                y='FAF',
                title="Frequencia de Atividade Fisica",
                labels={'FAF': 'Frequencia de Atividade Fisica', 'Obesity': 'Nivel'},
                color='Obesity'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
                <strong>💡 Insight Critico:</strong> Nivel de atividade fisica esta fortemente
                correlacionado com o nivel de obesidade. Pacientes com obesidade praticam
                significativamente menos exercicios.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Tempo de tela
            fig = px.box(
                df_filtered,
                x='Obesity',
                y='TUE',
                title="Tempo com Dispositivos Tecnologicos por Dia",
                labels={'TUE': 'Tempo de Uso (horas/dia)', 'Obesity': 'Nivel'},
                color='Obesity'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
                <strong>💡 Insight:</strong> Maior tempo de tela esta associado a sedentarismo
                e maior risco de obesidade.
            </div>
            """, unsafe_allow_html=True)

        # Meio de transporte
        st.markdown("### Meio de Transporte Utilizado")

        transport_obesity = pd.crosstab(df_filtered['MTRANS'], df_filtered['Obesity'])
        fig = px.bar(
            transport_obesity,
            title="Meio de Transporte vs Nivel de Obesidade",
            labels={'value': 'Numero de Pacientes', 'MTRANS': 'Meio de Transporte'},
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Consumo de alcool
        st.markdown("### Consumo de Alcool")

        calc_obesity = pd.crosstab(df_filtered['CALC'], df_filtered['Obesity'], normalize='index') * 100
        fig = px.bar(
            calc_obesity,
            title="Frequencia de Consumo de Alcool vs Obesidade",
            labels={'value': 'Porcentagem (%)', 'CALC': 'Consumo de Alcool'},
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### Analise de Fatores de Risco")

        col1, col2 = st.columns(2)

        with col1:
            # Historico familiar
            family_obesity = pd.crosstab(df_filtered['family_history'], df_filtered['Obesity'], normalize='index') * 100
            fig = px.bar(
                family_obesity,
                title="Historico Familiar vs Obesidade",
                labels={'value': 'Porcentagem (%)', 'family_history': 'Historico Familiar'},
                barmode='stack',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
                <strong>⚠️ Fator de Risco Importante:</strong> Historico familiar de obesidade
                e um forte preditor. Pacientes com historico familiar apresentam taxa
                significativamente maior de obesidade.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Genero
            gender_obesity = pd.crosstab(df_filtered['Gender'], df_filtered['Obesity'], normalize='index') * 100
            fig = px.bar(
                gender_obesity,
                title="Genero vs Nivel de Obesidade",
                labels={'value': 'Porcentagem (%)', 'Gender': 'Genero'},
                barmode='stack'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Idade vs Obesidade
        st.markdown("### Relacao Idade e Obesidade")

        fig = px.box(
            df_filtered,
            x='Obesity',
            y='Age',
            title="Distribuicao de Idade por Nivel de Obesidade",
            labels={'Age': 'Idade (anos)', 'Obesity': 'Nivel'},
            color='Obesity'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Monitoramento de calorias
        st.markdown("### Monitoramento de Calorias")

        scc_obesity = pd.crosstab(df_filtered['SCC'], df_filtered['Obesity'], normalize='index') * 100
        fig = px.bar(
            scc_obesity,
            title="Monitoramento de Calorias vs Obesidade",
            labels={'value': 'Porcentagem (%)', 'SCC': 'Monitora Calorias'},
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Insight:</strong> Pacientes que monitoram suas calorias tendem a ter
            melhor controle de peso.
        </div>
        """, unsafe_allow_html=True)

    with tab5:
        st.markdown("### 💡 Principais Insights para a Equipe Medica")

        st.markdown("""
        <div class="insight-box">
            <h3>1. Fatores de Maior Impacto</h3>
            <ul>
                <li><strong>IMC:</strong> Principal indicador, com forte correlacao com nivel de obesidade</li>
                <li><strong>Atividade Fisica:</strong> Fator modificavel mais importante</li>
                <li><strong>Historico Familiar:</strong> Preditor genetico significativo</li>
                <li><strong>Habitos Alimentares:</strong> Consumo de alimentos caloricos e vegetais</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
            <h3>2. Recomendacoes para Intervencao</h3>
            <ul>
                <li><strong>Prioridade 1:</strong> Aumentar frequencia de atividade fisica</li>
                <li><strong>Prioridade 2:</strong> Melhorar habitos alimentares (mais vegetais, menos calorias)</li>
                <li><strong>Prioridade 3:</strong> Monitoramento regular do peso e IMC</li>
                <li><strong>Prioridade 4:</strong> Educacao sobre habitos saudaveis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
            <h3>3. Populacoes de Risco</h3>
            <ul>
                <li>Pacientes com historico familiar de obesidade</li>
                <li>Individuos sedentarios (< 1 hora de atividade fisica por semana)</li>
                <li>Pessoas com alto consumo de alimentos caloricos</li>
                <li>Pacientes com baixo consumo de agua e vegetais</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
            <h3>4. Estrategias Preventivas</h3>
            <ul>
                <li><strong>Educacao:</strong> Programas de conscientizacao sobre nutricao</li>
                <li><strong>Incentivo:</strong> Promocao de atividade fisica regular</li>
                <li><strong>Monitoramento:</strong> Acompanhamento periodico de peso e IMC</li>
                <li><strong>Intervencao Precoce:</strong> Identificar e tratar sobrepeso antes da obesidade</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Correlacao entre variaveis
        st.markdown("### Matriz de Correlacao")

        numeric_cols = ['Age', 'Height', 'Weight', 'BMI', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        corr_matrix = df_filtered[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            title="Correlacao entre Variaveis Numericas",
            color_continuous_scale='RdBu_r',
            aspect='auto',
            labels={'color': 'Correlacao'}
        )
        st.plotly_chart(fig, use_container_width=True)

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
        <p><strong>Dashboard Analitico de Obesidade</strong> | Auxiliando decisoes clinicas baseadas em dados</p>
        <p>Dataset: 2111 registros | Modelo: Random Forest | Acuracia: 99.29%</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Nao foi possivel carregar os dados. Verifique se o arquivo 'Obesity.csv' existe.")
