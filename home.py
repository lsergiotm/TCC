import streamlit as st
import os

# Inicializar o estado da página atual
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"

# Estilização com CSS para design visual
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        color: #6C757D;
        margin-bottom: 2rem;
    }
    .button-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin: 0 auto;
        max-width: 900px;
        justify-content: center;
    }
    .button-card {
        padding: 20px;
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .button-card:hover {
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15);
        transform: translateY(-5px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Título principal e subtítulo
st.markdown('<h1 class="main-title">Bem-vindo ao Sistema de Análise de Internações Hospitalares</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Escolha uma análise para explorar os dados sobre internações hospitalares na RIDE de Brasília.</p>', unsafe_allow_html=True)

# Botões organizados
col1, col2 ,col3, col4 = st.columns(4)

# Definir os redirecionamentos
if col1.button("📊 Análise Descritiva e Estatística"):
    st.session_state["current_page"] = "descriptive_analysis"

if col2.button("💰 Análise de Custos"):
    st.session_state["current_page"] = "cost_analysis"

if col3.button("📅 Análise Temporal e Sazonal"):
    st.session_state["current_page"] = "temporal_analysis"

if col4.button("🗺️ Distribuição Geográfica"):
    st.session_state["current_page"] = "geographic_distribution"

col5,col6, col7, col8 = st.columns(4)

if col5.button("📅 Modelo de Série Temporal - MST"):
    st.session_state["current_page"] = "modelo_serie_temporal"

if col6.button("📈 Regressão Linear"):
    st.session_state["current_page"] = "regressao_linear"

if col7.button("📊 Método K-Means"):
    st.session_state["current_page"] = "k_means"

if col8.button("💡 Gradient Boosting"):
    st.session_state["current_page"] = "gradient_boosting"

col9, col10 ,col11,col12 = st.columns(4)
    
if col9.button("🌲 Random Forest"):
    st.session_state["current_page"] = "random_forest"

if col10.button("🔍 Visualização de Dados"):
    st.session_state["current_page"] = "visualizacao"

# Redirecionar para a página correspondente
if st.session_state["current_page"] == "descriptive_analysis":
    exec(open("descriptive_analysis.py", encoding="utf-8").read())

elif st.session_state["current_page"] == "cost_analysis":
    exec(open("cost_analysis.py", encoding="utf-8").read())

elif st.session_state["current_page"] == "temporal_analysis":
    exec(open("temporal_analysis.py", encoding="utf-8").read())

elif st.session_state["current_page"] == "geographic_distribution":
    exec(open("geographic_distribution.py", encoding="utf-8").read())

elif st.session_state["current_page"] == "modelo_serie_temporal":
    exec(open("modelo_serie_temporal.py", encoding="utf-8").read())

elif st.session_state["current_page"] == "regressao_linear":
    exec(open("regressao_linear.py", encoding="utf-8").read())

elif st.session_state["current_page"] == "k_means":
    exec(open("k_means.py", encoding="utf-8").read())

elif st.session_state["current_page"] == "random_forest":
    exec(open("random_forest.py", encoding="utf-8").read())

elif st.session_state["current_page"] == "gradient_boosting":
    exec(open("gradient_boosting.py", encoding="utf-8").read())

elif st.session_state["current_page"] == "visualizacao":
    exec(open("visualizacao.py", encoding="utf-8").read())