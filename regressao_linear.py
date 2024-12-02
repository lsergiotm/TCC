import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import t
from statsmodels.stats.stattools import durbin_watson

from data_processing import load_data

# Carregar os dados processados
df = load_data()

# Garantir que a coluna 'ano_aih' seja numérica e válida
df['ano_aih'] = pd.to_numeric(df['ano_aih'], errors='coerce')
df = df[df['ano_aih'].notna()]  # Remove valores NaN
df['ano_aih'] = df['ano_aih'].astype(int)  # Converte para inteiro

# Garantir que as colunas de custo e quantidade sejam numéricas
df['Valor total dos procedimentos'] = pd.to_numeric(
    df['Valor total dos procedimentos'], errors='coerce'
).fillna(0)
df['Quantidade total de procedimentos'] = pd.to_numeric(
    df['Quantidade total de procedimentos'], errors='coerce'
).fillna(0)

# Título da página
st.title("Análise com Regressão Linear")
st.markdown("""
Explore a evolução dos custos e quantidades de procedimentos ao longo do tempo e utilize a regressão linear para prever relações entre os custos e quantidades.
""")

# ---------------------------------------------
# Filtros de Dados
# ---------------------------------------------
st.sidebar.header("Filtros de Dados")

# Dropdown para Estados com seleção múltipla e opção "Todos"
estados_disponiveis = ['Todos'] + sorted(df['uf_nome'].dropna().unique().tolist())
estado_selecionado = st.sidebar.selectbox(
    "Escolha o estado:",
    options=estados_disponiveis
)

# Filtrar Estados
if estado_selecionado != 'Todos':
    df = df[df['uf_nome'] == estado_selecionado]

# Dropdown para Municípios com seleção múltipla e opção "Todos"
municipios_disponiveis = ['Todos'] + sorted(df[df['uf_nome'] == estado_selecionado]['nome_municipio'].dropna().unique().tolist())
municipio_selecionado = st.sidebar.selectbox(
    "Escolha o município:",
    options=municipios_disponiveis
)

# Filtro por ano
anos_disponiveis = ['Todos'] + sorted(df['ano_aih'].dropna().unique().astype(str).tolist())
ano_selecionado = st.sidebar.selectbox("Selecione o Ano:", anos_disponiveis)

# Filtrar por ano
if ano_selecionado != 'Todos':
    df = df[df['ano_aih'] == ano_selecionado]

# ---------------------------------------------
# Regressão Linear - Preparação dos Dados
# ---------------------------------------------
st.subheader("Regressão Linear - Custos Médios vs Quantidades Médias")

if not df.empty:
    # Agrupar dados por mês
    regression_data = df.groupby('mes_aih').agg({
        'Valor total dos procedimentos': 'mean',
        'Quantidade total de procedimentos': 'mean'
    }).reset_index()

    # Renomear as colunas
    regression_data.rename(columns={
        'Valor total dos procedimentos': 'Custo Médio',
        'Quantidade total de procedimentos': 'Quantidade Média'
    }, inplace=True)

    # Regressão Linear
    X = regression_data[['Quantidade Média']]
    y = regression_data['Custo Médio']
    model = LinearRegression()
    model.fit(X, y)

    # Coeficientes
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, model.predict(X))
    residuals = y - model.predict(X)
    mse = mean_squared_error(y, model.predict(X))

    # Durbin-Watson para erros não correlacionados
    dw_stat = durbin_watson(residuals)

    # Intervalo de confiança para previsões
    n = len(y)
    alpha = 0.05  # Nível de significância
    t_critical = t.ppf(1 - alpha / 2, df=n - 2)
    interval = t_critical * np.sqrt(mse * (1 / n + (X - np.mean(X)) ** 2 / np.sum((X - np.mean(X)) ** 2)))

    # Tabela de Avaliação dos Pressupostos
    pressupostos_data = {
        "Critério": [
            "Homocedasticidade",
            "Linearidade",
            "Normalidade dos Resíduos",
            "Erros Não Correlacionados (Durbin-Watson)",
            "Significância dos Coeficientes (Teste T)"
        ],
        "Descrição": [
            "Os resíduos devem ter variância constante ao longo do intervalo de valores previstos.",
            "A relação entre as variáveis deve ser linear.",
            "Os resíduos devem seguir uma distribuição normal.",
            f"Valores próximos de 2 indicam que os resíduos não são correlacionados. Durbin-Watson: {dw_stat:.2f}",
            "Coeficientes significativos têm p-valor menor que 0.05."
        ],
        "Avaliação": [
            "Atende" if mse < 0.5 else "Não Atende",
            "Atende" if r2 > 0.8 else "Não Atende",
            "Avaliação adicional necessária",
            "Atende" if 1.5 < dw_stat < 2.5 else "Não Atende",
            "Avaliação adicional necessária"
        ]
    }
    pressupostos_df = pd.DataFrame(pressupostos_data)
    st.subheader("Avaliação dos Pressupostos do Modelo")
    st.table(pressupostos_df)

    # ---------------------------------------------
    # Visualização
    # ---------------------------------------------
    regression_data['Custo Previsto'] = model.predict(X)

    fig_regressao = go.Figure()
    fig_regressao.add_trace(go.Scatter(
        x=regression_data['Quantidade Média'],
        y=regression_data['Custo Médio'],
        mode='markers',
        name='Custos Reais',
        marker=dict(size=8, color='blue')
    ))
    fig_regressao.add_trace(go.Scatter(
        x=regression_data['Quantidade Média'],
        y=regression_data['Custo Previsto'],
        mode='lines',
        name='Linha de Regressão',
        line=dict(color='red')
    ))
    fig_regressao.update_layout(
        title="Regressão Linear - Custos Médios vs Quantidades Médias",
        xaxis_title="Quantidade Média",
        yaxis_title="Custo Médio (R$)",
        legend_title="Legenda"
    )
    st.plotly_chart(fig_regressao)

    # ---------------------------------------------
    # Conclusão da Avaliação do Modelo
    # ---------------------------------------------
    st.subheader("Conclusão do Modelo")
    if r2 > 0.8 and 1.5 < dw_stat < 2.5 and mse < 0.5:
        st.success("A regressão linear apresenta um ótimo desempenho com base nos pressupostos.")
    elif r2 > 0.5:
        st.info("A regressão linear apresenta desempenho moderado. Alguns pressupostos podem não ser totalmente atendidos.")
    else:
        st.warning("A regressão linear apresenta desempenho fraco. Considere ajustar os dados ou o modelo.")

else:
    st.error("Não há dados suficientes para realizar a análise. Verifique os filtros selecionados.")
