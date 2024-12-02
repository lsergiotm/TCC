import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from data_processing import load_data

# Carregar os dados usando o script data_processing.py
df = load_data()

# Garantir que a coluna 'ano_aih' seja numérica e válida
df['ano_aih'] = pd.to_numeric(df['ano_aih'], errors='coerce')
df = df[df['ano_aih'].notna()]  # Remove valores NaN
df['ano_aih'] = df['ano_aih'].astype(int)  # Converte para inteiro

# Garantir que as colunas necessárias existam no DataFrame
df['faixa_populacao'] = pd.to_numeric(df['faixa_populacao'], errors='coerce')
df['Quantidade total de procedimentos'] = pd.to_numeric(df['Quantidade total de procedimentos'], errors='coerce')
df['Valor total dos procedimentos'] = pd.to_numeric(df['Valor total dos procedimentos'], errors='coerce')

# Substituir valores nulos
df.fillna(0, inplace=True)

# -------------------------------------------------
# Filtros de Estado, Município, Ano e Mês
# -------------------------------------------------
st.sidebar.header("Filtros de Dados")

# Dropdown para Estados com seleção única
estados_disponiveis = ['Todos'] + sorted(df['uf_nome'].dropna().unique().tolist())
estado_selecionado = st.sidebar.selectbox("Escolha o estado:", options=estados_disponiveis)

# Filtrar Estados
if estado_selecionado != 'Todos':
    df = df[df['uf_nome'] == estado_selecionado]

# Dropdown para Municípios com seleção única
municipios_disponiveis = ['Todos'] + sorted(df['nome_municipio'].dropna().unique().tolist())
municipio_selecionado = st.sidebar.selectbox("Escolha o município:", options=municipios_disponiveis)

# Filtro por ano
anos_disponiveis = ['Todos'] + sorted(df['ano_aih'].dropna().unique().astype(str).tolist())
ano_selecionado = st.sidebar.selectbox("Selecione o Ano:", anos_disponiveis)

# Aplicar filtros
if municipio_selecionado != "Todos":
    df = df[df['nome_municipio'] == municipio_selecionado]
if ano_selecionado != "Todos":
    df = df[df['ano_aih'] == int(ano_selecionado)]

# Garantir que existam dados após os filtros
if df.empty:
    st.warning("Nenhum dado encontrado com os filtros aplicados. Ajuste os filtros e tente novamente.")
    st.stop()

# -------------------------------------------------
# Configurações do Modelo
# -------------------------------------------------
st.sidebar.header("Configurações do Modelo")
n_estimators = st.sidebar.slider("Número de Árvores (n_estimators):", min_value=10, max_value=500, value=100, step=10)

# -------------------------------------------------
# Treinamento do Modelo
# -------------------------------------------------
X = df[['faixa_populacao', 'Quantidade total de procedimentos']]
y = df['Valor total dos procedimentos']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo Random Forest
model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# -------------------------------------------------
# Avaliação do Modelo
# -------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Importância das variáveis
feature_importances = pd.DataFrame({
    'Variável': X.columns,
    'Importância': model.feature_importances_
}).sort_values(by='Importância', ascending=False)

# Exibição das métricas
st.subheader("Métricas do Modelo")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**MSE:** {mse:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Interpretação das métricas
st.subheader("Interpretação das Métricas")
if r2 > 0.9:
    st.success("O modelo apresenta uma excelente capacidade de explicar a variabilidade nos dados.")
elif r2 > 0.75:
    st.info("O modelo apresenta uma boa capacidade de explicação, mas ainda há espaço para melhorias.")
else:
    st.warning("O modelo apresenta baixa capacidade de explicação. Considere ajustar os parâmetros ou utilizar mais dados.")

# -------------------------------------------------
# Visualizações
# -------------------------------------------------
# Importância das variáveis
st.subheader("Importância das Variáveis")
fig_importance = px.bar(
    feature_importances,
    x='Importância',
    y='Variável',
    orientation='h',
    title="Importância das Variáveis"
)
st.plotly_chart(fig_importance)

# Comparação de valores reais e previstos
st.subheader("Comparação de Valores Reais vs Previstos")
fig_comparison = go.Figure()
fig_comparison.add_trace(go.Scatter(
    x=y_test,
    y=y_pred,
    mode='markers',
    name='Valores Previstos',
    marker=dict(color='blue')
))
fig_comparison.add_trace(go.Scatter(
    x=[y_test.min(), y_test.max()],
    y=[y_test.min(), y_test.max()],
    mode='lines',
    name='Linha Ideal',
    line=dict(color='red', dash='dot')
))
st.plotly_chart(fig_comparison)
