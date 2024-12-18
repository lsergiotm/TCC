import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing import load_data

# ---------------------------------------------
# Carregar os dados processados
# ---------------------------------------------
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
st.title("Análise Temporal e Sazonal - Gráficos Diversificados")
st.markdown("""
Explore a evolução dos custos e quantidades de procedimentos ao longo do tempo com diferentes tipos de gráficos.
""")

# ---------------------------------------------
# Filtros Interativos
# ---------------------------------------------
st.sidebar.header("Filtros de Dados")

# Filtro por Estado
estados_disponiveis = ['Todos'] + sorted(df['uf_nome'].dropna().unique().tolist())
estado_selecionado = st.sidebar.selectbox(
    "Escolha o estado:",
    options=estados_disponiveis
)
df_filtrado = df.copy()
if estado_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['uf_nome'] == estado_selecionado]

# Filtro por Município
municipios_disponiveis = ['Todos'] + sorted(df_filtrado['nome_municipio'].dropna().unique().tolist())
municipio_selecionado = st.sidebar.selectbox(
    "Escolha o município:",
    options=municipios_disponiveis
)
if municipio_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['nome_municipio'] == municipio_selecionado]

# Filtro por Ano
anos_disponiveis = ['Todos'] + sorted(df_filtrado['ano_aih'].unique())
ano_selecionado = st.sidebar.selectbox("Selecione o Ano:", anos_disponiveis)

if ano_selecionado != 'Todos':
    ano_selecionado = int(ano_selecionado)  # Converter para inteiro
    df_filtrado = df_filtrado[df_filtrado['ano_aih'] == ano_selecionado]

# Verificar se há dados após os filtros
if df_filtrado.empty:
    st.warning("Nenhum dado encontrado com os filtros aplicados. Ajuste os filtros e tente novamente.")
    st.stop()

# ---------------------------------------------
# Análise Temporal
# ---------------------------------------------
# Agrupar dados para análises sazonais
custos_sazonais = df_filtrado.groupby('mes_aih')['Valor total dos procedimentos'].mean().reset_index()
quantidades_sazonais = df_filtrado.groupby('mes_aih')['Quantidade total de procedimentos'].mean().reset_index()

# ---------------------------------------------
# Gráfico de Barras - Custos Médios por Mês
# ---------------------------------------------
st.subheader("Evolução Mensal de Custos Médios (Barras)")

fig_barras_custos = px.bar(
    custos_sazonais,
    x='mes_aih',
    y='Valor total dos procedimentos',
    title="Custos Médios Mensais (Gráfico de Barras)",
    labels={'mes_aih': 'Mês', 'Valor total dos procedimentos': 'Custo Médio (R$)'},
    text_auto=True
)
st.plotly_chart(fig_barras_custos)

# ---------------------------------------------
# Gráfico de Linhas - Custos Médios por Mês
# ---------------------------------------------
st.subheader("Evolução Mensal de Custos Médios (Linhas)")

fig_linha = px.line(
    custos_sazonais,
    x='mes_aih',
    y='Valor total dos procedimentos',
    title="Custos Médios Mensais (Gráfico de Linhas)",
    labels={'mes_aih': 'Mês', 'Valor total dos procedimentos': 'Custo Médio (R$)'}
)
fig_linha.update_traces(mode='lines+markers', line=dict(color='blue'), marker=dict(size=8))
st.plotly_chart(fig_linha)

# ---------------------------------------------
# Gráfico de Área - Quantidades Médias por Mês
# ---------------------------------------------
st.subheader("Evolução Mensal de Quantidades Médias (Área)")

fig_area = px.area(
    quantidades_sazonais,
    x='mes_aih',
    y='Quantidade total de procedimentos',
    title="Quantidades Médias Mensais (Gráfico de Área)",
    labels={'mes_aih': 'Mês', 'Quantidade total de procedimentos': 'Quantidade Média'}
)
st.plotly_chart(fig_area)

# ---------------------------------------------
# Gráfico de Radar - Custos Médios por Mês
# ---------------------------------------------
st.subheader("Distribuição Mensal de Custos Médios (Radar)")

fig_radar = px.line_polar(
    custos_sazonais,
    r='Valor total dos procedimentos',
    theta='mes_aih',
    line_close=True,
    title="Distribuição Mensal de Custos Médios (Radar)",
    template="plotly_dark",
)
fig_radar.update_traces(
    mode='lines+markers',
    marker=dict(size=8, color='rgba(30, 144, 255, 0.7)')
)
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, custos_sazonais['Valor total dos procedimentos'].max() * 1.1],
            title="Custos Médios (R$)"
        )
    )
)
st.plotly_chart(fig_radar)

# ---------------------------------------------
# Gráfico de Linhas - Quantidades Médias por Mês
# ---------------------------------------------
st.subheader("Evolução Mensal de Quantidades Médias (Linhas)")

fig_linha_quantidades = px.line(
    quantidades_sazonais,
    x='mes_aih',
    y='Quantidade total de procedimentos',
    title="Quantidades Médias Mensais (Gráfico de Linhas)",
    labels={'mes_aih': 'Mês', 'Quantidade total de procedimentos': 'Quantidade Média'}
)
fig_linha_quantidades.update_traces(mode='lines+markers', line=dict(color='orange'), marker=dict(size=8))
st.plotly_chart(fig_linha_quantidades)
