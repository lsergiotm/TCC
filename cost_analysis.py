import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing import load_data

# ---------------------------------------------
# Carregar os dados processados
# ---------------------------------------------
df = load_data()

# ---------------------------------------------
# Ajustar colunas e forçar tipos numéricos
# ---------------------------------------------
# Identificar colunas relacionadas a quantidades e valores
quantidade_cols = [col for col in df.columns if col.startswith("Quantidade")]
valor_cols = [col for col in df.columns if col.startswith("Valor")]

# Garantir que colunas de quantidades e valores sejam numéricas
for col in quantidade_cols + valor_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Remover colunas de totais para evitar duplicação
quantidade_cols = [col for col in quantidade_cols if col != "Quantidade total de procedimentos"]
valor_cols = [col for col in valor_cols if col != "Valor total dos procedimentos"]

# Configuração do título
st.title("Análise Comparativa de Custos e Quantidades")
st.markdown("""
Explore os custos totais e as quantidades de procedimentos com base em diferentes filtros e visualizações interativas.
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
anos_disponiveis = ['Todos'] + sorted(df_filtrado['ano_aih'].dropna().unique().tolist())
ano_selecionado = st.sidebar.selectbox("Selecione o Ano:", anos_disponiveis)
if ano_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['ano_aih'] == int(ano_selecionado)]

# Filtro por Mês
meses = {
    "Todos": "Todos",
    "Janeiro": "01", "Fevereiro": "02", "Março": "03", "Abril": "04",
    "Maio": "05", "Junho": "06", "Julho": "07", "Agosto": "08",
    "Setembro": "09", "Outubro": "10", "Novembro": "11", "Dezembro": "12"
}
mes_selecionado = st.sidebar.selectbox("Selecione o Mês:", list(meses.keys()))
if mes_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['mes_aih'] == int(meses[mes_selecionado])]

# Verificar se há dados após os filtros
if df_filtrado.empty:
    st.warning("Nenhum dado encontrado com os filtros aplicados. Ajuste os filtros e tente novamente.")
    st.stop()

# ---------------------------------------------
# Análise de Custos Totais
# ---------------------------------------------
st.subheader("1. Análise de Custos Totais")

if valor_cols:
    custos_totais = pd.DataFrame({
        "Procedimento": valor_cols,
        "Valor Total (R$)": [df_filtrado[col].sum() for col in valor_cols]
    })

    # Ordenar pelos maiores valores
    custos_totais = custos_totais.sort_values(by="Valor Total (R$)", ascending=False)

    # Exibir tabela
    st.write("Top Procedimentos por Valor Total:")
    st.dataframe(custos_totais.head(10))

    # Gráfico de barras
    fig_bar_valores = px.bar(
        custos_totais.head(10),
        x="Valor Total (R$)",
        y="Procedimento",
        orientation="h",
        title="Top Procedimentos por Valor Total",
        labels={"Valor Total (R$)": "Valor Total (R$)", "Procedimento": "Procedimento"}
    )
    st.plotly_chart(fig_bar_valores)

# ---------------------------------------------
# Análise de Quantidades Totais
# ---------------------------------------------
st.subheader("2. Análise de Quantidades Totais")

if quantidade_cols:
    quantidades_totais = pd.DataFrame({
        "Procedimento": quantidade_cols,
        "Quantidade Total": [df_filtrado[col].sum() for col in quantidade_cols]
    })

    # Ordenar pelos maiores valores
    quantidades_totais = quantidades_totais.sort_values(by="Quantidade Total", ascending=False)

    # Exibir tabela
    st.write("Top Procedimentos por Quantidade Total:")
    st.dataframe(quantidades_totais.head(10))

    # Gráfico de barras
    fig_bar_quantidades = px.bar(
        quantidades_totais.head(10),
        x="Quantidade Total",
        y="Procedimento",
        orientation="h",
        title="Top Procedimentos por Quantidade Total",
        labels={"Quantidade Total": "Quantidade Total", "Procedimento": "Procedimento"}
    )
    st.plotly_chart(fig_bar_quantidades)
