import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing import load_data
from babel.numbers import format_currency

# Função para formatar valores no padrão brasileiro
def formatar_real(valor):
    try:
        return format_currency(valor, 'BRL', locale='pt_BR')
    except:
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ---------------------------------------------
# Carregar os dados processados
# ---------------------------------------------
df = load_data()

# Garantir que a coluna 'ano_aih' seja numérica e válida
df['ano_aih'] = pd.to_numeric(df['ano_aih'], errors='coerce')
df = df[df['ano_aih'].notna()]  # Remove valores NaN
df['ano_aih'] = df['ano_aih'].astype(int)  # Converte para inteiro

# ---------------------------------------------
# Ajustar colunas e forçar tipos numéricos
# ---------------------------------------------
quantidade_cols = [col for col in df.columns if col.startswith("Quantidade")]
valor_cols = [col for col in df.columns if col.startswith("Valor")]

# Garantir que colunas de quantidades e valores sejam numéricas
for col in quantidade_cols + valor_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ---------------------------------------------
# Filtros Interativos
# ---------------------------------------------
st.sidebar.header("Filtros de Dados")

# Filtro por Estado
estados_disponiveis = ['Todos'] + sorted(df['uf_nome'].dropna().unique().tolist())
estado_selecionado = st.sidebar.selectbox("Escolha o Estado:", estados_disponiveis)

df_filtrado = df.copy()
if estado_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['uf_nome'] == estado_selecionado]

# Filtro por Município
municipios_disponiveis = ['Todos'] + sorted(df_filtrado['nome_municipio'].dropna().unique().tolist())
municipio_selecionado = st.sidebar.selectbox("Escolha o Município:", municipios_disponiveis)

if municipio_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['nome_municipio'] == municipio_selecionado]


# Filtro por Ano
anos_disponiveis = ['Todos'] + sorted(df_filtrado['ano_aih'].unique())
ano_selecionado = st.sidebar.selectbox("Selecione o Ano:", anos_disponiveis)

if ano_selecionado != 'Todos':
    ano_selecionado = int(ano_selecionado)  # Garantir que é inteiro
    df_filtrado = df_filtrado[df_filtrado['ano_aih'] == ano_selecionado]

# Verificar se há dados após os filtros
if df_filtrado.empty:
    st.warning("Nenhum dado encontrado com os filtros aplicados. Ajuste os filtros e tente novamente.")
    st.stop()

# ---------------------------------------------
# Análise de Custos Totais
# ---------------------------------------------
st.subheader("Análise de Custos Totais")

if valor_cols:
    custos_totais = pd.DataFrame({
        "Procedimento": valor_cols,
        "Valor Total (R$)": [df_filtrado[col].sum() for col in valor_cols]
    })

    custos_totais = custos_totais.sort_values(by="Valor Total (R$)", ascending=False)
    custos_totais["Valor Total (R$)"] = custos_totais["Valor Total (R$)"].apply(formatar_real)

    st.write("Top Procedimentos por Valor Total:")
    st.table(custos_totais.head(10))

    fig = px.bar(
        custos_totais.head(10),
        x="Valor Total (R$)",
        y="Procedimento",
        orientation="h",
        title="Top Procedimentos por Valor Total"
    )
    st.plotly_chart(fig)

# ---------------------------------------------
# Análise de Quantidades Totais
# ---------------------------------------------
st.subheader("Análise de Quantidades Totais")

if quantidade_cols:
    quantidades_totais = pd.DataFrame({
        "Procedimento": quantidade_cols,
        "Quantidade Total": [df_filtrado[col].sum() for col in quantidade_cols]
    })

    quantidades_totais = quantidades_totais.sort_values(by="Quantidade Total", ascending=False)
    st.write("Top Procedimentos por Quantidade Total:")
    st.table(quantidades_totais.head(10))

    fig = px.bar(
        quantidades_totais.head(10),
        x="Quantidade Total",
        y="Procedimento",
        orientation="h",
        title="Top Procedimentos por Quantidade Total"
    )
    st.plotly_chart(fig)
