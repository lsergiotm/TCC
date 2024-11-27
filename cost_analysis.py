import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing import load_data

# Carregar os dados processados
df = load_data()

# Identificar colunas relacionadas a quantidades
quantidade_cols = [col for col in df.columns if col.startswith("Quantidade")]

# Remover a coluna de "Quantidade total de procedimentos" dos dados
quantidade_cols = [col for col in quantidade_cols if col != "Quantidade total de procedimentos"]

# Título da página
st.title("Análise Comparativa de Quantidades")

# -----------------------------------------
# Seção 1: Comparativo de Maiores Quantidades por Ano
# -----------------------------------------
st.subheader("1. Comparativo de Maiores Quantidades por Ano")

# Criar tabela com maiores valores por ano
if "ano_aih" in df.columns:
    maiores_quantidades_ano = []
    for ano in df["ano_aih"].unique():
        ano_df = df[df["ano_aih"] == ano]
        total_por_ano = ano_df[quantidade_cols].sum()
        procedimento_maior_quantidade = total_por_ano.idxmax()
        maior_quantidade = total_por_ano.max()

        maiores_quantidades_ano.append({
            "Ano": ano,
            "Maior Quantidade": int(maior_quantidade),
            "Procedimento": procedimento_maior_quantidade
        })

    maiores_quantidades_ano_df = pd.DataFrame(maiores_quantidades_ano)
    st.write("Maiores Quantidades por Ano:")
    st.dataframe(maiores_quantidades_ano_df)

    # Gráfico de linha
    fig_line = px.line(
        maiores_quantidades_ano_df,
        x="Ano",
        y="Maior Quantidade",
        title="Tendência das Maiores Quantidades por Ano",
        labels={"Ano": "Ano", "Maior Quantidade": "Quantidade"},
        markers=True
    )
    st.plotly_chart(fig_line)

# -----------------------------------------
# Seção 2: Comparativo Geral de Quantidade por Procedimento
# -----------------------------------------
st.subheader("2. Comparativo Geral de Quantidade por Procedimento")

# Somar todas as colunas de quantidade para obter os totais
quantidade_totais = df[quantidade_cols].sum().reset_index()
quantidade_totais.columns = ["Procedimento", "Quantidade Total"]

# Excluir a linha com "Quantidade total de procedimentos"
quantidade_totais = quantidade_totais[
    quantidade_totais["Procedimento"] != "Quantidade total de procedimentos"
]

# Ordenar pelos maiores valores e pegar os 10 maiores
quantidade_totais = quantidade_totais.sort_values(by="Quantidade Total", ascending=False).head(10)

# Exibir tabela com os 10 maiores procedimentos
st.write("Top 10 Procedimentos com Maiores Quantidades:")
st.dataframe(quantidade_totais)

# Gráfico de pizza
fig_pie = px.pie(
    quantidade_totais,
    names="Procedimento",
    values="Quantidade Total",
    title="Distribuição de Quantidades pelos Top 10 Procedimentos",
    hole=0.4
)
st.plotly_chart(fig_pie)

# Gráfico de barras horizontais
fig_barh = px.bar(
    quantidade_totais,
    x="Quantidade Total",
    y="Procedimento",
    orientation="h",
    title="Top 10 Procedimentos com Maiores Quantidades (Barras Horizontais)",
    labels={"Quantidade Total": "Quantidade Total", "Procedimento": "Procedimento"}
)
st.plotly_chart(fig_barh)

# Gráfico de dispersão (scatter plot)
fig_scatter = px.scatter(
    quantidade_totais,
    x="Quantidade Total",
    y="Procedimento",
    size="Quantidade Total",
    title="Relação de Quantidades com Procedimentos (Dispersão)",
    labels={"Quantidade Total": "Quantidade Total", "Procedimento": "Procedimento"}
)
st.plotly_chart(fig_scatter)
