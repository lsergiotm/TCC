import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing import load_data

# ---------------------------------------------
# Função para carregar e processar os dados
# ---------------------------------------------
@st.cache_data
def load_and_prepare_data():
    # Carregar os dados
    df = load_data()

    # Normalizar os nomes das colunas
    df.columns = (
        df.columns.str.strip()  # Remover espaços no início e fim
        .str.replace(" ", "_")  # Substituir espaços por _
        .str.replace("/", "_")  # Substituir barras por _
        .str.replace("-", "_")  # Substituir traços por _
        .str.lower()  # Transformar tudo para minúsculas
    )

    # Selecionar apenas colunas de quantidade total e valor total
    qtd_cols = [col for col in df.columns if col.startswith("quantidade_")]
    vl_cols = [col for col in df.columns if col.startswith("valor_")]

    return df, qtd_cols, vl_cols

# ---------------------------------------------
# Configuração da Página
# ---------------------------------------------
st.title("Soma Total dos Procedimentos")
st.markdown("""
Explore as somas totais dos procedimentos hospitalares com base nos dados filtrados.
""")

# Carregar os dados
df, qtd_cols, vl_cols = load_and_prepare_data()

# ---------------------------------------------
# Filtros Interativos
# ---------------------------------------------
st.sidebar.header("Filtros de Dados")

# Dropdown para Estado
estados_disponiveis = ['Todos'] + sorted(df['uf_nome'].dropna().unique().tolist())
estado_selecionado = st.sidebar.selectbox("Escolha o Estado:", estados_disponiveis)

# Dropdown para Município
if estado_selecionado != 'Todos':
    municipios_disponiveis = ['Todos'] + sorted(df[df['uf_nome'] == estado_selecionado]['nome_municipio'].dropna().unique())
else:
    municipios_disponiveis = ['Todos'] + sorted(df['nome_municipio'].dropna().unique())
municipio_selecionado = st.sidebar.selectbox("Escolha o Município:", municipios_disponiveis)

# Filtro por ano
anos_disponiveis = ['Todos'] + sorted(df['ano_aih'].dropna().unique().astype(str).tolist())
ano_selecionado = st.sidebar.selectbox("Selecione o Ano:", anos_disponiveis)

# Filtro por mês
meses = {
    "Todos": "Todos",
    "Janeiro": "01", "Fevereiro": "02", "Março": "03", "Abril": "04",
    "Maio": "05", "Junho": "06", "Julho": "07", "Agosto": "08",
    "Setembro": "09", "Outubro": "10", "Novembro": "11", "Dezembro": "12"
}
mes_selecionado = st.sidebar.selectbox("Selecione o Mês:", list(meses.keys()))

# Aplicar filtros
df_filtrado = df.copy()
if estado_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['uf_nome'] == estado_selecionado]
if municipio_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['nome_municipio'] == municipio_selecionado]
if ano_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['ano_aih'] == int(ano_selecionado)]
if mes_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['mes_aih'] == int(meses[mes_selecionado])]

# Garantir que existem dados filtrados
if df_filtrado.empty:
    st.warning("Nenhum dado encontrado com os filtros aplicados. Ajuste os filtros e tente novamente.")
    st.stop()

# ---------------------------------------------
# Soma Total das Colunas de Procedimentos
# ---------------------------------------------
st.subheader("Soma Total dos Procedimentos por Quantidade e Valor")

# Somar as colunas de quantidades e valores
qtd_totais = df_filtrado[qtd_cols].sum()
vl_totais = df_filtrado[vl_cols].sum()

# Criar DataFrame para visualização
totais_df = pd.DataFrame({
    "Procedimento": qtd_cols + vl_cols,
    "Soma Total": list(qtd_totais) + list(vl_totais)
})

# Exibir o DataFrame processado
st.write("Tabela de Somatórios:")
st.dataframe(totais_df)

# Gráfico de barras
fig_totais = px.bar(
    totais_df,
    x="Procedimento",
    y="Soma Total",
    title="Soma Total de Procedimentos por Quantidade e Valor",
    labels={"Procedimento": "Procedimento", "Soma Total": "Soma Total"},
)
st.plotly_chart(fig_totais)