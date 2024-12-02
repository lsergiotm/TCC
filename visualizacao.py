import streamlit as st
import pandas as pd
from data_processing import load_data

# ---------------------------------------------
# Função para carregar os dados e padronizar colunas
# ---------------------------------------------
@st.cache_data
def load_and_prepare_data():
    # Carregar os dados
    df = load_data()

    # Garantir que a coluna 'ano_aih' seja numérica e válida
    df['ano_aih'] = pd.to_numeric(df['ano_aih'], errors='coerce')
    df = df[df['ano_aih'].notna()]  # Remove valores NaN
    df['ano_aih'] = df['ano_aih'].astype(int)  # Converte para inteiro

    # Remover colunas duplicadas
    df = df.loc[:, ~df.columns.duplicated()]

    # Renomear colunas para evitar confusão
    df.columns = (
        df.columns.str.strip()  # Remover espaços no início e fim
        .str.replace(" ", "_")  # Substituir espaços por _
        .str.replace("/", "_")  # Substituir barras por _
        .str.replace("-", "_")  # Substituir traços por _
        .str.lower()  # Transformar tudo para minúsculas
    )

    # Verificar e remover colunas irrelevantes (personalize conforme necessário)
    colunas_irrelevantes = [
        "quantidade_de_ações_coletivas_individuais_em_saúde",
        "outros_dados_irrelevantes",  # Adicione mais colunas se necessário
    ]
    df = df.drop(columns=[col for col in colunas_irrelevantes if col in df.columns], errors="ignore")

    return df

# ---------------------------------------------
# Configuração da Página
# ---------------------------------------------
st.title("Visualização de Dados - Internações Hospitalares")
st.markdown("""
Explore e filtre os dados sobre internações hospitalares.  
Exporte os dados filtrados para análise externa e visualize gráficos interativos.
""")

# Carregar os dados
df = load_and_prepare_data()

# ---------------------------------------------
# Filtros Interativos
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

# Aplicar filtros
df_filtrado = df.copy()
# Filtrar por estado
if estado_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['uf_nome'] == estado_selecionado]  # Usar igualdade simples para 'selectbox'
# Filtrar por município
if municipio_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['nome_municipio'] == municipio_selecionado]  # Usar igualdade simples
if ano_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['ano_aih'] == int(ano_selecionado)]
if df_filtrado.empty:
    st.warning("Nenhum dado encontrado com os filtros aplicados. Ajuste os filtros e tente novamente.")
    st.stop()

# ---------------------------------------------
# Exibição dos Dados
# ---------------------------------------------
st.subheader("Dados Filtrados")
st.write(f"Total de registros filtrados: {len(df_filtrado)}")
st.dataframe(df_filtrado)

# ---------------------------------------------
# Exportação dos Dados
# ---------------------------------------------
st.subheader("Exportar Dados")
st.markdown("Você pode exportar os dados filtrados para análise externa.")

# Botão para baixar os dados filtrados
@st.cache_data
def convert_df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtrado)
st.download_button(
    label="Baixar CSV",
    data=csv,
    file_name="dados_filtrados.csv",
    mime="text/csv"
)

# ---------------------------------------------
# Resumo Estatístico
# ---------------------------------------------
st.subheader("Resumo Estatístico dos Dados Filtrados")
st.write(df_filtrado.describe())
