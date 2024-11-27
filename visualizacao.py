import streamlit as st
from data_processing import load_data

# ---------------------------------------------
# Função para carregar os dados
# ---------------------------------------------
@st.cache_data
def load_and_prepare_data():
    # Carregar os dados
    df = load_data()

    # Exibir as colunas disponíveis
    st.write("Colunas disponíveis no DataFrame:", df.columns)

    return df

# ---------------------------------------------
# Configuração da Página
# ---------------------------------------------
st.title("Visualização de Dados - Internações Hospitalares")
st.markdown("""
Esta página permite explorar e filtrar os dados sobre internações hospitalares.
Você também pode exportar os dados filtrados para análise externa.
""")

# Carregar os dados
df = load_and_prepare_data()

# ---------------------------------------------
# Filtros Interativos
# ---------------------------------------------
st.sidebar.header("Filtros")

# Filtro por Ano
anos_disponiveis = sorted(df['ano_aih'].unique())
ano_selecionado = st.sidebar.multiselect("Selecione o(s) Ano(s):", options=anos_disponiveis, default=anos_disponiveis)

# Filtro por Município
municipios_disponiveis = sorted(df['nome_municipio'].unique())
municipios_selecionados = st.sidebar.multiselect("Selecione o(s) Município(s):", options=municipios_disponiveis, default=municipios_disponiveis)

# Filtro por Faixa Populacional (se disponível)
if 'Faixa_Populacao' in df.columns:
    faixas_disponiveis = sorted(df['Faixa_Populacao'].unique())
    faixas_selecionadas = st.sidebar.multiselect("Selecione a(s) Faixa(s) Populacional(is):", options=faixas_disponiveis, default=faixas_disponiveis)
else:
    faixas_selecionadas = None

# ---------------------------------------------
# Aplicar Filtros
# ---------------------------------------------
df_filtrado = df.copy()

# Filtrar por Ano
if ano_selecionado:
    df_filtrado = df_filtrado[df_filtrado['ano_aih'].isin(ano_selecionado)]

# Filtrar por Município
if municipios_selecionados:
    df_filtrado = df_filtrado[df_filtrado['nome_municipio'].isin(municipios_selecionados)]

# Filtrar por Faixa Populacional (se disponível)
if faixas_selecionadas and 'Faixa_Populacao' in df.columns:
    df_filtrado = df_filtrado[df_filtrado['Faixa_Populacao'].isin(faixas_selecionadas)]

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

if not df_filtrado.empty:
    csv = convert_df_to_csv(df_filtrado)
    st.download_button(
        label="Baixar CSV",
        data=csv,
        file_name="dados_filtrados.csv",
        mime="text/csv"
    )
else:
    st.warning("Nenhum dado disponível para exportação.")

# ---------------------------------------------
# Visualizações Básicas
# ---------------------------------------------
st.subheader("Resumo Estatístico dos Dados Filtrados")
st.write(df_filtrado.describe())
