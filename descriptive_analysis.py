import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing import load_data
from babel.numbers import format_currency
import unicodedata

# ---------------------------------------------
# Função para carregar e processar os dados
# ---------------------------------------------
@st.cache_data
def load_and_prepare_data():
    # Carregar os dados
    df = load_data()

    # Garantir que a coluna 'ano_aih' seja numérica e válida
    df['ano_aih'] = pd.to_numeric(df['ano_aih'], errors='coerce')
    df = df[df['ano_aih'].notna()]  # Remove valores NaN
    df['ano_aih'] = df['ano_aih'].astype(int)  # Converte para inteiro

    # Normalizar os nomes das colunas
    def normalize_column_name(col):
        col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("utf-8")
        col = col.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        return col

    df.columns = [normalize_column_name(col) for col in df.columns]

    # Exibir nomes de colunas para depuração
    st.write("Colunas do DataFrame e seus Tipos:")
    st.write(df.dtypes)

    # Definir colunas de valores especificados
    vl_cols = [
        "valor_total_dos_procedimentos_com_finalidade_diagnostica",
        "valor_total_dos_procedimentos_clinicos",
        "valor_total_dos_procedimentos_cirurgicos",
        "valor_total_dos_transplantes_de_orgaos,_tecidos_e_celulas",
        "valor_total_dos_medicamentos",
        "valor_total_das_orteses,_proteses_e_materiais_especiais",
        "valor_total_das_acoes_complementares_da_atencao_a_saude",
    ]

    # Verificar se as colunas especificadas existem no DataFrame
    vl_cols = [col for col in vl_cols if col in df.columns]
    if not vl_cols:
        st.warning("Nenhuma das colunas especificadas foi encontrada no DataFrame.")
        st.stop()

    return df, vl_cols

# Função para formatar valores em padrão brasileiro
def formatar_real(valor):
    try:
        return format_currency(valor, 'BRL', locale='pt_BR')
    except:
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ---------------------------------------------
# Configuração da Página
# ---------------------------------------------
st.title("Soma Total dos Procedimentos Especificados")
st.markdown("""
Explore as somas totais dos procedimentos hospitalares com base nos dados filtrados.
""")

# Carregar os dados
df, vl_cols = load_and_prepare_data()

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
anos_disponiveis = ['Todos'] + sorted(df['ano_aih'].dropna().unique().astype(int).astype(str).tolist())
ano_selecionado = st.sidebar.selectbox("Selecione o Ano:", anos_disponiveis)

# Aplicar filtros
df_filtrado = df.copy()
if estado_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['uf_nome'] == estado_selecionado]
if municipio_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['nome_municipio'] == municipio_selecionado]
if ano_selecionado != "Todos":
    ano_selecionado = int(ano_selecionado)
    df_filtrado = df_filtrado[df_filtrado['ano_aih'] == ano_selecionado]

# Verificar DataFrame após os filtros
st.write("DataFrame Filtrado:")
st.write(df_filtrado)

# Verificar se as colunas existem no DataFrame filtrado
vl_cols_filtradas = [col for col in vl_cols if col in df_filtrado.columns]

# Converter todas as colunas para numéricas
for col in vl_cols_filtradas:
    try:
        # Se os valores já forem numéricos, a conversão será direta
        df_filtrado[col] = pd.to_numeric(df_filtrado[col], errors='coerce')
    except:
        # Se não forem numéricos, converta para string primeiro
        df_filtrado[col] = pd.to_numeric(df_filtrado[col].astype(str).str.replace(',', '').str.replace('.', ''), errors='coerce')

# Remover colunas que estão completamente vazias
vl_cols_com_dados = [col for col in vl_cols_filtradas if df_filtrado[col].notna().sum() > 0]

if not vl_cols_com_dados:
    st.warning("Nenhuma das colunas especificadas contém dados numéricos válidos após os filtros aplicados.")
    st.stop()

# Soma correta por coluna
vl_totais = df_filtrado[vl_cols_com_dados].sum(axis=0).fillna(0)

if len(vl_cols_com_dados) != len(vl_totais):
    st.error(f"Erro: O número de colunas ({len(vl_cols_com_dados)}) não corresponde ao número de valores somados ({len(vl_totais)}).")
    st.stop()

# Criar DataFrame para visualização
totais_df = pd.DataFrame({
    "Procedimento": vl_cols_com_dados,
    "Soma Total": vl_totais.values  # Use .values para evitar problemas de indexação
})

# Ordenar os valores pela Soma Total antes de formatar
totais_df = totais_df.sort_values(by="Soma Total", ascending=False)

# Tabela formatada para exibição
totais_df_exibicao = totais_df.copy()
totais_df_exibicao["Soma Total"] = totais_df_exibicao["Soma Total"].apply(formatar_real)

st.write("Tabela de Somatórios:")
st.dataframe(totais_df_exibicao)

# Gráfico de barras
fig_totais = px.bar(
    totais_df,
    x="Procedimento",
    y="Soma Total",
    title="Soma Total de Procedimentos Especificados",
    labels={"Procedimento": "Procedimento", "Soma Total": "Soma Total"},
    height=800,
    width=1200
)

# Ajustes no layout do gráfico
fig_totais.update_layout(
    font=dict(size=16),
    xaxis_title=dict(font=dict(size=18)),
    yaxis_title=dict(font=dict(size=18)),
    margin=dict(l=50, r=50, t=80, b=150),
    legend=dict(
        font=dict(size=14),
        orientation="h",
        x=0.5,
        xanchor="center"
    )
)

st.plotly_chart(fig_totais)