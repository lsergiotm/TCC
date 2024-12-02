import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import pandas as pd
from data_processing import load_data

# Carregar os dados processados
df = load_data()

# Garantir que a coluna 'ano_aih' seja numérica e válida
df['ano_aih'] = pd.to_numeric(df['ano_aih'], errors='coerce')
df = df[df['ano_aih'].notna()]  # Remove valores NaN
df['ano_aih'] = df['ano_aih'].astype(int)  # Converte para inteiro

# Verificar se as colunas necessárias estão presentes
if {'latitude', 'longitude', 'Valor total dos procedimentos', 'Quantidade total de procedimentos', 'nome_municipio'}.issubset(df.columns):
    # Título da página
    st.title("Análise da Distribuição Geográfica - Custos e Quantidades")
    st.markdown("""
    Explore a distribuição geográfica dos custos e quantidades de procedimentos realizados.
    Visualize mapas de calor e marcadores para uma melhor compreensão da distribuição espacial.
    """)

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

    # Geração dinâmica da lista de anos disponíveis
    anos_disponiveis = ['Todos'] + sorted(df['ano_aih'].dropna().unique().astype(int).tolist())
    ano_selecionado = st.sidebar.selectbox("Selecione o Ano:", anos_disponiveis)

    # Aplicar o filtro de ano
    if ano_selecionado != 'Todos':
        ano_selecionado = int(ano_selecionado)
        df_filtrado = df_filtrado[df_filtrado['ano_aih'] == ano_selecionado]
        st.write(f"**Ano Selecionado:** {ano_selecionado}")
    else:
        # Usar o DataFrame original para "Todos os Anos"
        df_filtrado = df.copy()
        st.write("**Todos os Anos Selecionados**")

    # Diagnóstico do DataFrame filtrado
    st.write("**Dados Após os Filtros Aplicados:**")
    st.write(f"Linhas Restantes: {len(df_filtrado)}")
    st.dataframe(df_filtrado.head())

    # Verificar se há dados disponíveis após os filtros
    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado com os filtros aplicados. Ajuste os filtros e tente novamente.")
        st.stop()

    # Garantir que as colunas de custos e quantidades sejam numéricas
    df_filtrado['Valor total dos procedimentos'] = pd.to_numeric(
        df_filtrado['Valor total dos procedimentos'], errors='coerce'
    ).fillna(0)
    df_filtrado['Quantidade total de procedimentos'] = pd.to_numeric(
        df_filtrado['Quantidade total de procedimentos'], errors='coerce'
    ).fillna(0)

    # ---------------------------------------------
    # Diagnóstico e Somatórios
    # ---------------------------------------------
    total_valor = df_filtrado['Valor total dos procedimentos'].sum()
    total_quantidade = df_filtrado['Quantidade total de procedimentos'].sum()

    st.write(f"**Somatório Total de Valores:** R$ {total_valor:,.2f}")
    st.write(f"**Somatório Total de Quantidades:** {int(total_quantidade):,}")

    # ---------------------------------------------
    # Mapa de Calor - Distribuição Geográfica de Custos e Quantidades
    # ---------------------------------------------
    st.subheader("Mapa de Calor - Distribuição Geográfica de Custos e Quantidades")

    # Criar o mapa base
    mapa_calor = folium.Map(location=[df_filtrado['latitude'].mean(), df_filtrado['longitude'].mean()], zoom_start=6)

    # Processar os dados para o mapa de calor
    heat_data = df_filtrado[['latitude', 'longitude', 'Valor total dos procedimentos', 'Quantidade total de procedimentos']].dropna()

    # Combinar os pesos de custos e quantidades (ex.: somar ou calcular média)
    heat_data['peso'] = heat_data['Valor total dos procedimentos'] + heat_data['Quantidade total de procedimentos']

    # Criar lista de dados para o HeatMap
    heatmap_data = heat_data[['latitude', 'longitude', 'peso']].values.tolist()

    # Adicionar HeatMap ao mapa
    HeatMap(data=heatmap_data, radius=15, blur=10, max_zoom=1, min_opacity=0.5).add_to(mapa_calor)

    # Exibir o mapa no Streamlit
    st_folium(mapa_calor, width=800, height=500)

    # ---------------------------------------------
    # Mapa com Marcadores - Custos e Quantidade
    # ---------------------------------------------
    st.subheader("Mapa com Marcadores de Custos e Quantidade")
    mapa_marcadores_custos = folium.Map(location=[df_filtrado['latitude'].mean(), df_filtrado['longitude'].mean()], zoom_start=6)
    for _, row in df_filtrado.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Município: {row['nome_municipio']}<br>Custo: R$ {row['Valor total dos procedimentos']:.2f}<br>Quantidade: {int(row['Quantidade total de procedimentos'])}",
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(mapa_marcadores_custos)
    st_folium(mapa_marcadores_custos, width=800, height=500)

else:
    st.error("As colunas 'latitude', 'longitude', 'Valor total dos procedimentos', 'Quantidade total de procedimentos' ou 'nome_municipio' não estão disponíveis no dataset.")
