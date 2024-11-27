import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import pandas as pd
from data_processing import load_data

# Carregar os dados processados
df = load_data()

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
    estado_selecionado = st.sidebar.selectbox("Escolha o estado:", estados_disponiveis)
    df_filtrado = df.copy()
    if estado_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['uf_nome'] == estado_selecionado]

    # Filtro por Município
    municipios_disponiveis = ['Todos'] + sorted(df_filtrado['nome_municipio'].dropna().unique().tolist())
    municipio_selecionado = st.sidebar.selectbox("Escolha o município:", municipios_disponiveis)
    if municipio_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['nome_municipio'] == municipio_selecionado]

    # Filtro por Ano
    anos_disponiveis = ['Todos'] + sorted(df_filtrado['ano_aih'].dropna().unique().tolist())
    ano_selecionado = st.sidebar.selectbox("Selecione o Ano:", anos_disponiveis)
    if ano_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['ano_aih'] == int(ano_selecionado)]

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
    # Mapa de Calor - Custos
    # ---------------------------------------------
    st.subheader("Mapa de Calor - Distribuição Geográfica de Custos")
    mapa_custos = folium.Map(location=[df_filtrado['latitude'].mean(), df_filtrado['longitude'].mean()], zoom_start=6)
    heat_data_custos = df_filtrado[['latitude', 'longitude', 'Valor total dos procedimentos']].dropna()
    HeatMap(data=heat_data_custos.values.tolist(), radius=15, blur=10, max_zoom=1, min_opacity=0.5).add_to(mapa_custos)
    st_folium(mapa_custos, width=800, height=500)

    # ---------------------------------------------
    # Mapa com Marcadores - Custos
    # ---------------------------------------------
    st.subheader("Mapa com Marcadores de Custos")
    mapa_marcadores_custos = folium.Map(location=[df_filtrado['latitude'].mean(), df_filtrado['longitude'].mean()], zoom_start=6)
    for _, row in df_filtrado.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Município: {row['nome_municipio']}<br>Custo: R$ {row['Valor total dos procedimentos']:.2f}",
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(mapa_marcadores_custos)
    st_folium(mapa_marcadores_custos, width=800, height=500)

    # ---------------------------------------------
    # Mapa de Calor - Quantidades
    # ---------------------------------------------
    st.subheader("Mapa de Calor - Distribuição Geográfica de Quantidades")
    mapa_quantidades = folium.Map(location=[df_filtrado['latitude'].mean(), df_filtrado['longitude'].mean()], zoom_start=6)
    heat_data_quantidades = df_filtrado[['latitude', 'longitude', 'Quantidade total de procedimentos']].dropna()
    HeatMap(data=heat_data_quantidades.values.tolist(), radius=15, blur=10, max_zoom=1, min_opacity=0.5).add_to(mapa_quantidades)
    st_folium(mapa_quantidades, width=800, height=500)

    # ---------------------------------------------
    # Mapa com Marcadores - Quantidades
    # ---------------------------------------------
    st.subheader("Mapa com Marcadores de Quantidades")
    mapa_marcadores_quantidades = folium.Map(location=[df_filtrado['latitude'].mean(), df_filtrado['longitude'].mean()], zoom_start=6)
    for _, row in df_filtrado.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Município: {row['nome_municipio']}<br>Quantidade: {int(row['Quantidade total de procedimentos'])}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(mapa_marcadores_quantidades)
    st_folium(mapa_marcadores_quantidades, width=800, height=500)

else:
    st.error("As colunas 'latitude', 'longitude', 'Valor total dos procedimentos', 'Quantidade total de procedimentos' ou 'nome_municipio' não estão disponíveis no dataset.")
