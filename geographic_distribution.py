import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import pandas as pd
from data_processing import load_data

# Carregar os dados processados
df = load_data()

# Garantir que as colunas necessárias estão presentes
if 'latitude' in df.columns and 'longitude' in df.columns:
    # Título da página
    st.title("Análise da Distribuição Geográfica - Custos e Quantidades")
    st.markdown("""
    Nesta análise, exploramos a distribuição geográfica dos custos e das quantidades de procedimentos realizados na região.
    Os dados são exibidos em mapas e gráficos para facilitar a análise.
    """)

    # ---------------------------------------------
    # Filtro de ano
    # ---------------------------------------------
    st.subheader("Selecione o Ano para Análise")
    anos = sorted(df['ano_aih'].unique())
    ano_selecionado = st.selectbox("Ano:", anos)

    # Filtrar os dados pelo ano selecionado
    df_filtrado = df[df['ano_aih'] == ano_selecionado]

    # Verificar se há dados disponíveis para o ano selecionado
    if not df_filtrado.empty:
        # Garantir que as colunas de custos e quantidades sejam numéricas
        df_filtrado['Valor total dos procedimentos'] = pd.to_numeric(
            df_filtrado['Valor total dos procedimentos'], errors='coerce'
        ).fillna(0)
        df_filtrado['Quantidade total de procedimentos'] = pd.to_numeric(
            df_filtrado['Quantidade total de procedimentos'], errors='coerce'
        ).fillna(0)

        # ---------------------------------------------
        # Mapa de Calor (HeatMap) com Custos
        # ---------------------------------------------
        st.subheader("Mapa de Calor - Distribuição Geográfica de Custos")
        
        # Criar o mapa centralizado na média das coordenadas
        mapa_custos = folium.Map(location=[df_filtrado['latitude'].mean(), df_filtrado['longitude'].mean()], zoom_start=8)
        
        # Preparar os dados para o HeatMap (latitude, longitude, custos)
        heat_data_custos = df_filtrado[['latitude', 'longitude', 'Valor total dos procedimentos']].dropna()
        
        # Adicionar o HeatMap ao mapa, utilizando os custos como peso
        HeatMap(data=heat_data_custos.values.tolist(), radius=15, blur=10, max_zoom=1, min_opacity=0.5).add_to(mapa_custos)
        
        # Exibir o mapa no Streamlit
        st_folium(mapa_custos, width=800, height=500)

        # ---------------------------------------------
        # Mapa com Marcadores de Custos
        # ---------------------------------------------
        st.subheader("Mapa com Marcadores de Custos")
        
        # Criar o mapa com marcadores
        mapa_marcadores_custos = folium.Map(location=[df_filtrado['latitude'].mean(), df_filtrado['longitude'].mean()], zoom_start=8)

        for _, row in df_filtrado.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"Custo: R$ {row['Valor total dos procedimentos']:.2f}",
                icon=folium.Icon(color="green", icon="info-sign")
            ).add_to(mapa_marcadores_custos)
        
        # Exibir o mapa com marcadores no Streamlit
        st_folium(mapa_marcadores_custos, width=800, height=500)

        # ---------------------------------------------
        # Mapa de Calor (HeatMap) com Quantidades
        # ---------------------------------------------
        st.subheader("Mapa de Calor - Distribuição Geográfica de Quantidades")
        
        # Criar o mapa centralizado na média das coordenadas
        mapa_quantidades = folium.Map(location=[df_filtrado['latitude'].mean(), df_filtrado['longitude'].mean()], zoom_start=8)
        
        # Preparar os dados para o HeatMap (latitude, longitude, quantidades)
        heat_data_quantidades = df_filtrado[['latitude', 'longitude', 'Quantidade total de procedimentos']].dropna()
        
        # Adicionar o HeatMap ao mapa, utilizando as quantidades como peso
        HeatMap(data=heat_data_quantidades.values.tolist(), radius=15, blur=10, max_zoom=1, min_opacity=0.5).add_to(mapa_quantidades)
        
        # Exibir o mapa no Streamlit
        st_folium(mapa_quantidades, width=800, height=500)

        # ---------------------------------------------
        # Mapa com Marcadores de Quantidades
        # ---------------------------------------------
        st.subheader("Mapa com Marcadores de Quantidades")
        
        # Criar o mapa com marcadores
        mapa_marcadores_quantidades = folium.Map(location=[df_filtrado['latitude'].mean(), df_filtrado['longitude'].mean()], zoom_start=8)

        for _, row in df_filtrado.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"Quantidade: {int(row['Quantidade total de procedimentos'])}",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(mapa_marcadores_quantidades)
        
        # Exibir o mapa com marcadores no Streamlit
        st_folium(mapa_marcadores_quantidades, width=800, height=500)

    else:
        st.warning(f"Não há dados disponíveis para o ano {ano_selecionado}.")
else:
    st.error("As colunas 'latitude', 'longitude', 'Valor total dos procedimentos' ou 'Quantidade total de procedimentos' não estão disponíveis no dataset.")
