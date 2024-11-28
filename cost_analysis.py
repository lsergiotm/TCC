import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px
from data_processing import load_data

# Carregar os dados processados
df = load_data()

# Ajustar colunas para garantir que as variáveis numéricas sejam interpretadas corretamente
numeric_columns = ['faixa_populacao', 'Valor total dos procedimentos', 'Quantidade total de procedimentos']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Título da página
st.title("Validação do Modelo de Clustering com K-Means")
st.markdown("""
Explore os clusters de municípios com base em variáveis selecionadas, avaliando a qualidade dos agrupamentos gerados.
""")

# ---------------------------------------------
# Filtros de Dados
# ---------------------------------------------
st.sidebar.header("Filtros de Dados")

# Filtro por Estado
estados_disponiveis = ['Todos'] + sorted(df['uf_nome'].dropna().unique().tolist())
estado_selecionado = st.sidebar.selectbox(
    "Escolha o estado:",
    options=estados_disponiveis
)

# Filtro por Município
df_filtrado = df.copy()
if estado_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['uf_nome'] == estado_selecionado]

municipios_disponiveis = ['Todos'] + sorted(df_filtrado['nome_municipio'].dropna().unique().tolist())
municipio_selecionado = st.sidebar.selectbox(
    "Escolha o município:",
    options=municipios_disponiveis
)
if municipio_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['nome_municipio'] == municipio_selecionado]

# Filtro por Ano
anos_disponiveis = ['Todos'] + sorted(df_filtrado['ano_aih'].dropna().unique().tolist())
ano_selecionado = st.sidebar.selectbox("Escolha o ano:", anos_disponiveis)
if ano_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['ano_aih'] == int(ano_selecionado)]

# Filtro por Mês
meses = {
    "Todos": "Todos",
    "Janeiro": "01", "Fevereiro": "02", "Março": "03", "Abril": "04",
    "Maio": "05", "Junho": "06", "Julho": "07", "Agosto": "08",
    "Setembro": "09", "Outubro": "10", "Novembro": "11", "Dezembro": "12"
}
mes_selecionado = st.sidebar.selectbox("Escolha o mês:", list(meses.keys()))
if mes_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['mes_aih'] == int(meses[mes_selecionado])]

# Verificar disponibilidade de dados
registros_antes = len(df)
registros_apos_filtro = len(df_filtrado)
st.write(f"**Registros disponíveis antes dos filtros:** {registros_antes}")
st.write(f"**Registros após filtros:** {registros_apos_filtro}")

if df_filtrado.empty:
    st.error("Nenhum dado encontrado com os filtros aplicados. Ajuste os filtros e tente novamente.")
    st.stop()

# ---------------------------------------------
# Configuração do Modelo
# ---------------------------------------------
st.sidebar.header("Configurações do Modelo")
n_clusters = st.sidebar.slider("Número de Clusters", min_value=2, max_value=10, value=3)

# Preparar os dados para clustering
X = df_filtrado[numeric_columns]

# Aplicar K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
df_filtrado['Cluster'] = kmeans.fit_predict(X)

# ---------------------------------------------
# Avaliação do Modelo
# ---------------------------------------------
silhouette_avg = silhouette_score(X, df_filtrado['Cluster'])
calinski_harabasz = calinski_harabasz_score(X, df_filtrado['Cluster'])
davies_bouldin = davies_bouldin_score(X, df_filtrado['Cluster'])

# Exibir as métricas de avaliação
st.subheader("Métricas de Avaliação do Modelo")
metrics_df = pd.DataFrame({
    "Métrica": ["Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Index"],
    "Valor": [silhouette_avg, calinski_harabasz, davies_bouldin],
    "Interpretação": [
        "Bom (> 0.5)" if silhouette_avg > 0.5 else "Ruim",
        "Melhor quanto maior",
        "Melhor quanto menor (< 1.5)" if davies_bouldin < 1.5 else "Ruim"
    ]
})
st.table(metrics_df)

# Interpretação do desempenho
st.subheader("Interpretação dos Resultados")
if silhouette_avg > 0.5 and davies_bouldin < 1:
    st.success("O modelo apresenta uma boa separação entre os clusters.")
elif silhouette_avg > 0.3 and davies_bouldin < 1.5:
    st.warning("O modelo é moderado, considere ajustar o número de clusters ou as variáveis.")
else:
    st.error("O modelo apresenta clusters fracos. Ajustes são necessários.")

# ---------------------------------------------
# Visualização dos Clusters
# ---------------------------------------------
st.subheader("Visualização dos Clusters")
fig = px.scatter_3d(
    df_filtrado,
    x='faixa_populacao',
    y='Valor total dos procedimentos',
    z='Quantidade total de procedimentos',
    color='Cluster',
    hover_name='nome_municipio',
    title="Clusters com Base em Características Selecionadas",
    labels={
        'faixa_populacao': 'Faixa Populacional',
        'Valor total dos procedimentos': 'Custo Total (R$)',
        'Quantidade total de procedimentos': 'Quantidade Total',
        'Cluster': 'Cluster'
    }
)
st.plotly_chart(fig)

# Exibição dos clusters gerados
st.subheader("Resumo dos Clusters")
cluster_summary = df_filtrado.groupby('Cluster').agg({
    'faixa_populacao': 'mean',
    'Valor total dos procedimentos': 'mean',
    'Quantidade total de procedimentos': 'mean'
}).reset_index()
cluster_summary.columns = ['Cluster', 'Faixa Populacional (Média)', 'Valor Total (Média)', 'Quantidade Total (Média)']
st.dataframe(cluster_summary)