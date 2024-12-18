import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px

# Carregar os dados processados
from data_processing import load_data

df = load_data()

# Garantir que a coluna 'ano_aih' seja numérica e válida
df['ano_aih'] = pd.to_numeric(df['ano_aih'], errors='coerce')
df = df[df['ano_aih'].notna()]  # Remove valores NaN
df['ano_aih'] = df['ano_aih'].astype(int)  # Converte para inteiro

# Garantir que as colunas necessárias existam no DataFrame
numeric_columns = ['faixa_populacao', 'Valor total dos procedimentos', 'Quantidade total de procedimentos']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# Título da página
st.title("Validação do Modelo de Clustering com K-Means")
st.markdown("""
Explore os clusters de municípios com base em variáveis selecionadas, avaliando a qualidade dos agrupamentos gerados.
""")

# -------------------------------------------------
# Filtros de Estado, Município, Ano e Mês
# -------------------------------------------------
st.sidebar.header("Filtros de Dados")

# Dropdown para Estados com seleção única e opção "Todos"
estados_disponiveis = ['Todos'] + sorted(df['uf_nome'].dropna().unique().tolist())
estado_selecionado = st.sidebar.selectbox(
    "Escolha o estado:",
    options=estados_disponiveis
)

# Filtrar Estados
if estado_selecionado != 'Todos':
    df = df[df['uf_nome'] == estado_selecionado]
    if estado_selecionado == "Distrito Federal":
        st.info("O Distrito Federal só possui um município (Brasília).")

# Dropdown para Municípios com seleção única e opção "Todos"
municipios_disponiveis = ['Todos'] + sorted(df['nome_municipio'].dropna().unique().tolist())
municipio_selecionado = st.sidebar.selectbox(
    "Escolha o município:",
    options=municipios_disponiveis
)

# Filtro por ano
anos_disponiveis = ['Todos'] + sorted(df['ano_aih'].dropna().unique().astype(str).tolist())
ano_selecionado = st.sidebar.selectbox("Selecione o Ano:", anos_disponiveis)

# Filtrar Ano
if ano_selecionado != "Todos":
    df = df[df['ano_aih'] == int(ano_selecionado)]

# Garantir que existam dados após os filtros
if df.empty:
    st.warning("Nenhum dado encontrado com os filtros aplicados. Ajuste os filtros e tente novamente.")
    st.stop()

# -------------------------------------------------
# Configurações do Modelo
# -------------------------------------------------
st.sidebar.header("Configurações do Modelo")
n_clusters = st.sidebar.slider("Número de Clusters", min_value=2, max_value=10, value=3)

# Preparação dos dados para clustering
X = df[numeric_columns]

# Aplicar K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(X)

# -------------------------------------------------
# Avaliação do Modelo
# -------------------------------------------------
silhouette_avg = silhouette_score(X, df['Cluster'])
calinski_harabasz = calinski_harabasz_score(X, df['Cluster'])
davies_bouldin = davies_bouldin_score(X, df['Cluster'])

# Exibir as métricas de avaliação
st.subheader("Métricas de Avaliação do Modelo")
st.markdown(f"""
- **Silhouette Score:** {silhouette_avg:.2f} (quanto mais próximo de 1, melhor)
- **Calinski-Harabasz Index:** {calinski_harabasz:.2f} (quanto maior, melhor)
- **Davies-Bouldin Index:** {davies_bouldin:.2f} (quanto menor, melhor)
""")

# Adicionar uma tabela de métricas para comparação
metrics_df = pd.DataFrame({
    "Métrica": ["Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Index"],
    "Valor": [silhouette_avg, calinski_harabasz, davies_bouldin]
})
st.table(metrics_df)

# -------------------------------------------------
# Visualização dos Clusters
# -------------------------------------------------
st.subheader("Visualização dos Clusters")
fig = px.scatter_3d(
    df,
    x='faixa_populacao',
    y='Valor total dos procedimentos',
    z='Quantidade total de procedimentos',
    color='Cluster',
    text='nome_municipio',  # Exibir o nome do município no hover
    title="Clusters com Base em Características Selecionadas",
    labels={
        'faixa_populacao': 'Faixa Populacional',
        'Valor total dos procedimentos': 'Custo Total (R$)',
        'Quantidade total de procedimentos': 'Quantidade Total',
        'Cluster': 'Cluster'
    }
)
fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
st.plotly_chart(fig)

# -------------------------------------------------
# Interpretação dos Resultados
# -------------------------------------------------
st.subheader("Interpretação dos Resultados")
if silhouette_avg > 0.5 and davies_bouldin < 1:
    st.success("O modelo apresenta uma boa separação entre os clusters.")
elif silhouette_avg > 0.3 and davies_bouldin < 1.5:
    st.warning("O modelo é moderado, considere ajustar o número de clusters ou as variáveis.")
else:
    st.error("O modelo apresenta clusters fracos. Ajustes são necessários.")

# -------------------------------------------------
# Tabela de Clusters e Municípios
# -------------------------------------------------
st.subheader("Tabela de Clusters e Municípios")
cluster_table = df[['nome_municipio', 'uf_nome', 'Cluster']].sort_values(by='Cluster')
st.table(cluster_table)