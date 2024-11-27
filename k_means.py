import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px

# Carregar os dados processados
from data_processing import load_data

df = load_data()

# Garantir que as colunas necessárias existam no DataFrame
numeric_columns = ['faixa_populacao', 'Valor total dos procedimentos', 'Quantidade total de procedimentos']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# Título da página
st.title("Validação do Modelo de Clustering")

# Selecionar o número de clusters
st.sidebar.header("Configurações do Modelo")
n_clusters = st.sidebar.slider("Número de Clusters", min_value=2, max_value=10, value=3)

# Preparação dos dados para clustering
X = df[numeric_columns]

# Aplicar K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(X)

# Avaliação do modelo
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
st.subheader("Tabela de Métricas")
metrics_df = pd.DataFrame({
    "Métrica": ["Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Index"],
    "Valor": [silhouette_avg, calinski_harabasz, davies_bouldin]
})
st.table(metrics_df)

# Interpretação dos resultados
st.subheader("Interpretação dos Resultados")
if silhouette_avg > 0.5 and davies_bouldin < 1:
    st.success("O modelo apresenta uma boa separação entre os clusters.")
elif silhouette_avg > 0.3 and davies_bouldin < 1.5:
    st.warning("O modelo é moderado, considere ajustar o número de clusters ou as variáveis.")
else:
    st.error("O modelo apresenta clusters fracos. Ajustes são necessários.")

# Exibir as coordenadas das centróides
st.subheader("Centróides dos Clusters")
centroid_df = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_columns)
centroid_df['Cluster'] = range(n_clusters)
st.dataframe(centroid_df)

# Contagem de elementos por cluster
st.subheader("Quantidade de Municípios em Cada Cluster")
cluster_counts = df['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Número de Municípios']
st.dataframe(cluster_counts)

# Visualizar os clusters
st.subheader("Visualização dos Clusters")
fig = px.scatter_3d(
    df,
    x='faixa_populacao',
    y='Valor total dos procedimentos',
    z='Quantidade total de procedimentos',
    color='Cluster',
    title="Clusters com Base em Características Selecionadas",
    labels={
        'faixa_populacao': 'Faixa Populacional',
        'Valor total dos procedimentos': 'Custo Total (R$)',
        'Quantidade total de procedimentos': 'Quantidade Total',
        'Cluster': 'Cluster'
    }
)
st.plotly_chart(fig)
