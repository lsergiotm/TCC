import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Carregar os dados
from data_processing import load_data
df = load_data()

# Visualizar os dados carregados
st.title("Diagnóstico e Melhoria do Modelo - Gradient Boosting")
st.markdown("""
Este modelo utiliza o **Gradient Boosting Regressor** para prever o **valor total dos procedimentos** com base nas variáveis disponíveis no conjunto de dados. 
A análise inclui filtros de dados, avaliação de métricas do modelo, importância das variáveis e comparação de valores reais com previstos.
""")

st.subheader("Visualização dos Dados Carregados")
st.write("Colunas disponíveis no DataFrame:")
st.write(df.columns.tolist())
st.dataframe(df.head())

# -------------------------------------------------
# Filtros de Dados
# -------------------------------------------------
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
anos = ['Todos'] + sorted(df['ano_aih'].dropna().unique().astype(str).tolist())
ano_selecionado = st.sidebar.selectbox("Escolha o ano:", anos)

# Filtro por mês
meses = {
    "Todos": "Todos",
    "Janeiro": "01", "Fevereiro": "02", "Março": "03", "Abril": "04",
    "Maio": "05", "Junho": "06", "Julho": "07", "Agosto": "08",
    "Setembro": "09", "Outubro": "10", "Novembro": "11", "Dezembro": "12"
}
mes_selecionado = st.sidebar.selectbox("Escolha o mês:", list(meses.keys()))

# Filtrar os dados
df_filtrado = df.copy()
if estado_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['uf_nome'] == estado_selecionado]
if municipio_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['nome_municipio'] == municipio_selecionado]
if ano_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['ano_aih'] == int(ano_selecionado)]
if mes_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['mes_aih'] == int(meses[mes_selecionado])]

if df_filtrado.empty:
    st.error("Nenhum dado encontrado com os filtros aplicados. Ajuste os filtros e tente novamente.")
    st.stop()

# -------------------------------------------------
# Garantir que a coluna de target exista
# -------------------------------------------------
target_column_name = None
for col in df_filtrado.columns:
    if "total" in col.lower() and "procedimentos" in col.lower():
        target_column_name = col
        break

if not target_column_name:
    st.error("Não foi possível localizar a coluna de 'Valor total dos procedimentos' nos dados filtrados.")
    st.stop()

st.write(f"Coluna de target identificada: **{target_column_name}**")

# -------------------------------------------------
# Codificação de Colunas Categóricas
# -------------------------------------------------
categorical_columns = df_filtrado.select_dtypes(include=['object']).columns.tolist()
if categorical_columns:
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = pd.DataFrame(
        encoder.fit_transform(df_filtrado[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    df_filtrado = pd.concat([df_filtrado.reset_index(drop=True), encoded_features], axis=1).drop(columns=categorical_columns)

# Garantir que só existam colunas numéricas após codificação
df_filtrado = df_filtrado.select_dtypes(include=['int64', 'float64'])

# -------------------------------------------------
# Configuração do Modelo Gradient Boosting
# -------------------------------------------------
st.subheader("Treinamento do Modelo")
st.markdown("""
O **Gradient Boosting Regressor** é um modelo baseado em árvores de decisão, ideal para prever valores numéricos em problemas complexos.
Este modelo aprende iterativamente, reduzindo erros em cada etapa.
""")
n_estimators = st.slider("Número de Estimadores (n_estimators):", min_value=10, max_value=500, value=100, step=10)
learning_rate = st.slider("Taxa de Aprendizado (learning_rate):", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

# Dividir os dados em X (features) e y (target)
X = df_filtrado.drop(columns=[target_column_name], errors='ignore')
y = df_filtrado[target_column_name]

if X.empty or y.empty:
    st.error("Dados insuficientes para treinar o modelo. Verifique os filtros aplicados.")
    st.stop()

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# -------------------------------------------------
# Avaliação do Modelo
# -------------------------------------------------
st.subheader("Avaliação do Modelo")
st.markdown("""
As métricas abaixo avaliam o desempenho do modelo:
- **MAE**: Mostra o erro absoluto médio entre os valores reais e previstos.
- **RMSE**: Penaliza mais erros grandes, fornecendo uma visão geral da precisão.
- **R²**: Mede a proporção da variância explicada pelo modelo. Quanto mais próximo de 1, melhor o ajuste.
""")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write(f"**Erro Absoluto Médio (MAE):** R$ {mae:,.2f}")
st.write(f"**Erro Quadrático Médio (MSE):** R$ {mse:,.2f}")
st.write(f"**Raiz do Erro Quadrático Médio (RMSE):** R$ {rmse:,.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# -------------------------------------------------
# Gráfico de Comparação
# -------------------------------------------------
st.subheader("Gráfico de Comparação: Valores Reais vs Previstos")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Valores Reais (R$)')
ax.set_ylabel('Valores Previstos (R$)')
ax.set_title('Comparação entre Valores Reais e Previstos')
st.pyplot(fig)

# -------------------------------------------------
# Importância das Variáveis
# -------------------------------------------------
st.subheader("Importância das Variáveis")
st.markdown("""
A importância das variáveis indica quais fatores têm maior influência no modelo.
Isso é útil para identificar quais características devem ser priorizadas ou monitoradas.
""")
feature_importances = pd.DataFrame({
    'Variável': X.columns,
    'Importância': model.feature_importances_
}).sort_values(by='Importância', ascending=False)
st.dataframe(feature_importances)

# -------------------------------------------------
# Interpretação Final
# -------------------------------------------------
st.subheader("Interpretação Final do Modelo")
if r2 > 0.75 and rmse < mae * 1.2:
    st.success("O modelo apresenta um desempenho sólido, com bom ajuste aos dados!")
elif r2 > 0.5:
    st.warning("O modelo apresenta desempenho moderado.")
else:
    st.error("O modelo apresenta desempenho fraco.")
