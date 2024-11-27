import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Simulação de carregamento de dados
@st.cache_data
def fetch_data():
    # Dados de exemplo para simulação
    data = {
        'faixa_populacao': ['A', 'B', 'C', 'A', 'C', 'B', 'A'],
        'qtd_total': [100, 200, 300, 150, 350, 250, 180],
        'vl_total': [1000, 2000, 3000, 1500, 3500, 2500, 1800]
    }
    return pd.DataFrame(data)

# Carregar os dados
df = fetch_data()

# Visualizar os dados carregados
st.title("Diagnóstico e Melhoria do Modelo - Gradient Boosting")
st.subheader("Visualização dos Dados Carregados")
st.write("Colunas disponíveis no DataFrame:")
st.write(list(df.columns))
st.write("Prévia dos dados carregados:")
st.write(df)

# Verificar se as colunas necessárias estão presentes
required_columns = ['faixa_populacao', 'qtd_total', 'vl_total']
if all(col in df.columns for col in required_columns):
    st.success("Todas as colunas necessárias foram encontradas!")
else:
    st.error("As colunas necessárias não foram encontradas. Verifique os dados de entrada.")
    st.stop()

# Processamento dos Dados
st.subheader("Processamento dos Dados")
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Atualizado para sparse_output=False
faixa_encoded = pd.DataFrame(
    encoder.fit_transform(df[['faixa_populacao']]),
    columns=encoder.get_feature_names_out(['faixa_populacao'])
)
df = pd.concat([df, faixa_encoded], axis=1).drop(columns=['faixa_populacao'])
st.write("Dados após codificação:", df)

# Separar dados em X (features) e y (target)
X = df.drop(columns=['vl_total'])
y = df['vl_total']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verificar se há dados suficientes
if X_train.empty or X_test.empty:
    st.error("Dados insuficientes para treinar o modelo. Verifique o processamento ou os filtros aplicados.")
    st.stop()

# Modelo Gradient Boosting
st.subheader("Treinamento do Modelo")
n_estimators = st.slider("Número de Estimadores (n_estimators):", min_value=10, max_value=500, value=100, step=10)
learning_rate = st.slider("Taxa de Aprendizado (learning_rate):", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliação do Modelo
st.subheader("Avaliação do Modelo")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Interpretação das métricas
st.subheader("Interpretação das Métricas")
if r2 > 0.9:
    st.success("O modelo apresenta uma excelente capacidade de explicar a variabilidade nos dados.")
elif r2 > 0.75:
    st.info("O modelo apresenta uma boa capacidade de explicação, mas ainda há espaço para melhorias.")
else:
    st.warning("O modelo apresenta baixa capacidade de explicação. Considere ajustar os parâmetros ou utilizar mais dados.")

# Importância das Variáveis
st.subheader("Importância das Variáveis")
feature_importances = pd.DataFrame({
    'Variável': X.columns,
    'Importância': model.feature_importances_
}).sort_values(by='Importância', ascending=False)

st.write("Tabela destacando a contribuição de cada variável para o modelo:")
st.dataframe(feature_importances)

# Gráfico Comparativo
st.subheader("Comparação de Valores Reais vs Previstos")
comparison_df = pd.DataFrame({'Valor Real (R$)': y_test, 'Valor Previsto (R$)': y_pred})
st.dataframe(comparison_df)
