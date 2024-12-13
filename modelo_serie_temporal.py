import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from babel.numbers import format_currency
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Função para formatar valores no padrão brasileiro
def formatar_real(valor):
    try:
        return format_currency(valor, 'BRL', locale='pt_BR')
    except:
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def formatar_quantidade(quantidade):
    try:
        return f"{quantidade:,.0f}".replace(",", "X").replace(".", ",").replace("X", ",")
    except:
        return str(quantidade)

# Funções de Métricas de Avaliação
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0  # Evitar divisões por zero
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.any() else float('inf')

def calculate_accuracy(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return 100 - calculate_mape(actual, predicted) if mask.any() else float('inf')

# Carregar os dados processados
from data_processing import load_data
df = load_data()

# Garantir que a coluna 'ano_aih' seja numérica e válida
df['ano_aih'] = pd.to_numeric(df['ano_aih'], errors='coerce')
df = df[df['ano_aih'].notna()]  # Remove valores NaN
df['ano_aih'] = df['ano_aih'].astype(int)

# Garantir que os dados necessários estão em formato numérico
df['Valor total dos procedimentos'] = pd.to_numeric(df['Valor total dos procedimentos'], errors='coerce').fillna(0)
df['Quantidade total de procedimentos'] = pd.to_numeric(df['Quantidade total de procedimentos'], errors='coerce').fillna(0)

# Filtrar apenas os anos relevantes
df = df[df['ano_aih'].between(2019, 2025)]

# ---------------------------------------------
# Adicionar filtros de Estado e Município
# ---------------------------------------------
st.sidebar.header("Filtros de Dados")

# Filtro de Estado
estados_disponiveis = sorted(df['uf_nome'].dropna().unique())
estado_selecionado = st.sidebar.selectbox("Escolha o Estado:", ['Todos'] + estados_disponiveis)

# Filtrar os municípios de acordo com o estado selecionado
if estado_selecionado != 'Todos':
    municipios_disponiveis = sorted(df[df['uf_nome'] == estado_selecionado]['nome_municipio'].dropna().unique())
else:
    municipios_disponiveis = sorted(df['nome_municipio'].dropna().unique())

# Filtro de Município
municipio_selecionado = st.sidebar.selectbox("Escolha o Município:", ['Todos'] + municipios_disponiveis)

# ---------------------------------------------
# Aplicar os filtros de Estado e Município
# ---------------------------------------------
df_filtrado = df.copy()

# Aplicar o filtro de estado
if estado_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['uf_nome'] == estado_selecionado]

# Aplicar o filtro de município
if municipio_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['nome_municipio'] == municipio_selecionado]

# Verificar se há dados após os filtros
if df_filtrado.empty:
    st.warning("Nenhum dado encontrado com os filtros aplicados.")
    st.stop()

# Agrupar os dados filtrados por ano
df_grouped = df_filtrado.groupby('ano_aih').agg({
    'Valor total dos procedimentos': 'sum',
    'Quantidade total de procedimentos': 'sum'
}).reset_index()

# Atualizar a divisão entre treino e teste com os dados filtrados
train = df_grouped[df_grouped['ano_aih'] <= 2023]
test = pd.DataFrame({'ano_aih': [2024, 2025]})

# ---------------------------------------------
# Análise Temporal
# ---------------------------------------------
st.title("Previsão de Custos e Quantidades - 2024 e 2025")
st.markdown("""
Este modelo usa dados históricos de 2019 a 2023 para prever os custos e as quantidades de procedimentos para os anos de 2024 e 2025.
""")

# Exibir os dados agrupados
st.subheader("Dados Agrupados por Ano")
df_grouped_display = df_grouped.copy()
df_grouped_display['Valor total dos procedimentos'] = df_grouped_display['Valor total dos procedimentos'].apply(formatar_real)
df_grouped_display['Quantidade total de procedimentos'] = df_grouped_display['Quantidade total de procedimentos'].apply(formatar_quantidade)
st.dataframe(df_grouped_display)

# Divisão entre treino (2019-2023) e teste (2024-2025)
train = df_grouped[df_grouped['ano_aih'] <= 2023].copy()
train['ano_aih'] = pd.to_datetime(train['ano_aih'], format='%Y')
test = pd.DataFrame({'ano_aih': [2024, 2025]})

# ---------------------------------------------
# Decomposição da Série Temporal
# ---------------------------------------------

# Decomposição da Série Temporal
serie_temporal = train.set_index('ano_aih')['Valor total dos procedimentos']
result = seasonal_decompose(serie_temporal, model='additive', period=1, extrapolate_trend='freq')

# Gráficos de decomposição
st.header("Decomposição da Série Temporal")

fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Série Original
axs[0].plot(result.observed, color="blue", label="Original")
axs[0].set_title("Série Original")
axs[0].set_ylabel("Valores (R$)")
axs[0].legend()

# Tendência
axs[1].plot(result.trend, color="orange", label="Tendência")
axs[1].set_title("Tendência")
axs[1].set_ylabel("Valores (R$)")
axs[1].legend()

# Sazonalidade
axs[2].plot(result.seasonal, color="green", label="Sazonalidade")
axs[2].set_title("Sazonalidade")
axs[2].set_ylabel("Variação Relativa")
axs[2].legend()

# Resíduos
axs[3].plot(result.resid, color="red", label="Resíduo")
axs[3].set_title("Resíduo")
axs[3].set_xlabel("Ano")
axs[3].set_ylabel("Erro ou Desvio")
axs[3].legend()

# Ajustar os rótulos do eixo X para mostrar os anos
axs[3].set_xticks(result.observed.index)
axs[3].set_xticklabels(result.observed.index.year, rotation=45)

plt.tight_layout()
st.pyplot(fig)

# ---------------------------------------------
# Modelo ARIMA para Custos
# ---------------------------------------------
st.header("Score e Previsão para Custos com ARIMA")

try:
    model_arima_custos = ARIMA(train['Valor total dos procedimentos'], order=(1, 1, 1))
    arima_custos_result = model_arima_custos.fit()

    # Previsões para 2024 e 2025
    predicted_train_custos = arima_custos_result.predict(start=0, end=len(train) - 1)
    forecast_arima_custos = arima_custos_result.forecast(steps=2)
    test['Previsão Custos (ARIMA)'] = forecast_arima_custos.values

    # Cálculo das métricas
    mae_custos = mean_absolute_error(train['Valor total dos procedimentos'], predicted_train_custos)
    rmse_custos = calculate_rmse(train['Valor total dos procedimentos'], predicted_train_custos)
    mape_custos = calculate_mape(train['Valor total dos procedimentos'], predicted_train_custos)
    accuracy_custos = calculate_accuracy(train['Valor total dos procedimentos'], predicted_train_custos)

    # Exibir métricas
    st.write(f"MAE: {formatar_real(mae_custos)}")
    st.write(f"RMSE: {formatar_real(rmse_custos)}")
    st.write(f"MAPE: {mape_custos:.2f}%")
    st.write(f"Acurácia: {accuracy_custos:.2f}%")

    # Gráfico
    fig_custos, ax_custos = plt.subplots()
    ax_custos.plot(train['ano_aih'], train['Valor total dos procedimentos'], label="Treino", color="blue")
    ax_custos.plot(test['ano_aih'], test['Previsão Custos (ARIMA)'], label="Previsão", color="orange")
    ax_custos.set_title("Previsão de Custos com ARIMA")
    ax_custos.set_xlabel("Ano")
    ax_custos.set_ylabel("Valor Total (R$)")
    ax_custos.legend()
    st.pyplot(fig_custos)
except Exception as e:
    st.error(f"Erro ao ajustar o modelo ARIMA para custos: {e}")

# ---------------------------------------------
# Modelo ARIMA para Quantidades
# ---------------------------------------------
st.header("Score e Previsão para Quantidades com ARIMA")

try:
    model_arima_quantidades = ARIMA(train['Quantidade total de procedimentos'], order=(1, 1, 1))
    arima_quantidades_result = model_arima_quantidades.fit()

    # Previsões para 2024 e 2025
    predicted_train_quantidades = arima_quantidades_result.predict(start=0, end=len(train) - 1)
    forecast_arima_quantidades = arima_quantidades_result.forecast(steps=2)
    test['Previsão Quantidades (ARIMA)'] = forecast_arima_quantidades.values

    # Cálculo das métricas
    mae_quantidades = mean_absolute_error(train['Quantidade total de procedimentos'], predicted_train_quantidades)
    rmse_quantidades = calculate_rmse(train['Quantidade total de procedimentos'], predicted_train_quantidades)
    mape_quantidades = calculate_mape(train['Quantidade total de procedimentos'], predicted_train_quantidades)
    accuracy_quantidades = calculate_accuracy(train['Quantidade total de procedimentos'], predicted_train_quantidades)

    # Exibir métricas
    st.write(f"MAE: {formatar_quantidade(mae_quantidades)}")
    st.write(f"RMSE: {formatar_quantidade(rmse_quantidades)}")
    st.write(f"MAPE: {mape_quantidades:.2f}%")
    st.write(f"Acurácia: {accuracy_quantidades:.2f}%")

    # Gráfico
    fig_quantidades, ax_quantidades = plt.subplots()
    ax_quantidades.plot(train['ano_aih'], train['Quantidade total de procedimentos'], label="Treino", color="blue")
    ax_quantidades.plot(test['ano_aih'], test['Previsão Quantidades (ARIMA)'], label="Previsão", color="green")
    ax_quantidades.set_title("Previsão de Quantidades com ARIMA")
    ax_quantidades.set_xlabel("Ano")
    ax_quantidades.set_ylabel("Quantidade Total")
    ax_quantidades.legend()
    st.pyplot(fig_quantidades)
except Exception as e:
    st.error(f"Erro ao ajustar o modelo ARIMA para quantidades: {e}")

# ---------------------------------------------
# Comparação de Previsões
# ---------------------------------------------
st.header("Resultados das Previsões para 2024 e 2025")

# Formatar os valores para exibição
test['Previsão Custos (ARIMA)'] = test['Previsão Custos (ARIMA)'].apply(formatar_real)
test['Previsão Quantidades (ARIMA)'] = test['Previsão Quantidades (ARIMA)'].apply(formatar_quantidade)

# Exibir tabela com previsões
st.table(test)

# ---------------------------------------------------------------------------------------------------


# ---------------------------------------------
# Modelo SARIMA para Custos
# ---------------------------------------------
st.header("Score e Previsão para Custos com SARIMA")

try:
    # Ajustar modelo SARIMA para custos
    model_sarima_custos = SARIMAX(
        train['Valor total dos procedimentos'],
        order=(1, 1, 1),  # p, d, q
        seasonal_order=(1, 1, 1, 12),  # P, D, Q, m (m=12 meses, sazonalidade anual)
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_custos_result = model_sarima_custos.fit(disp=False)

    # Previsões e métricas para custos
    predicted_train_sarima_custos = sarima_custos_result.predict(start=0, end=len(train) - 1)
    forecast_sarima_custos = sarima_custos_result.get_forecast(steps=2).predicted_mean
    test['Previsão Custos (SARIMA)'] = forecast_sarima_custos.values

    mae_sarima_custos = mean_absolute_error(train['Valor total dos procedimentos'], predicted_train_sarima_custos)
    rmse_sarima_custos = calculate_rmse(train['Valor total dos procedimentos'], predicted_train_sarima_custos)
    mape_sarima_custos = calculate_mape(train['Valor total dos procedimentos'], predicted_train_sarima_custos)
    accuracy_sarima_custos = calculate_accuracy(train['Valor total dos procedimentos'], predicted_train_sarima_custos)

    # Exibir métricas para custos
    st.write(f"MAE (SARIMA): {formatar_real(mae_sarima_custos)}")
    st.write(f"RMSE (SARIMA): {formatar_real(rmse_sarima_custos)}")
    st.write(f"MAPE (SARIMA): {mape_sarima_custos:.2f}%")
    st.write(f"Acurácia (SARIMA): {accuracy_sarima_custos:.2f}%")

    # Gráfico para o modelo SARIMA - Custos
    fig_sarima_custos, ax_sarima_custos = plt.subplots()
    ax_sarima_custos.plot(train['ano_aih'], train['Valor total dos procedimentos'], label="Treino", color="blue")
    ax_sarima_custos.plot(test['ano_aih'], test['Previsão Custos (SARIMA)'], label="Previsão", color="orange")
    ax_sarima_custos.set_title("Previsão de Custos com SARIMA")
    ax_sarima_custos.set_xlabel("Ano")
    ax_sarima_custos.set_ylabel("Valor Total (R$)")
    ax_sarima_custos.legend()
    st.pyplot(fig_sarima_custos)
except Exception as e:
    st.error(f"Erro ao ajustar o modelo SARIMA para custos: {e}")

# ---------------------------------------------
# Modelo SARIMA para Quantidades
# ---------------------------------------------
st.header("Score e Previsão para Quantidades com SARIMA")

try:
    # Ajustar modelo SARIMA para quantidades
    model_sarima_quantidades = SARIMAX(
        train['Quantidade total de procedimentos'],
        order=(1, 1, 1),  # p, d, q
        seasonal_order=(1, 1, 1, 12),  # P, D, Q, m (m=12 meses, sazonalidade anual)
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_quantidades_result = model_sarima_quantidades.fit(disp=False)

    # # Previsões e métricas para quantidades
    predicted_train_sarima_quantidades = sarima_quantidades_result.predict(start=train.index[0], end=train.index[-1])
    forecast_sarima_quantidades = sarima_quantidades_result.get_forecast(steps=2).predicted_mean
    test['Previsão Quantidades (SARIMA)'] = forecast_sarima_quantidades.values

    mae_sarima_quantidades = mean_absolute_error(train['Quantidade total de procedimentos'], predicted_train_sarima_quantidades)
    rmse_sarima_quantidades = calculate_rmse(train['Quantidade total de procedimentos'], predicted_train_sarima_quantidades)
    mape_sarima_quantidades = calculate_mape(train['Quantidade total de procedimentos'], predicted_train_sarima_quantidades)
    accuracy_sarima_quantidades = calculate_accuracy(train['Quantidade total de procedimentos'], predicted_train_sarima_quantidades)

    # Exibir métricas para quantidades
    st.write(f"MAE (SARIMA): {formatar_quantidade(mae_sarima_quantidades)}")
    st.write(f"RMSE (SARIMA): {formatar_quantidade(rmse_sarima_quantidades)}")
    st.write(f"MAPE (SARIMA): {mape_sarima_quantidades:.2f}%")
    st.write(f"Acurácia (SARIMA): {accuracy_sarima_quantidades:.2f}%")

    # Gráfico para o modelo SARIMA - Quantidades
    fig_sarima_quantidades, ax_sarima_quantidades = plt.subplots()
    ax_sarima_quantidades.plot(train['ano_aih'], train['Quantidade total de procedimentos'], label="Treino", color="blue")
    ax_sarima_quantidades.plot(test['ano_aih'], test['Previsão Quantidades (SARIMA)'], label="Previsão", color="green")
    ax_sarima_quantidades.set_title("Previsão de Quantidades com SARIMA")
    ax_sarima_quantidades.set_xlabel("Ano")
    ax_sarima_quantidades.set_ylabel("Quantidade Total")
    ax_sarima_quantidades.legend()
    st.pyplot(fig_sarima_quantidades)
except Exception as e:
    st.error(f"Erro ao ajustar o modelo SARIMA para quantidades: {e}")

# ---------------------------------------------
# Comparação de Previsões - ARIMA x SARIMA
# ---------------------------------------------
st.header("Comparação de Previsões para 2024 e 2025")

# Comparar previsões ARIMA e SARIMA
test_display = test.copy()
test_display['Previsão Custos (SARIMA)'] = test_display['Previsão Custos (SARIMA)'].apply(formatar_real)
test_display['Previsão Quantidades (SARIMA)'] = test_display['Previsão Quantidades (SARIMA)'].apply(formatar_quantidade)

# Exibir tabela com previsões
st.table(test_display)