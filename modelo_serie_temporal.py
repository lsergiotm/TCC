import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_processing import load_data
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Função para calcular o RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# Carregar os dados processados
df = load_data()

# Garantir que os dados necessários estão em formato numérico
df['Valor total dos procedimentos'] = pd.to_numeric(df['Valor total dos procedimentos'], errors='coerce').fillna(0)

# ---------------------------------------------
# Filtros de Ano, Estado e Município
# ---------------------------------------------
st.sidebar.header("Filtros de Dados")

# Filtro de Ano
anos_disponiveis = sorted(df['ano_aih'].dropna().unique())
ano_selecionado = st.sidebar.selectbox("Escolha o Ano:", ['Todos'] + anos_disponiveis)

# Filtro de Estado
estados_disponiveis = sorted(df['uf_nome'].dropna().unique())
estado_selecionado = st.sidebar.selectbox("Escolha o Estado:", ['Todos'] + estados_disponiveis)

# Filtrar os municípios de acordo com o estado selecionado
if estado_selecionado != 'Todos':
    municipios_disponiveis = sorted(df[df['uf_nome'] == estado_selecionado]['nome_municipio'].dropna().unique())
else:
    municipios_disponiveis = sorted(df['nome_municipio'].dropna().unique())

municipio_selecionado = st.sidebar.selectbox("Escolha o Município:", ['Todos'] + municipios_disponiveis)

# Aplicar os filtros
df_filtrado = df.copy()

if ano_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['ano_aih'] == ano_selecionado]

if estado_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['uf_nome'] == estado_selecionado]

if municipio_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['nome_municipio'] == municipio_selecionado]

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado com os filtros aplicados.")
    st.stop()

# Agrupar os dados filtrados por mês
df_grouped = df_filtrado.groupby('mes_aih')['Valor total dos procedimentos'].sum().reset_index()

# ---------------------------------------------
# Análise Temporal
# ---------------------------------------------
st.title("Análise de Série Temporal - Validação e Ajuste")
st.markdown("""
Explore as séries temporais com filtros interativos e avalie a qualidade do modelo usando métricas estatísticas.
""")

# Exibir a série temporal original
st.subheader("Série Temporal - Dados Filtrados")
st.line_chart(df_grouped.set_index("mes_aih"))

# ---------------------------------------------
# Estacionariedade
# ---------------------------------------------
st.header("1. Estacionariedade")
adf_test = adfuller(df_grouped["Valor total dos procedimentos"])
p_value_adf = adf_test[1]

if p_value_adf < 0.05:
    st.success(f"A série é estacionária (p-valor = {p_value_adf:.4f}).")
else:
    st.warning(f"A série não é estacionária (p-valor = {p_value_adf:.4f}). Considere transformações, como diferenciação.")

# ---------------------------------------------
# Autocorrelação
# ---------------------------------------------
st.header("2. Autocorrelação")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Limitar nlags ao máximo permitido
max_lags = len(df_grouped) // 2  # 50% do tamanho da amostra
acf_vals = acf(df_grouped["Valor total dos procedimentos"], nlags=max_lags)
pacf_vals = pacf(df_grouped["Valor total dos procedimentos"], nlags=max_lags)

ax[0].stem(range(len(acf_vals)), acf_vals, linefmt='b-', basefmt=" ")
ax[0].set_title("Autocorrelação (ACF)")
ax[1].stem(range(len(pacf_vals)), pacf_vals, linefmt='r-', basefmt=" ")
ax[1].set_title("Autocorrelação Parcial (PACF)")

st.pyplot(fig)

# ---------------------------------------------
# Validação com Dados Fora da Amostra
# ---------------------------------------------
st.header("3. Validação com Dados Fora da Amostra")

# Divisão entre treino e teste
train_size = int(len(df_grouped) * 0.75)
train, test = df_grouped.iloc[:train_size], df_grouped.iloc[train_size:]

# Modelo de previsão simples (média do treino)
predicted_test = [train["Valor total dos procedimentos"].mean()] * len(test)

# Métricas de avaliação
mae = mean_absolute_error(test["Valor total dos procedimentos"], predicted_test)
rmse = calculate_rmse(test["Valor total dos procedimentos"], predicted_test)

# Exibição
st.write(f"Tamanho do conjunto de treino: {len(train)}")
st.write(f"Tamanho do conjunto de teste: {len(test)}")

fig, ax = plt.subplots()
ax.plot(train["mes_aih"], train["Valor total dos procedimentos"], label="Treino", color="blue")
ax.plot(test["mes_aih"], test["Valor total dos procedimentos"], label="Teste", color="orange")
ax.legend()
ax.set_title("Divisão Treino/Teste")
ax.set_xlabel("Mês")
ax.set_ylabel("Valor Total dos Procedimentos")
st.pyplot(fig)

# ---------------------------------------------
# Ajustes Adicionais - Transformação para Estacionariedade
# ---------------------------------------------
if p_value_adf >= 0.05:
    st.subheader("Ajustes na Série Temporal")
    # Aplicar diferenciação
    df_grouped["Diferenciado"] = df_grouped["Valor total dos procedimentos"].diff()
    df_grouped = df_grouped.dropna()

    # Recalcular ADF após diferenciação
    adf_test_diff = adfuller(df_grouped["Diferenciado"])
    p_value_diff = adf_test_diff[1]

    st.write(f"Após diferenciação, o p-valor do ADF é: {p_value_diff:.4f}")
    if p_value_diff < 0.05:
        st.success("A série agora é estacionária.")
    else:
        st.error("Mesmo após a diferenciação, a série não é estacionária.")

    # Exibir a série ajustada
    st.line_chart(df_grouped[["mes_aih", "Diferenciado"]].set_index("mes_aih"))

    # Ajustes na Série Temporal - Gráfico de Treino e Teste
    st.subheader("Treino e Teste Após Transformação para Estacionariedade")

    # Divisão entre treino e teste para série diferenciada
    train_size_diff = int(len(df_grouped) * 0.75)
    train_diff, test_diff = df_grouped["Diferenciado"][:train_size_diff], df_grouped["Diferenciado"][train_size_diff:]

    # Previsão simples para a série diferenciada (média do treino)
    predicted_test_diff = [train_diff.mean()] * len(test_diff)

    # Gráfico para treino e teste na série diferenciada
    fig_diff, ax_diff = plt.subplots()
    ax_diff.plot(train_diff.index, train_diff, label="Treino", color="blue")
    ax_diff.plot(test_diff.index, test_diff, label="Teste", color="orange")
    ax_diff.legend()
    ax_diff.set_title("Treino e Teste na Série Transformada")
    ax_diff.set_xlabel("Mês")
    ax_diff.set_ylabel("Valores Diferenciados")
    st.pyplot(fig_diff)

    # Comparação de Métricas Antes e Depois da Transformação
    st.subheader("Comparação de Métricas - Antes e Depois da Transformação")

    # Cálculo das métricas para a série diferenciada
    mae_diff = mean_absolute_error(test_diff, predicted_test_diff)
    rmse_diff = calculate_rmse(test_diff, predicted_test_diff)

    # Criação da tabela comparativa
    comparacao_metrica = pd.DataFrame({
        "Métrica": ["Erro Absoluto Médio (MAE)", "Raiz do Erro Quadrático Médio (RMSE)"],
        "Antes da Transformação": [f"R$ {mae:,.2f}", f"R$ {rmse:,.2f}"],
        "Depois da Transformação": [f"R$ {mae_diff:,.2f}", f"R$ {rmse_diff:,.2f}"]
    })

    st.table(comparacao_metrica)

    # Avaliação final após transformação
    if mae_diff < mae and rmse_diff < rmse:
        st.success("O modelo melhorou após a transformação para estacionariedade!")
    else:
        st.warning("A transformação não resultou em melhorias significativas. Considere técnicas adicionais.")

##################################################################################################################

# ---------------------------------------------
# Modelo ARIMA
# ---------------------------------------------
st.header("Modelo ARIMA")

# Determinar o valor de p, d, q para o ARIMA
p = 1  # Número de lags na autoregressão
d = 1  # Número de diferenciações (já aplicamos 1)
q = 1  # Número de termos na média móvel

# Ajustar o modelo ARIMA
try:
    model_arima = ARIMA(df_grouped["Valor total dos procedimentos"], order=(p, d, q))
    arima_result = model_arima.fit()

    # Previsões
    forecast_arima = arima_result.forecast(steps=len(test))
    mae_arima = mean_absolute_error(test["Valor total dos procedimentos"], forecast_arima)
    rmse_arima = calculate_rmse(test["Valor total dos procedimentos"], forecast_arima)

    # Exibir métricas do modelo ARIMA
    st.subheader("Resultados do Modelo ARIMA")
    st.write(f"MAE: R$ {mae_arima:,.2f}")
    st.write(f"RMSE: R$ {rmse_arima:,.2f}")

    # Gráfico de Previsão do ARIMA
    fig_arima, ax_arima = plt.subplots()
    ax_arima.plot(df_grouped["mes_aih"], df_grouped["Valor total dos procedimentos"], label="Original")
    ax_arima.plot(test["mes_aih"], forecast_arima, label="Previsão (ARIMA)", color="orange")
    ax_arima.legend()
    ax_arima.set_title("Previsão com ARIMA")
    ax_arima.set_xlabel("Mês")
    ax_arima.set_ylabel("Valor Total dos Procedimentos")
    st.pyplot(fig_arima)
except Exception as e:
    st.error(f"Erro ao ajustar o modelo ARIMA: {e}")

# ---------------------------------------------
# Modelo SARIMA
# ---------------------------------------------
st.header("Modelo SARIMA")

# Determinar os valores de (p, d, q) e os sazonais (P, D, Q, m)
P = 1  # Sazonal autoregressivo
D = 1  # Diferenciação sazonal
Q = 1  # Sazonal média móvel
m = 12  # Sazonalidade anual (12 meses)

# Ajustar o modelo SARIMA
try:
    model_sarima = SARIMAX(
        df_grouped["Valor total dos procedimentos"],
        order=(p, d, q),
        seasonal_order=(P, D, Q, m)
    )
    sarima_result = model_sarima.fit(disp=False)

    # Previsões
    forecast_sarima = sarima_result.forecast(steps=len(test))
    mae_sarima = mean_absolute_error(test["Valor total dos procedimentos"], forecast_sarima)
    rmse_sarima = calculate_rmse(test["Valor total dos procedimentos"], forecast_sarima)

    # Exibir métricas do modelo SARIMA
    st.subheader("Resultados do Modelo SARIMA")
    st.write(f"MAE: R$ {mae_sarima:,.2f}")
    st.write(f"RMSE: R$ {rmse_sarima:,.2f}")

    # Gráfico de Previsão do SARIMA
    fig_sarima, ax_sarima = plt.subplots()
    ax_sarima.plot(df_grouped["mes_aih"], df_grouped["Valor total dos procedimentos"], label="Original")
    ax_sarima.plot(test["mes_aih"], forecast_sarima, label="Previsão (SARIMA)", color="green")
    ax_sarima.legend()
    ax_sarima.set_title("Previsão com SARIMA")
    ax_sarima.set_xlabel("Mês")
    ax_sarima.set_ylabel("Valor Total dos Procedimentos")
    st.pyplot(fig_sarima)
except Exception as e:
    st.error(f"Erro ao ajustar o modelo SARIMA: {e}")

# ---------------------------------------------
# Comparação de Modelos
st.header("Comparação de Modelos")

# Criação da tabela comparativa com os valores tratados
comparacao_modelos = pd.DataFrame({
    "Modelo": ["Antes da Transformação", "ARIMA", "SARIMA"],
    "MAE": [
        mae if 'mae' in locals() else float('nan'),
        mae_arima if 'mae_arima' in locals() else float('nan'),
        mae_sarima if 'mae_sarima' in locals() else float('nan')
    ],
    "RMSE": [
        rmse if 'rmse' in locals() else float('nan'),
        rmse_arima if 'rmse_arima' in locals() else float('nan'),
        rmse_sarima if 'rmse_sarima' in locals() else float('nan')
    ]
})

# Formatar a tabela para exibição
comparacao_modelos['MAE'] = comparacao_modelos['MAE'].apply(lambda x: f"R$ {x:,.2f}" if not pd.isna(x) else "N/A")
comparacao_modelos['RMSE'] = comparacao_modelos['RMSE'].apply(lambda x: f"R$ {x:,.2f}" if not pd.isna(x) else "N/A")

# Exibir a tabela comparativa
st.table(comparacao_modelos)

# Seleção do melhor modelo com base no RMSE
melhor_modelo_idx = comparacao_modelos['RMSE'].apply(
    lambda x: float(x.replace("R$", "").replace(",", "")) if "R$" in x else float('inf')
).idxmin()
melhor_modelo = comparacao_modelos.iloc[melhor_modelo_idx]["Modelo"]

# Exibir o melhor modelo
st.success(f"O melhor modelo com base no RMSE é: **{melhor_modelo}**")

# ---------------------------------------------
# Tabela Informativa de Critérios
# ---------------------------------------------
st.header("Critérios de Aceitação dos Modelos")

criterios = pd.DataFrame({
    "Métrica": ["MAE (Erro Absoluto Médio)", "RMSE (Raiz do Erro Quadrático Médio)"],
    "Critério": [
        "Bom se < R$ 10.000.000,00",
        "Bom se < R$ 10.000.000,00"
    ],
    "Interpretação": [
        "Erro médio absoluto dos valores previstos em relação aos reais. Valores baixos indicam boa precisão.",
        "Raiz quadrada do erro médio quadrático. Valores baixos indicam melhor ajuste do modelo."
    ]
})

st.table(criterios)

# ---------------------------------------------
# Avaliação dos Modelos
# ---------------------------------------------
st.header("Avaliação Detalhada dos Modelos")

# Adiciona a avaliação com base nos critérios
avaliacao_modelos = comparacao_modelos.copy()
avaliacao_modelos["MAE Aceitação"] = comparacao_modelos["MAE"].apply(
    lambda x: "Bom" if "R$" in x and float(x.replace("R$", "").replace(",", "")) < 10_000_000 else "Ruim"
)
avaliacao_modelos["RMSE Aceitação"] = comparacao_modelos["RMSE"].apply(
    lambda x: "Bom" if "R$" in x and float(x.replace("R$", "").replace(",", "")) < 10_000_000 else "Ruim"
)

# Exibir a tabela de avaliação
st.table(avaliacao_modelos)

# Mensagem final
if (avaliacao_modelos["MAE Aceitação"] == "Bom").all() and (avaliacao_modelos["RMSE Aceitação"] == "Bom").all():
    st.success("Todos os modelos atendem aos critérios de aceitação.")
else:
    st.warning("Nem todos os modelos atendem aos critérios. Considere revisar o modelo com melhor desempenho.")
