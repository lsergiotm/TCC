import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from data_processing import load_data

# Carregar os dados processados
df = load_data()

# Garantir que as colunas de custo e quantidade sejam numéricas
df['Valor total dos procedimentos'] = pd.to_numeric(
    df['Valor total dos procedimentos'], errors='coerce'
).fillna(0)
df['Quantidade total de procedimentos'] = pd.to_numeric(
    df['Quantidade total de procedimentos'], errors='coerce'
).fillna(0)

# Título da página
st.title("Análise Temporal e Sazonal com Regressão Linear")
st.markdown("""
Explore a evolução dos custos e quantidades de procedimentos ao longo do tempo e utilize a regressão linear para prever relações entre os custos e quantidades.
""")

# ---------------------------------------------
# Filtros de Dados
# ---------------------------------------------
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

# Filtro de Ano
anos = list(df['ano_aih'].unique())
ano_selecionado = st.sidebar.selectbox("Escolha o ano:", ['Todos'] + anos)

# Filtrar por ano
if ano_selecionado != 'Todos':
    df = df[df['ano_aih'] == ano_selecionado]

# Filtro de Mês
meses = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}
mes_selecionado = st.sidebar.selectbox("Escolha o mês:", ['Todos'] + list(meses.values()))

# Filtrar por mês
if mes_selecionado != 'Todos':
    mes_numero = list(meses.keys())[list(meses.values()).index(mes_selecionado)]
    df = df[df['mes_aih'] == mes_numero]

# ---------------------------------------------
# Regressão Linear - Preparação dos Dados
# ---------------------------------------------
st.subheader("Regressão Linear - Custos Médios vs Quantidades Médias")

# Agrupar dados por mês para custos e quantidades
if not df.empty:
    regression_data = df.groupby('mes_aih').agg({
        'Valor total dos procedimentos': 'mean',
        'Quantidade total de procedimentos': 'mean'
    }).reset_index()

    # Renomear as colunas para facilitar a visualização
    regression_data.rename(columns={
        'Valor total dos procedimentos': 'Custo Médio',
        'Quantidade total de procedimentos': 'Quantidade Média'
    }, inplace=True)

    # Regressão Linear
    X = regression_data[['Quantidade Média']]
    y = regression_data['Custo Médio']
    model = LinearRegression()
    model.fit(X, y)

    # Obter coeficientes e intercepto
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, model.predict(X))

    # Exibir a equação da regressão linear
    st.latex(f"Custo = {slope:.2f} \\times Quantidade + {intercept:.2f}")
    st.write(f"**Coeficiente de Determinação (R²): {r2:.2f}**")

    # ---------------------------------------------
    # Visualização da Regressão Linear
    # ---------------------------------------------
    st.subheader("Visualização da Regressão Linear")

    # Adicionar a linha de regressão aos dados
    regression_data['Custo Previsto'] = model.predict(X)

    fig_regressao = go.Figure()
    fig_regressao.add_trace(go.Scatter(
        x=regression_data['Quantidade Média'],
        y=regression_data['Custo Médio'],
        mode='markers',
        name='Custos Reais',
        marker=dict(size=8, color='blue')
    ))
    fig_regressao.add_trace(go.Scatter(
        x=regression_data['Quantidade Média'],
        y=regression_data['Custo Previsto'],
        mode='lines',
        name='Linha de Regressão',
        line=dict(color='red')
    ))
    fig_regressao.update_layout(
        title="Regressão Linear - Custos Médios vs Quantidades Médias",
        xaxis_title="Quantidade Média",
        yaxis_title="Custo Médio (R$)",
        legend_title="Legenda"
    )
    st.plotly_chart(fig_regressao)

    # ---------------------------------------------
    # Previsões Baseadas no Modelo
    # ---------------------------------------------
    st.subheader("Previsões Baseadas na Regressão")

    # Intervalo observado nos dados
    min_quantidade = regression_data['Quantidade Média'].min()
    max_quantidade = regression_data['Quantidade Média'].max()

    # Entrada do usuário para novas quantidades
    nova_quantidade = st.number_input(
        "Insira uma nova quantidade média para prever o custo médio (R$):",
        min_value=0, step=100
    )

    # Fazer a previsão
    if nova_quantidade > 0:
        if min_quantidade <= nova_quantidade <= max_quantidade:
            previsao = model.predict([[nova_quantidade]])
            st.write(f"**Para uma quantidade média de {nova_quantidade}, o custo médio previsto é: R$ {previsao[0]:,.2f}**")
        else:
            st.warning(f"A quantidade inserida ({nova_quantidade}) está fora do intervalo observado nos dados ({min_quantidade:.0f} a {max_quantidade:.0f}). "
                       f"As previsões podem não ser confiáveis.")

    # ---------------------------------------------
    # Avaliação do Modelo
    # ---------------------------------------------
    st.subheader("Avaliação do Modelo")

    # Análise baseada no coeficiente R²
    if r2 > 0.8:
        st.success("O modelo apresenta um **ótimo desempenho**!")
        st.write("""
        O coeficiente de determinação (R²) indica que a maioria da variabilidade nos custos médios 
        é explicada pela quantidade média de procedimentos. O modelo pode ser usado para previsões confiáveis.
        """)
    elif 0.5 < r2 <= 0.8:
        st.info("O modelo apresenta um **desempenho razoável**.")
        st.write("""
        O coeficiente de determinação (R²) mostra que parte significativa da variabilidade nos custos 
        é explicada pela quantidade média de procedimentos, mas outros fatores não considerados podem estar impactando o modelo.
        Considere adicionar mais variáveis (exemplo: região, tipo de procedimento) para melhorar o desempenho.
        """)
    else:
        st.warning("O modelo apresenta um **desempenho fraco**.")
        st.write("""
        O coeficiente de determinação (R²) indica que o modelo explica pouco da variabilidade nos custos médios. 
        Isso pode ocorrer devido à ausência de variáveis importantes ou pela relação não ser linear.
        Considere revisar os dados ou aplicar um modelo mais robusto, como regressão polinomial ou aprendizado de máquina.
        """)

else:
    st.error("Não há dados suficientes para realizar a análise. Verifique os filtros selecionados.")
